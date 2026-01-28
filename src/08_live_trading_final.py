import time
import hmac
import hashlib
import requests
import logging
import pandas as pd
import pandas_ta as ta
import numpy as np
import os
import json
import sys
import csv
from datetime import datetime
from stable_baselines3 import PPO
from tabulate import tabulate
from colorama import init, Fore, Back, Style
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

init(autoreset=True)

# --- 1. CONFIGURATION ---
BASE_URL = "https://cdn-ind.testnet.deltaex.org"
API_KEY = "6cWxkYDq6xydqkqsHP0LHnYOjeXNHL"       # <--- ENTER KEY
API_SECRET = "j23ZPsTrZJgMZmLuZyJGAONJsVN1swdZbNDFRhmXwEnIJK6gJgEioP3rPjsQ" # <--- ENTER SECRET
SYMBOL = "BTCUSD"
MODEL_PATH = os.path.join("models", "best_model.zip")

# OUTPUT PATHS
LOG_DIR = 'logs'
TRADES_CSV_PATH = os.path.join(LOG_DIR, 'trades.csv')
PERF_CSV_PATH = os.path.join(LOG_DIR, 'performance.csv')
PNL_IMG_PATH = os.path.join(LOG_DIR, 'live_pnl.png')

# RISK SETTINGS
TRADE_SIZE = 1 
INITIAL_STOP_LOSS_PCT = 0.0015 
TAKE_PROFIT_PCT = 0.0015 
MAX_LEVERAGE_CAP = 50
MAX_HOLD_MINUTES = 30
MAX_DRAWDOWN_PCT = 0.20 

# HYBRID LOGIC SETTINGS
ENABLE_HYBRID_OVERRIDES = True 
RSI_OVERBOUGHT = 70            
RSI_OVERSOLD = 30              

# --- 2. SESSION & ACCOUNT TRACKER ---
class LocalAccount:
    def __init__(self, initial_balance=10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.peak_balance = initial_balance
        self.equity_history = [initial_balance]
        self.drawdown_history = [0.0]
        self.time_history = [datetime.now()]
        self.trades = [] 
        
    def update_pnl(self, pnl_amount):
        self.balance += pnl_amount
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        
        drawdown = 0.0
        if self.peak_balance > 0:
            drawdown = (self.peak_balance - self.balance) / self.peak_balance
            
        self.trades.append(pnl_amount)
        self.equity_history.append(self.balance)
        self.drawdown_history.append(drawdown)
        self.time_history.append(datetime.now())
        
        return drawdown

    def get_metrics(self):
        total_trades = len(self.trades)
        if total_trades == 0:
            return 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0

        wins = [t for t in self.trades if t > 0]
        losses = [t for t in self.trades if t <= 0]
        
        win_count = len(wins)
        win_rate = (win_count / total_trades * 100)
        
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        
        # Profit Factor
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0
        
        # Sharpe (Simplified)
        arr = np.array(self.trades)
        std = np.std(arr)
        mean = np.mean(arr)
        sharpe = (mean / std) * np.sqrt(total_trades) if std != 0 else 0.0
            
        return total_trades, win_count, win_rate, sharpe, avg_win, avg_loss, profit_factor

account = LocalAccount(initial_balance=10000.0)
CACHED_PRODUCT_ID = None

# --- 3. LOGGING HELPERS ---
def log_trade_to_csv(entry_time, side, entry_price, exit_time, exit_price, pnl, reason):
    file_exists = os.path.isfile(TRADES_CSV_PATH)
    try:
        with open(TRADES_CSV_PATH, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Entry Time', 'Side', 'Entry Price', 'Exit Time', 'Exit Price', 'PnL', 'Reason'])
            writer.writerow([entry_time, side, entry_price, exit_time, exit_price, f"{pnl:.2f}", reason])
    except Exception as e:
        logging.error(f"Trades CSV Error: {e}")

def log_performance_to_csv(balance, drawdown, account_obj=None, final=False):
    file_exists = os.path.isfile(PERF_CSV_PATH)
    try:
        with open(PERF_CSV_PATH, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'Timestamp', 'Balance', 'Drawdown_Pct', 'Equity', 'Net Profit',
                    'Total Trades', 'Wins', 'Win Rate', 'Sharpe', 'Avg Win', 'Avg Loss', 'Profit Factor', 'Max Drawdown', 'Final'
                ])
            # Get metrics if account_obj is provided
            if account_obj:
                total_trades, wins, win_rate, sharpe, avg_win, avg_loss, pf = account_obj.get_metrics()
                max_dd = max(account_obj.drawdown_history) * 100 if account_obj.drawdown_history else 0.0
            else:
                total_trades = wins = win_rate = sharpe = avg_win = avg_loss = pf = max_dd = 0
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                f"{balance:.2f}",
                f"{drawdown*100:.2f}",
                f"{balance * (1 - drawdown):.2f}",
                f"{balance - (balance * (1 - drawdown)):.2f}",
                total_trades, wins, f"{win_rate:.1f}", f"{sharpe:.2f}", f"{avg_win:.2f}", f"{avg_loss:.2f}", f"{pf:.2f}", f"{max_dd:.2f}",
                'YES' if final else ''
            ])
    except Exception as e:
        logging.error(f"Perf CSV Error: {e}")

# --- 4. API FUNCTIONS (ROBUST) ---
def generate_signature(method, timestamp, path, query_string, payload):
    data = method + timestamp + path + query_string + payload
    return hmac.new(API_SECRET.encode(), data.encode(), hashlib.sha256).hexdigest()

def send_api_request(method, path, payload=None, query_params=None):
    url = f"{BASE_URL}{path}"
    query_string = ""
    if query_params:
        q_parts = [f"{k}={v}" for k, v in query_params.items()]
        query_string = "?" + "&".join(q_parts)
        url += query_string

    payload_str = json.dumps(payload) if payload else ""
    
    for attempt in range(3):
        try:
            ts = str(int(time.time()))
            sig = generate_signature(method, ts, path, query_string, payload_str)
            headers = {"api-key": API_KEY, "timestamp": ts, "signature": sig, "Content-Type": "application/json"}
            
            if method == "GET":
                response = requests.get(url, headers=headers, timeout=10)
            else:
                response = requests.post(url, headers=headers, data=payload_str, timeout=10)
            
            if response.status_code in [502, 504, 500]:
                time.sleep(2 * (attempt + 1))
                continue
            
            try:
                return response.json()
            except json.JSONDecodeError:
                return None
        except:
            time.sleep(2)
    return None

def get_product_id():
    global CACHED_PRODUCT_ID
    if CACHED_PRODUCT_ID: return CACHED_PRODUCT_ID
    data = send_api_request("GET", "/v2/products")
    if data and data.get('success'):
        for p in data['result']:
            if p['symbol'] == SYMBOL:
                CACHED_PRODUCT_ID = p['id']
                return p['id']
    return "84" 

def set_leverage(leverage):
    pid = get_product_id()
    payload = {"product_id": int(pid), "leverage": str(int(leverage))}
    send_api_request("POST", "/v2/orders/leverage", payload=payload)

def place_order(side, size, sl_price=None, tp_price=None, reduce_only=False):
    pid = get_product_id()
    if size <= 0: return False

    payload = {
        "product_id": int(pid), 
        "limit_price": "0", 
        "size": int(size),
        "side": side, 
        "order_type": "market_order", 
        "reduce_only": reduce_only
    }
    
    if not reduce_only and sl_price and tp_price:
        payload["bracket_stop_loss_price"] = str(int(sl_price))
        payload["bracket_take_profit_price"] = str(int(tp_price))

    res = send_api_request("POST", "/v2/orders", payload=payload)
    
    if res and res.get('success'):
        return True
    elif res:
        err_code = res.get('error', {}).get('code')
        if err_code == 'bracket_order_position_exists':
            return "POSITION_EXISTS"
        logging.error(f"Order Failed: {res}")
        return False
    return False

def get_live_data():
    end = int(time.time())
    start = end - 6000 
    params = {"symbol": SYMBOL, "resolution": "1m", "start": str(start), "end": str(end)}
    res = send_api_request("GET", "/v2/history/candles", query_params=params)
    
    if not res or not res.get('success'): return None, 0, 0, 0, None

    try:
        df = pd.DataFrame(res['result'])
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        df.ta.bbands(length=20, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.obv(append=True)
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        t = df['hour'] * 60 + df['minute']
        df['time_sin'] = np.sin(2 * np.pi * t / 1440)
        df['time_cos'] = np.cos(2 * np.pi * t / 1440)
        df['funding_rate'] = 0.0 
        
        df.fillna(0, inplace=True)

        cols = ['log_ret', 'RSI_14', 'ATRr_14', 'OBV', 'MACD_12_26_9', 'MACDh_12_26_9', 
                'BBU_20_2.0_2.0', 'BBL_20_2.0_2.0', 'time_sin', 'time_cos', 'funding_rate']
        
        for c in cols:
            if c not in df.columns: df[c] = 0.0
        
        obs = df.iloc[-1][cols].values.reshape(1, -1)
        return obs, df.iloc[-1]['close'], df.iloc[-1]['ATRr_14'], df.iloc[-1]['RSI_14'], df
    except Exception as e:
        logging.error(f"Data Process Error: {e}")
        return None, 0, 0, 0, None

def calculate_dynamic_leverage(atr, price):
    if price == 0: return 1
    vol_pct = (atr / price) * 100
    lev = int(50 / (vol_pct * 5)) if vol_pct > 0 else 10
    return max(1, min(lev, MAX_LEVERAGE_CAP))

def get_position_details():
    res = send_api_request("GET", "/v2/positions")
    pid = get_product_id()
    
    if res and res.get('success'):
        result_data = res.get('result', [])
        
        def parse(p):
            return {
                'size': int(p.get('size', 0)),
                'entry_price': float(p.get('entry_price', 0)),
                'u_pnl': float(p.get('unrealized_pnl', 0))
            }

        if isinstance(result_data, list):
            for pos in result_data:
                if str(pos.get('product_id')) == str(pid): return parse(pos)
        elif isinstance(result_data, dict):
            if str(result_data.get('product_id')) == str(pid): return parse(result_data)
        
        return {'size': 0, 'entry_price': 0.0, 'u_pnl': 0.0}
    return None

def get_wallet_balance():
    res = send_api_request("GET", "/v2/wallet/balances")
    if res and res.get('success'):
        for asset in res.get('result', []):
            if asset['asset_symbol'] in ['USDT', 'DET']: return float(asset['total_balance'])
    return 0.0

# --- 5. VISUALIZATION ---
class LiveDashboard:
    def __init__(self):
        try:
            plt.ion()
            self.fig, self.axes = plt.subplots(2, 1, figsize=(10, 8))
            self.fig.canvas.manager.set_window_title(f'Live Trading Bot - {SYMBOL}')
        except:
            self.fig = None

    def update(self, df, account_obj):
        if self.fig is None: return
        try:
            ax_pnl, ax_dd = self.axes
            ax_pnl.clear(); ax_dd.clear()

            # PnL
            if len(account_obj.equity_history) > 0:
                ax_pnl.plot(account_obj.time_history, account_obj.equity_history, color='green', linewidth=2)
                ax_pnl.set_title("Live PnL Curve (Balance)")
                ax_pnl.grid(True, alpha=0.3)
                ax_pnl.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

            # Drawdown
            if len(account_obj.drawdown_history) > 0:
                dd_pct = [d * 100 for d in account_obj.drawdown_history]
                ax_dd.plot(account_obj.time_history, dd_pct, color='red', linewidth=2)
                ax_dd.fill_between(account_obj.time_history, dd_pct, 0, color='red', alpha=0.1)
                ax_dd.set_title("Live Drawdown (%)")
                ax_dd.grid(True, alpha=0.3)
                ax_dd.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

            plt.tight_layout()
            plt.draw()
            plt.pause(0.1)
            try: plt.savefig(PNL_IMG_PATH)
            except: pass
        except: pass

def print_dashboard(price, rsi, action, pos_state, pnl_live, margin, atr, leverage, account_obj, duration):
    os.system('cls' if os.name == 'nt' else 'clear')
    total_trades, wins, win_rate, sharpe, avg_win, avg_loss, pf = account_obj.get_metrics()
    
    print(Fore.CYAN + "="*60)
    print(f"  AI TRADING BOT | {datetime.now().strftime('%H:%M:%S')} | {SYMBOL}")
    print("="*60 + Style.RESET_ALL)

    print(f"\n{Fore.YELLOW}MARKET STATUS:{Style.RESET_ALL}")
    print(f"Price: {Fore.WHITE}${price:,.2f}{Style.RESET_ALL} | RSI: {Fore.WHITE}{rsi:.1f}{Style.RESET_ALL}")
    print(f"ATR:   {Fore.WHITE}{atr:.2f}{Style.RESET_ALL}   | Leverage: {Fore.WHITE}{leverage}x{Style.RESET_ALL}")
    print(f"Signal: {Fore.MAGENTA}{action}{Style.RESET_ALL}")

    pos_color = Fore.GREEN if "LONG" in pos_state else (Fore.RED if "SHORT" in pos_state else Fore.WHITE)
    print(f"\n{Fore.YELLOW}ACTIVE POSITION (LOCAL TRACKING):{Style.RESET_ALL}")
    print(f"State:      {pos_color}{pos_state}{Style.RESET_ALL}")
    
    if "FLAT" not in pos_state:
        pnl_color = Fore.GREEN if pnl_live >= 0 else Fore.RED
        print(f"Open PnL:   {pnl_color}${pnl_live:+.2f}{Style.RESET_ALL}")
        print(f"Margin Used:{Fore.BLUE}${margin:.2f}{Style.RESET_ALL}")
        print(f"Duration:   {Fore.WHITE}{duration:.1f} mins{Style.RESET_ALL}")
    else:
        print(f"Open PnL:   -")
        print(f"Margin Used:-")
        print(f"Duration:   -")

    print(f"\n{Fore.YELLOW}PERFORMANCE (Simulated):{Style.RESET_ALL}")
    print(f"Balance:    {Fore.BLUE}${account_obj.balance:.2f}{Style.RESET_ALL}")
    print(f"Trades:     {total_trades} | Wins: {wins} | Win Rate: {win_rate:.1f}%")
    print(f"Sharpe:     {sharpe:.2f} | PF: {pf:.2f}")
    print(f"Avg Win:    ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f}")
    
    curr_dd = account_obj.drawdown_history[-1] * 100
    dd_col = Fore.RED if curr_dd > 10 else Fore.GREEN
    print(f"Drawdown:   {dd_col}{curr_dd:.2f}%{Style.RESET_ALL} (Limit: {MAX_DRAWDOWN_PCT*100}%)")
    print("\nPress Ctrl+C to stop.")

# --- 6. MAIN TRADER CLASS ---
class LiveTrader:
    def __init__(self):
        self.model = PPO.load(MODEL_PATH)
        self.plotter = LiveDashboard()
        
        # Local State
        self.pos = 0        
        self.entry_price = 0.0
        self.entry_time = 0
        self.sl_price = 0.0
        self.tp_price = 0.0
        self.leverage = 1
        self.zero_size_counter = 0

    def check_exit_conditions(self, current_price):
        if self.pos == 0: return False, 0

        elapsed = (time.time() - self.entry_time) / 60
        if elapsed >= MAX_HOLD_MINUTES: return True, "Time Limit"

        if self.pos == 1:
            if current_price <= self.sl_price: return True, "Stop Loss"
            if current_price >= self.tp_price: return True, "Take Profit"
        elif self.pos == -1:
            if current_price >= self.sl_price: return True, "Stop Loss"
            if current_price <= self.tp_price: return True, "Take Profit"

        return False, None

    def run(self):
        print("Initializing...")
        if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
        logging.basicConfig(filename=os.path.join(LOG_DIR, 'live_trading.log'), level=logging.INFO, format='%(asctime)s %(message)s')
        logging.info("Bot started.")
        time.sleep(2)

        while True:
            try:
                obs, price, atr, rsi, df = get_live_data()
                if obs is None: 
                    time.sleep(5); continue

                # 1. SYNC Check
                pos_details = get_position_details()
                if pos_details:
                    real_size = pos_details.get('size', 0)
                    if real_size == 0 and self.pos != 0:
                        self.zero_size_counter += 1
                        if self.zero_size_counter >= 5: # Grace Period
                            logging.info("Position closed on Exchange (SL/TP). Updating Local.")
                            
                            dist_sl = abs(price - self.sl_price)
                            dist_tp = abs(price - self.tp_price)
                            exit_p = self.sl_price if dist_sl < dist_tp else self.tp_price
                            pnl = (exit_p - self.entry_price) * TRADE_SIZE if self.pos == 1 else (self.entry_price - exit_p) * TRADE_SIZE
                            
                            log_trade_to_csv(datetime.fromtimestamp(self.entry_time).strftime('%Y-%m-%d %H:%M:%S'), 'LONG' if self.pos == 1 else 'SHORT', self.entry_price, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), exit_p, pnl, 'Exchange Trigger')
                            account.update_pnl(pnl)
                            
                            self.pos = 0
                            self.zero_size_counter = 0
                    elif real_size != 0:
                        self.zero_size_counter = 0

                # 2. LOCAL RISK CHECK
                if self.pos != 0:
                    should_exit, reason = self.check_exit_conditions(price)
                    if should_exit:
                        pnl = (price - self.entry_price) * TRADE_SIZE if self.pos == 1 else (self.entry_price - price) * TRADE_SIZE
                        
                        log_trade_to_csv(datetime.fromtimestamp(self.entry_time).strftime('%Y-%m-%d %H:%M:%S'), 'LONG' if self.pos == 1 else 'SHORT', self.entry_price, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), price, pnl, reason)
                        dd = account.update_pnl(pnl)
                        
                        place_order('sell' if self.pos==1 else 'buy', TRADE_SIZE, reduce_only=True)
                        logging.info(f"Local Exit ({reason}): PnL ${pnl:.2f}")
                        
                        if account.balance <= 0 or dd >= MAX_DRAWDOWN_PCT:
                            print(Fore.RED + "\n!!! MAX DRAWDOWN HIT !!!"); break
                        
                        self.pos = 0

                # 3. AI + HYBRID LOGIC
                action, _ = self.model.predict(obs, deterministic=True)
                act_str = ["HOLD", "BUY", "SELL"][action[0]]
                self.leverage = calculate_dynamic_leverage(atr, price)

                if ENABLE_HYBRID_OVERRIDES:
                    if rsi > RSI_OVERBOUGHT and self.pos == 0:
                        action = [2]; act_str = "FORCE SELL (RSI)"
                    elif rsi < RSI_OVERSOLD and self.pos == 0:
                        action = [1]; act_str = "FORCE BUY (RSI)"

                # 4. ENTRY
                if self.pos == 0:
                    sl_dist = price * INITIAL_STOP_LOSS_PCT
                    tp_dist = price * TAKE_PROFIT_PCT
                    
                    if action[0] == 1: 
                        set_leverage(self.leverage)
                        sl = int(price - sl_dist)
                        tp = int(price + tp_dist)
                        if place_order('buy', TRADE_SIZE, sl_price=sl, tp_price=tp):
                            self.pos = 1; self.entry_price = price; self.sl_price = sl; self.tp_price = tp; self.entry_time = time.time()
                            logging.info(f"Entered LONG at {price}")
                        elif place_order('buy', TRADE_SIZE, sl_price=sl, tp_price=tp) == "POSITION_EXISTS":
                             self.pos = 1; logging.info("Sync: Found existing LONG")

                    elif action[0] == 2: 
                        set_leverage(self.leverage)
                        sl = int(price + sl_dist)
                        tp = int(price - tp_dist)
                        if place_order('sell', TRADE_SIZE, sl_price=sl, tp_price=tp):
                            self.pos = -1; self.entry_price = price; self.sl_price = sl; self.tp_price = tp; self.entry_time = time.time()
                            logging.info(f"Entered SHORT at {price}")
                        elif place_order('sell', TRADE_SIZE, sl_price=sl, tp_price=tp) == "POSITION_EXISTS":
                             self.pos = -1; logging.info("Sync: Found existing SHORT")

                # 5. DASHBOARD
                float_pnl = 0.0
                margin_used = 0.0
                pos_duration = 0.0
                if self.pos != 0:
                    float_pnl = (price - self.entry_price) * TRADE_SIZE if self.pos == 1 else (self.entry_price - price) * TRADE_SIZE
                    margin_used = (price * TRADE_SIZE) / self.leverage
                    pos_duration = (time.time() - self.entry_time) / 60
                
                state_str = "FLAT"
                if self.pos == 1: state_str = f"LONG (SL: {self.sl_price:.0f})"
                if self.pos == -1: state_str = f"SHORT (SL: {self.sl_price:.0f})"
                
                # Log Perf CSV
                log_performance_to_csv(account.balance, account.drawdown_history[-1], account_obj=account, final=False)
                
                print_dashboard(price, rsi, act_str, state_str, float_pnl, margin_used, atr, self.leverage, account, pos_duration)
                self.plotter.update(df, account)
                
                time.sleep(60)

            except KeyboardInterrupt:
                log_performance_to_csv(account.balance, account.drawdown_history[-1], account_obj=account, final=True)
                break
            except Exception as e:
                logging.error(f"Loop Error: {e}")
                time.sleep(10)

if __name__ == "__main__":
    trader = LiveTrader()
    trader.run()