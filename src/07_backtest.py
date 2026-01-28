import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas_ta as ta
from stable_baselines3 import PPO
from trading_env import BitcoinTradingEnv
from datetime import datetime

# --- Configuration ---
SYMBOL = "BTCUSD"
INPUT_FILE = "test.csv"
INPUT_PATH = os.path.join("data", "processed", INPUT_FILE)
MODELS_DIR = "models"
OUTPUT_DIR = "results"

# Load the Champion Model
MODEL_PATH = os.path.join(MODELS_DIR, "PPO_BTCUSD_FINAL_1768304053.zip") 

# Ensure results folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_advanced_metrics(df_trades, initial_balance, final_balance, equity_curve):
    """Calculates professional financial metrics."""
    
    # 1. Basic Stats
    total_return = (final_balance - initial_balance) / initial_balance * 100
    n_trades = len(df_trades)
    
    if n_trades > 0:
        wins = df_trades[df_trades['PnL'] > 0]
        losses = df_trades[df_trades['PnL'] < 0]
        win_rate = len(wins) / n_trades * 100
        
        gross_profit = wins['PnL'].sum()
        gross_loss = abs(losses['PnL'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf
        
        # Directional Analysis
        longs = df_trades[df_trades['Direction'] == 'Long']
        shorts = df_trades[df_trades['Direction'] == 'Short']
        
        long_win_rate = len(longs[longs['PnL'] > 0]) / len(longs) * 100 if len(longs) > 0 else 0
        short_win_rate = len(shorts[shorts['PnL'] > 0]) / len(shorts) * 100 if len(shorts) > 0 else 0
    else:
        win_rate = 0; profit_factor = 0
        long_win_rate = 0; short_win_rate = 0
        longs = []; shorts = []

    # 2. Drawdown
    equity_curve = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    # 3. Sharpe Ratio (Annualized)
    returns = pd.Series(equity_curve).pct_change().dropna()
    # Assuming 1m candles -> 525,600 mins/year
    sharpe = (returns.mean() / returns.std()) * np.sqrt(525600) if returns.std() != 0 else 0

    return {
        "Total Return": f"{total_return:.2f}%",
        "Total Trades": n_trades,
        "Win Rate": f"{win_rate:.2f}%",
        "Profit Factor": f"{profit_factor:.2f}",
        "Max Drawdown": f"{max_drawdown:.2f}%",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Long Win Rate": f"{long_win_rate:.2f}% ({len(longs)} trades)",
        "Short Win Rate": f"{short_win_rate:.2f}% ({len(shorts)} trades)"
    }, drawdown

def run_backtest():
    print(f"--- Starting Full Backtest for {SYMBOL} ---")
    
    # 1. Load Data
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found.")
        return
    df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
    
    if 'ATRr_14' not in df.columns:
        print("   - Calculating ATR...")
        df.ta.atr(length=14, append=True)
        df['ATRr_14'] = df['ATRr_14'].fillna(0)

    print(f"   - Loaded {len(df)} rows.")
    
    # 2. Load Agent
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return
    print(f"   - Loading Model: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)
    
    # 3. Initialize Environment (FIX: Unlimited Drawdown)
    env = BitcoinTradingEnv(
        df, 
        execution_mode='taker', 
        leverage=2,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        max_drawdown_pct=2.0  # Set > 1.0 to disable early stopping
    )
    
    # 4. Run Simulation
    obs, info = env.reset()
    env.current_step = 0 
    
    balances, positions, prices = [], [], []
    trade_log = []
    current_trade = {}
    
    done = False
    print("   - Running simulation (this may take a moment)...")
    
    while not done:
        action, _ = model.predict(obs, deterministic=False)
        
        prev_balance = env.balance
        prev_pos = env.position
        
        obs, reward, done, truncated, info = env.step(action)
        
        current_price = df.iloc[env.current_step]['close']
        timestamp = df.index[env.current_step]
        
        balances.append(env.balance)
        positions.append(env.position)
        prices.append(current_price)
        
        # --- Trade Logging ---
        # Entry
        if prev_pos == 0 and env.position != 0:
            current_trade = {
                'Entry_Time': timestamp,
                'Entry_Price': current_price,
                'Direction': 'Long' if env.position == 1 else 'Short',
                'Size': env.balance * env.leverage
            }
            
        # Exit
        elif prev_pos != 0 and env.position != prev_pos:
            current_trade['Exit_Time'] = timestamp
            current_trade['Exit_Price'] = current_price
            
            if current_trade['Direction'] == 'Long':
                pnl_pct = (current_price - current_trade['Entry_Price']) / current_trade['Entry_Price']
            else:
                pnl_pct = (current_trade['Entry_Price'] - current_price) / current_trade['Entry_Price']
            
            current_trade['PnL'] = (env.balance - prev_balance) 
            current_trade['Return_Pct'] = pnl_pct * 100
            current_trade['Duration_Mins'] = (timestamp - current_trade['Entry_Time']).seconds / 60
            
            trade_log.append(current_trade)
            
            # If Flip (Exit + New Entry)
            if env.position != 0:
                current_trade = {
                    'Entry_Time': timestamp,
                    'Entry_Price': current_price,
                    'Direction': 'Long' if env.position == 1 else 'Short',
                    'Size': env.balance * env.leverage
                }
            else:
                current_trade = {}

    # 5. Analysis
    print("\n--- RESULTS SUMMARY ---")
    
    test_index = df.index[:len(balances)]
    portfolio_series = pd.Series(balances, index=test_index)
    price_series = pd.Series(prices, index=test_index)
    
    # Benchmark
    initial_price = price_series.iloc[0]
    buy_hold_series = (price_series / initial_price) * env.initial_balance
    
    df_trades = pd.DataFrame(trade_log)
    metrics, drawdown_curve = calculate_advanced_metrics(df_trades, env.initial_balance, env.balance, balances)
    
    # Print Metrics
    print(f"{'Metric':<25} | {'Value'}")
    print("-" * 40)
    for k, v in metrics.items():
        print(f"{k:<25} | {v}")
    print("-" * 40)
    print(f"Final Balance:   ${env.balance:.2f}")
    print(f"Buy & Hold:      ${buy_hold_series.iloc[-1]:.2f}")

    # 6. Save Results
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not df_trades.empty:
        trades_file = os.path.join(OUTPUT_DIR, f"trades_{timestamp_str}.csv")
        df_trades.to_csv(trades_file)
        print(f"   - Trade Log: {trades_file}")
    
    # 7. Visualization
    plt.figure(figsize=(12, 12))
    
    # Equity
    plt.subplot(3, 1, 1)
    plt.plot(portfolio_series, label='DRL Agent', color='blue')
    plt.plot(buy_hold_series, label='Buy & Hold', color='gray', linestyle='--')
    plt.title(f"Equity Curve: {SYMBOL} (Test Set)")
    plt.ylabel("Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Trades
    plt.subplot(3, 1, 2)
    plt.plot(price_series, label='Price', color='black', alpha=0.5)
    if not df_trades.empty:
        longs = df_trades[df_trades['Direction'] == 'Long']
        shorts = df_trades[df_trades['Direction'] == 'Short']
        plt.scatter(longs['Entry_Time'], longs['Entry_Price'], marker='^', color='green', s=80, label='Long', zorder=5)
        plt.scatter(shorts['Entry_Time'], shorts['Entry_Price'], marker='v', color='red', s=80, label='Short', zorder=5)
        plt.scatter(df_trades['Exit_Time'], df_trades['Exit_Price'], marker='x', color='black', s=30, label='Exit', zorder=5)
    plt.title("Trade Entries & Exits")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Drawdown
    plt.subplot(3, 1, 3)
    plt.fill_between(test_index, drawdown_curve * 100, 0, color='red', alpha=0.3)
    plt.plot(test_index, drawdown_curve * 100, color='red', linewidth=1)
    plt.title("Drawdown %")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"backtest_{timestamp_str}.png")
    plt.savefig(plot_path, dpi=300)
    print(f"   - Plot: {plot_path}")
    plt.show()

if __name__ == "__main__":
    run_backtest()