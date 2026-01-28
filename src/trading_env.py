import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class BitcoinTradingEnv(gym.Env):
    """
    UPGRADED DRL Environment
    Improvements:
    - Continuous Reward (Net Worth Change) -> Fixes "Sparse Reward" problem
    - Reward Scaling -> Helps Neural Network converge
    - Step Penalty -> Discourages inaction
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df, initial_balance=1000, execution_mode='taker', 
                 leverage=5, 
                 # --- TIGHTER SCALPING TARGETS ---
                 stop_loss_pct=0.0025,    # 0.25% (Tight Stop)
                 take_profit_pct=0.005,   # 0.5% (Realistic 30m Target)
                 trailing_stop_pct=0.002, # 0.2% Trail
                 max_drawdown_pct=0.10,
                 reward_scaling=1e-4):      # <--- NEW: Scale rewards to be ~1.0
        
        super(BitcoinTradingEnv, self).__init__()

        self.df = df
        self.initial_balance = initial_balance
        self.execution_mode = execution_mode
        self.leverage = leverage
        self.reward_scaling = reward_scaling 
        
        # Risk Config
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_drawdown_pct = max_drawdown_pct
        
        # Fees (Delta India)
        self.taker_fee_pct = 0.0005 
        self.maker_fee_pct = 0.0002
        self.gst_rate = 0.18  

        # Action Space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

        # Observation Space
        self.feature_cols = [
            'log_ret', 'RSI_14', 'ATRr_14', 'OBV', 
            'MACD_12_26_9', 'MACDh_12_26_9', 
            'BBU_20_2.0_2.0', 'BBL_20_2.0_2.0', 
            'time_sin', 'time_cos', 'funding_rate'
        ]
        
        # Safety check
        if 'funding_rate' not in self.df.columns:
            self.df['funding_rate'] = 0.0

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.feature_cols),), dtype=np.float32
        )

    def _calculate_fee(self, trade_value):
        notional_value = trade_value * self.leverage
        if self.execution_mode == 'maker':
            raw_fee = notional_value * self.maker_fee_pct
        else:
            raw_fee = notional_value * self.taker_fee_pct
        return raw_fee * (1 + self.gst_rate)

    def _get_net_worth(self, current_price):
        """Calculates Balance + Unrealized PnL"""
        net_worth = self.balance
        if self.position != 0:
            # Unrealized PnL
            if self.position == 1:
                pnl = (current_price - self.entry_price) / self.entry_price
            else:
                pnl = (self.entry_price - current_price) / self.entry_price
            
            # Value = Margin Balance + (Notional * PnL)
            # Notional = Balance * Leverage
            unrealized_profit_usd = (self.balance * self.leverage) * pnl
            net_worth += unrealized_profit_usd
        
        return net_worth

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.highest_balance = self.initial_balance 
        self.position = 0 
        self.entry_price = 0
        self.peak_price_since_entry = 0 
        
        # Start random
        self.current_step = np.random.randint(100, len(self.df) - 5000)
        
        # Track previous Net Worth for Continuous Reward
        self.prev_net_worth = self.initial_balance
        
        return self._next_observation(), {}

    def _next_observation(self):
        obs = self.df.iloc[self.current_step][self.feature_cols].values
        return obs.astype(np.float32)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        current_time = self.df.index[self.current_step]
        
        terminated = False
        step_reward = 0
        
        # --- 1. Funding Rate Logic ---
        if current_time.minute == 0 and current_time.hour % 8 == 0:
            f_rate = self.df.iloc[self.current_step]['funding_rate']
            funding_payment = (self.balance * self.leverage) * f_rate
            if self.position == 1: self.balance -= funding_payment
            elif self.position == -1: self.balance += funding_payment

        # --- 2. Execution Logic ---
        # NOTE: Fees reduce self.balance, which immediately drops Net Worth
        if action == 1: # BUY
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
                self.peak_price_since_entry = current_price 
                self.balance -= self._calculate_fee(self.balance)
            elif self.position == -1: # Flip Short -> Long
                # Close Short
                pnl = ((self.entry_price - current_price) / self.entry_price) * self.leverage
                self.balance += (self.balance * pnl)
                self.balance -= self._calculate_fee(self.balance) # Exit fee
                # Open Long
                self.balance -= self._calculate_fee(self.balance) # Entry fee
                self.position = 1
                self.entry_price = current_price
                self.peak_price_since_entry = current_price

        elif action == 2: # SELL
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price
                self.peak_price_since_entry = current_price
                self.balance -= self._calculate_fee(self.balance)
            elif self.position == 1: # Flip Long -> Short
                # Close Long
                pnl = ((current_price - self.entry_price) / self.entry_price) * self.leverage
                self.balance += (self.balance * pnl)
                self.balance -= self._calculate_fee(self.balance)
                # Open Short
                self.balance -= self._calculate_fee(self.balance)
                self.position = -1
                self.entry_price = current_price
                self.peak_price_since_entry = current_price

        # --- 3. Check Risk Management (Hard Stops) ---
        if self.position != 0:
            if self.position == 1:
                price_move_pct = (current_price - self.entry_price) / self.entry_price
                if current_price > self.peak_price_since_entry: self.peak_price_since_entry = current_price
                dd = (self.peak_price_since_entry - current_price) / self.peak_price_since_entry
            else:
                price_move_pct = (self.entry_price - current_price) / self.entry_price
                if current_price < self.peak_price_since_entry: self.peak_price_since_entry = current_price
                dd = (current_price - self.peak_price_since_entry) / self.peak_price_since_entry
            
            leveraged_pnl_pct = price_move_pct * self.leverage

            # A. STOP LOSS
            if leveraged_pnl_pct <= -self.stop_loss_pct:
                self.balance += (self.balance * leveraged_pnl_pct) # Realize loss
                self.balance -= self._calculate_fee(self.balance)  # Fee
                self.position = 0
                step_reward -= 0.5 # Extra Penalty

            # B. TAKE PROFIT
            elif leveraged_pnl_pct >= self.take_profit_pct:
                self.balance += (self.balance * leveraged_pnl_pct) # Realize profit
                self.balance -= self._calculate_fee(self.balance)
                self.position = 0
                step_reward += 1.0 # Extra Bonus

            # C. TRAILING STOP
            elif dd >= self.trailing_stop_pct:
                if self.position == 1:
                    pnl_realized = ((current_price - self.entry_price)/self.entry_price) * self.leverage
                else:
                    pnl_realized = ((self.entry_price - current_price)/self.entry_price) * self.leverage
                
                self.balance += (self.balance * pnl_realized)
                self.balance -= self._calculate_fee(self.balance)
                self.position = 0
                # Small bonus for trailing stop hit (profitable)
                if pnl_realized > 0: step_reward += 0.5

        # --- 4. Continuous Reward Calculation ---
        current_net_worth = self._get_net_worth(current_price)
        
        # Reward = (New Net Worth - Old Net Worth)
        # This captures price moves, fees paid, and funding paid immediately.
        net_worth_delta = current_net_worth - self.prev_net_worth
        
        # Scale reward (e.g., $1 gain = 0.0001 reward, or similar scaling to fit Neural Net)
        # We can use percentage change to make it balance-independent
        # reward = (Delta / Initial) * 100
        step_reward += (net_worth_delta / self.initial_balance) * 100 

        # Update tracking
        self.prev_net_worth = current_net_worth

        # --- 5. Step Penalty (Fight Inaction) ---
        step_reward -= 0.0001

        # --- 6. Circuit Breaker ---
        if self.balance > self.highest_balance: self.highest_balance = self.balance
        drawdown = (self.highest_balance - self.balance) / self.highest_balance
        if drawdown >= self.max_drawdown_pct:
            terminated = True
            step_reward -= 10 # Huge Penalty

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            terminated = True
            
        return self._next_observation(), step_reward, terminated, False, {}