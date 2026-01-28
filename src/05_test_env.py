import pandas as pd
from trading_env import BitcoinTradingEnv
import os
#Not required
# Load Data
SYMBOL = "BTCUSD"
INPUT_FILE = f"FEATURES_{SYMBOL}_2024_2025.csv"
INPUT_PATH = os.path.join("data", "processed", INPUT_FILE)

if os.path.exists(INPUT_PATH):
    df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
    
    # Initialize Environment
    env = BitcoinTradingEnv(df)
    
    # Reset Environment
    obs, info = env.reset()
    print("Environment Reset Successful!")
    print(f"Observation Shape: {obs.shape}")
    print(f"Initial Observation (First 5 features): {obs[:5]}")
    
    # Take 10 Random Actions
    print("\nTaking 10 Random Steps...")
    for i in range(10):
        # Sample random action (0, 1, or 2)
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.4f}")
        
        if term or trunc:
            print("Episode Ended!")
            break
            
    print("\nTest Complete: Environment is ready for AI Training.")
else:
    print("Error: Feature file not found.")