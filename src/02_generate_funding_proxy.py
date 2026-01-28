import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt

# --- Configuration ---
SYMBOL = "BTCUSD"
RAW_PRICE_FILE = os.path.join("data", "raw", f"delta_{SYMBOL}_1m.csv")
OUTPUT_FILE = os.path.join("data", "raw", f"delta_{SYMBOL}_funding.csv")
LOG_FILE = os.path.join("data", "funding_generation.log")

# --- Setup Logging (Append Mode) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'), # 'a' = Append (Don't overwrite)
        logging.StreamHandler()
    ]
)

def visualize_funding(df):
    """
    Generates a verification plot for the synthetic funding rates.
    """
    logging.info("Generating Funding Rate Verification Plot...")
    
    plt.figure(figsize=(12, 6))
    
    # Plot Funding Rate
    plt.plot(df.index, df['funding_rate'], label='Synthetic Funding Rate', color='purple', linewidth=0.5)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    
    # Add title and labels
    plt.title(f"Synthetic Funding Rate Profile: {SYMBOL}")
    plt.ylabel("Funding Rate (8h)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save
    plot_path = os.path.join("data", "raw", "funding_check.png")
    plt.savefig(plot_path)
    logging.info(f"Visualization saved to: {plot_path}")
    plt.close()

def generate_proxy_funding():
    logging.info(f"--- Starting Synthetic Funding Generation for {SYMBOL} ---")
    
    # 1. Load Price Data
    if not os.path.exists(RAW_PRICE_FILE):
        logging.error(f"Price file not found at {RAW_PRICE_FILE}")
        return
    
    df = pd.read_csv(RAW_PRICE_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    logging.info(f"Loaded {len(df)} price rows for processing.")
    
    # 2. Calculate Proxy Funding Rate
    # Logic: Funding is proportional to price deviation from a moving average.
    
    # Calculate 8-hour Moving Average (8 * 60 = 480 minutes)
    window = 480
    df['ma_8h'] = df['close'].rolling(window=window).mean()
    
    # Calculate Deviation (Premium)
    df['premium_proxy'] = (df['close'] - df['ma_8h']) / df['ma_8h']
    
    # Scale to realistic funding levels (e.g., 0.01%)
    SCALING_FACTOR = 0.01 
    df['funding_rate'] = df['premium_proxy'] * SCALING_FACTOR
    
    # 3. Apply Hard Limits (Capping)
    # Standard exchanges cap at +/- 0.375% per 8h
    MAX_FUNDING = 0.00375
    df['funding_rate'] = df['funding_rate'].clip(lower=-MAX_FUNDING, upper=MAX_FUNDING)
    
    # Handle NaNs (caused by the rolling window start)
    df['funding_rate'] = df['funding_rate'].fillna(0)
    
    # 4. Save
    output_df = df[['funding_rate']].copy()
    output_df.to_csv(OUTPUT_FILE)
    
    logging.info(f"Successfully generated {len(output_df)} funding records.")
    logging.info(f"Saved to: {OUTPUT_FILE}")
    
    # Log Statistics for audit
    stats = output_df['funding_rate'].describe()
    logging.info(f"Funding Stats:\n{stats}")
    
    # 5. Visualize
    visualize_funding(output_df)

if __name__ == "__main__":
    generate_proxy_funding()