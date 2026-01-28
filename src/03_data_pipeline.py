import pandas as pd
import pandas_ta as ta
import numpy as np
import os
import matplotlib.pyplot as plt
import logging

# --- Configuration ---
SYMBOL = "BTCUSD"
CUTOFF_DATE = "2024-02-01 12:00:00"  # Cutoff date for training data

# Paths
RAW_PRICE = os.path.join("data", "raw", f"delta_{SYMBOL}_1m.csv")
RAW_FUNDING = os.path.join("data", "raw", f"delta_{SYMBOL}_funding.csv")
PROCESSED_DIR = os.path.join("data", "processed")
LOG_FILE = os.path.join("data", "data_pipeline.log")

# Setup Logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w' # Overwrite log each run
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

def augment_data(df):
    """Generates 'Mirror World' data."""
    logging.info("   -> Augmenting data (Inverting Prices)...")
    df_aug = df.copy()
    
    mean_price = df['close'].mean()
    df_aug['close'] = (1 / df_aug['close']) 
    scale_factor = mean_price / df_aug['close'].mean()
    df_aug['close'] = df_aug['close'] * scale_factor

    # Flip Indicators
    if 'log_ret' in df_aug.columns: df_aug['log_ret'] *= -1
    if 'RSI_14' in df_aug.columns: df_aug['RSI_14'] = 100 - df_aug['RSI_14']
    if 'MACD_12_26_9' in df_aug.columns: df_aug['MACD_12_26_9'] *= -1
    if 'MACDh_12_26_9' in df_aug.columns: df_aug['MACDh_12_26_9'] *= -1
    if 'OBV' in df_aug.columns: df_aug['OBV'] *= -1
    
    # Flip Funding Rate
    if 'funding_rate' in df_aug.columns: df_aug['funding_rate'] *= -1

    return df_aug

def plot_validation(df_real, df_aug):
    """Generates a dissertation-quality validation plot."""
    logging.info("   -> Generating validation plots...")
    
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Price Paths
    plt.subplot(1, 2, 1)
    plt.plot(df_real['close'].values, label='Real (Bullish)', color='blue', alpha=0.7)
    plt.plot(df_aug['close'].values, label='Augmented (Bearish)', color='red', alpha=0.7)
    plt.title(f"{SYMBOL} Price Regimes")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Return Distribution
    plt.subplot(1, 2, 2)
    plt.hist(df_real['log_ret'], bins=50, alpha=0.5, label='Real Returns', color='blue', density=True)
    plt.hist(df_aug['log_ret'], bins=50, alpha=0.5, label='Augmented Returns', color='red', density=True)
    plt.title("Statistical Distribution of Returns")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(PROCESSED_DIR, "data_distribution.png")
    plt.savefig(output_path, dpi=300)
    logging.info(f"   -> Validation plot saved to {output_path}")

def run_pipeline():
    logging.info(f"--- Starting Full Data Pipeline for {SYMBOL} ---")
    
    # 1. Load Raw
    if not os.path.exists(RAW_PRICE):
        logging.error("Raw files missing.")
        return

    df_price = pd.read_csv(RAW_PRICE, parse_dates=['timestamp'], index_col='timestamp')
    df_fund = pd.read_csv(RAW_FUNDING, parse_dates=['timestamp'], index_col='timestamp')
    
    logging.info(f"Raw Price Rows: {len(df_price)}")
    
    # Clean & Cutoff
    df_price = df_price[df_price.index >= CUTOFF_DATE].sort_index()
    
    # Gap Fill
    full_idx = pd.date_range(start=df_price.index.min(), end=df_price.index.max(), freq='1min')
    df_price = df_price.reindex(full_idx).ffill()
    logging.info(f"Post-Gap Fill Rows: {len(df_price)}")
    
    # Merge Funding
    df_fund_aligned = df_fund.reindex(df_price.index, method='ffill')
    df_price['funding_rate'] = df_fund_aligned['funding_rate'].fillna(0)
    
    # 2. Features
    logging.info("Generating Indicators...")
    df = df_price.copy()
    df.ta.bbands(length=20, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.obv(append=True)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # Time Encodings
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    t = df['hour'] * 60 + df['minute']
    df['time_sin'] = np.sin(2 * np.pi * t / 1440)
    df['time_cos'] = np.cos(2 * np.pi * t / 1440)
    
    df.dropna(inplace=True)
    logging.info(f"Clean Features Rows: {len(df)}")

    # 3. Split
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    

    # 4. Augment
    train_df_aug = augment_data(train_df)
    train_final = pd.concat([train_df, train_df_aug])

    logging.info(f"Final Train Rows (with Augmentation): {len(train_final)}")

    
    # 5. Visual Validation
    plot_validation(train_df, train_df_aug)

    # 6. Save
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_final.to_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    val_df.to_csv(os.path.join(PROCESSED_DIR, "val.csv"))
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"))

    # Visualize Train/Val/Test split on the price series
    logging.info("   -> Generating train/val/test split visualization...")
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['close'], color='black', linewidth=0.8, label='Close Price')

    plt.axvspan(train_df.index.min(), train_df.index.max(), color='green', alpha=0.2, label='Train (70%)')
    plt.axvspan(val_df.index.min(), val_df.index.max(), color='orange', alpha=0.2, label='Validation (15%)')
    plt.axvspan(test_df.index.min(), test_df.index.max(), color='red', alpha=0.2, label='Test (15%)')

    plt.title(f"{SYMBOL} — Train / Validation / Test Split")
    plt.xlabel("Timestamp")
    plt.ylabel("Price")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    split_path = os.path.join(PROCESSED_DIR, "data_split.png")
    plt.savefig(split_path, dpi=300, bbox_inches='tight')
    logging.info(f"   -> Split plot saved to {split_path}")
    plt.close()
    
    logging.info("Pipeline Complete. Files saved.")
    print("Pipeline Finished. Check 'data/data_pipeline.log' for details.")

if __name__ == "__main__":
    run_pipeline()