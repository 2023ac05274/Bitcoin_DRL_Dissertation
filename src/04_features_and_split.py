import pandas as pd
import pandas_ta as ta
import numpy as np
import os
import matplotlib.pyplot as plt
# Not required
# --- CONFIGURATION ---
SYMBOL = "BTCUSD"

# Input File (From Step 3)
INPUT_FILE = f"MASTER_{SYMBOL}_2024_2025.csv"
INPUT_PATH = os.path.join("data", "processed", INPUT_FILE)

# Split Ratios (Scientific Standard)
TRAIN_RATIO = 0.70  # 70% for learning
VAL_RATIO = 0.15    # 15% for checking progress (Mock Exam)
TEST_RATIO = 0.15   # 15% for final proof (Final Exam)

def process_features_and_split():
    print(f"--- Starting Feature Engineering & Splitting for {SYMBOL} ---")
    
    # 1. LOAD CLEANED DATA
    if not os.path.exists(INPUT_PATH):
        print(f"CRITICAL ERROR: Master file not found at {INPUT_PATH}")
        print("Run 'src/03_process_data.py' first.")
        return
    
    print("1. Loading Master Dataset...")
    # robustness: ensure index is datetime
    df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    print(f"   - Loaded {len(df)} rows.")

    # 2. FEATURE ENGINEERING
    print("\n2. Generating Technical Indicators...")
    
    # A. Volatility: Bollinger Bands (Length 20)
    df.ta.bbands(length=20, append=True)
    
    # B. Momentum: RSI (Relative Strength Index)
    df.ta.rsi(length=14, append=True)
    
    # C. Trend: MACD
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    
    # D. Volatility: ATR (Crucial for Risk Mgmt)
    df.ta.atr(length=14, append=True)
    
    # E. Volume: OBV
    df.ta.obv(append=True)

    # F. Log Returns (Normalization for AI)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # G. Time Encoding (Cyclical Time)
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    minutes_in_day = 24 * 60
    current_minute = df['hour'] * 60 + df['minute']
    df['time_sin'] = np.sin(2 * np.pi * current_minute / minutes_in_day)
    df['time_cos'] = np.cos(2 * np.pi * current_minute / minutes_in_day)
    
    # Drop rows with NaN (Resulting from indicator lookback periods)
    initial_len = len(df)
    df.dropna(inplace=True)
    print(f"   - Features generated. Dropped {initial_len - len(df)} rows due to lookback.")
    
    # Save the full feature set (Optional, but good for reference)
    full_feature_path = os.path.join("data", "processed", f"FEATURES_{SYMBOL}_FULL.csv")
    df.to_csv(full_feature_path)
    print(f"   - Full feature set saved to {full_feature_path}")

    # 3. SPLITTING DATA
    print(f"\n3. Splitting Data (Train={TRAIN_RATIO}, Val={VAL_RATIO}, Test={TEST_RATIO})...")
    
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    print(f"   - Training Set:   {len(train_df)} rows")
    print(f"   - Validation Set: {len(val_df)} rows")
    print(f"   - Test Set:       {len(test_df)} rows")
    
    # 4. SAVING SPLITS
    print("\n4. Saving Split Files...")
    train_path = os.path.join("data", "processed", "train.csv")
    val_path = os.path.join("data", "processed", "val.csv")
    test_path = os.path.join("data", "processed", "test.csv")
    
    train_df.to_csv(train_path)
    val_df.to_csv(val_path)
    test_df.to_csv(test_path)
    
    print(f"   - Saved: {train_path}")
    print(f"   - Saved: {val_path}")
    print(f"   - Saved: {test_path}")

    # 5. VISUALIZATION
    print("\n5. Generating Visual Report...")
    plt.figure(figsize=(15, 6))
    plt.plot(train_df.index, train_df['close'], label='Training (Learn)', color='blue', linewidth=1)
    plt.plot(val_df.index, val_df['close'], label='Validation (Tune)', color='orange', linewidth=1)
    plt.plot(test_df.index, test_df['close'], label='Testing (Prove)', color='green', linewidth=1)
    
    plt.title(f"Dissertation Data Split: {SYMBOL}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = "data_split_visual.png"
    plt.savefig(plot_path)
    print(f"   - Visualization saved to {plot_path}")
    print("\nSUCCESS: Data Pipeline Complete. Ready for Training.")

if __name__ == "__main__":
    process_features_and_split()