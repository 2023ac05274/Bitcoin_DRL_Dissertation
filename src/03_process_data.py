import pandas as pd
import numpy as np
import os
# Not required 
# --- Configuration ---
SYMBOL = "BTCUSD"
CUTOFF_DATE = "2024-02-01 14:11:00" # Date to start "Clean" data from

# File Paths
RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")

PRICE_FILE = os.path.join(RAW_DIR, f"delta_{SYMBOL}_1m.csv")
FUNDING_FILE = os.path.join(RAW_DIR, f"delta_{SYMBOL}_funding.csv")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, f"MASTER_{SYMBOL}_2024_2025.csv")

def process_data():
    print(f"--- Starting Data Processing for {SYMBOL} ---")

    # 1. LOAD RAW DATA
    print("1. Loading Raw Files...")
    if not os.path.exists(PRICE_FILE):
        print(f"CRITICAL ERROR: Price file not found at {PRICE_FILE}")
        return
    if not os.path.exists(FUNDING_FILE):
        print(f"CRITICAL ERROR: Funding file not found at {FUNDING_FILE}")
        print("   -> Run 'src/02_fetch_funding.py' first.")
        return

    # Load Price
    df_price = pd.read_csv(PRICE_FILE)
    df_price['timestamp'] = pd.to_datetime(df_price['timestamp'])
    df_price.set_index('timestamp', inplace=True)
    df_price.sort_index(inplace=True)
    print(f"   - Raw Price Rows: {len(df_price)}")

    # Load Funding
    df_fund = pd.read_csv(FUNDING_FILE)
    df_fund['timestamp'] = pd.to_datetime(df_fund['timestamp'])
    df_fund.set_index('timestamp', inplace=True)
    df_fund.sort_index(inplace=True)
    print(f"   - Raw Funding Rows: {len(df_fund)}")

    # 2. CLEANING (AMPUTATION)
    print(f"\n2. Applying Cutoff Date ({CUTOFF_DATE})...")
    # Remove the "messy" data before Feb 2024
    df_clean = df_price[df_price.index >= CUTOFF_DATE].copy()
    rows_removed = len(df_price) - len(df_clean)
    print(f"   - Removed {rows_removed} rows (messy data before cutoff).")

    # 3. GAP DETECTION & FILLING
    print("\n3. Checking for Time Gaps...")
    # Calculate time difference between rows
    time_diff = df_clean.index.to_series().diff()
    # Find gaps > 1 minute (allowing 5s buffer)
    gaps = time_diff[time_diff > pd.Timedelta(minutes=1, seconds=5)]

    if len(gaps) > 0:
        print(f"   - WARNING: Found {len(gaps)} gaps.")
        print("   - Filling small gaps with Forward Fill (ffill)...")
        # Create a perfect 1-minute index
        full_idx = pd.date_range(start=df_clean.index.min(), end=df_clean.index.max(), freq='1min')
        # Reindex to this perfect grid
        df_clean = df_clean.reindex(full_idx)
        # Fill missing prices with the previous minute's price
        df_clean.ffill(inplace=True)
        print("   - Gaps filled.")
    else:
        print("   - Data is perfectly continuous. No gaps found.")

    # 4. MERGING FUNDING RATES
    print("\n4. Merging Funding Rates...")
    # Funding happens every 8 hours. We need to broadcast this to every minute.
    # We reindex funding to match the CLEAN price index and forward fill.
    df_fund_aligned = df_fund.reindex(df_clean.index, method='ffill')
    
    # Add funding column to price data
    df_clean['funding_rate'] = df_fund_aligned['funding_rate']
    
    # Fill any NaNs at the start (if price starts before first funding record)
    df_clean['funding_rate'] = df_clean['funding_rate'].fillna(0)

    # 5. FINAL CHECKS & SAVE
    print("\n5. Saving Master Dataset...")
    # Ensure processed directory exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Final NaN check
    if df_clean.isnull().values.any():
        print("   WARNING: Final dataset contains NaNs! Checking...")
        print(df_clean.isnull().sum())
        # Drop leftover NaNs just in case
        df_clean.dropna(inplace=True) 
    
    df_clean.to_csv(OUTPUT_FILE)
    print(f"   - SUCCESS: Saved to {OUTPUT_FILE}")
    print(f"   - Final Shape: {df_clean.shape}")
    print("\nSample Data:")
    print(df_clean[['close', 'funding_rate']].head())

if __name__ == "__main__":
    process_data()