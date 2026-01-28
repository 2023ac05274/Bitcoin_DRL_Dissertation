import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Configuration ---
SYMBOL = "BTCUSD"
RESOLUTION = "1m"
# Adjust the filename if yours is different
csv_filename = f"delta_{SYMBOL}_{RESOLUTION}.csv"
DATA_PATH = os.path.join("data", "raw", csv_filename)

def run_analysis():
    print(f"--- Starting EDA for {csv_filename} ---")
    
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: File not found at {DATA_PATH}")
        print("Please check if the file exists in 'data/raw/'")
        return

    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    print(f"Total Rows: {len(df)}")
    print(f"Date Range: {df.index.min()} to {df.index.max()}")
    
    # 2. Check for Duplicates
    duplicates = df.index.duplicated().sum()
    print(f"Duplicate Timestamps: {duplicates}")
    
    # 3. Check for Gaps
    # Calculate time difference
    time_diff = df.index.to_series().diff()
    # Find gaps larger than 1 minute
    gaps = time_diff[time_diff > pd.Timedelta(minutes=1, seconds=5)]
    print(f"Number of time gaps found: {len(gaps)}")
    
    if len(gaps) > 0:
        print("\nTop 5 Largest Gaps:")
        print(gaps.sort_values(ascending=False).head(5))
    
    # 4. Generate Plot
    print("\nGenerating price plot...")
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df['close'], label='Close Price', linewidth=0.5)
    plt.title(f'{SYMBOL} Price History (1m Candles)')
    plt.ylabel('Price (USD)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot to file instead of showing it
    output_img = "eda_price_plot.png"
    plt.savefig(output_img)
    print(f"Graph saved successfully as '{output_img}'")
    print("Check your project folder to see the image!")

if __name__ == "__main__":
    run_analysis()