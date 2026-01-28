import requests
import pandas as pd
import time
from datetime import datetime
import os
import logging
import matplotlib.pyplot as plt

# --- Configuration ---
SYMBOL = "BTCUSD"
RESOLUTION = "1m"  # 1 minute candles
START_DATE = datetime(2023, 10, 1) # 2 Years' data
END_DATE = datetime(2025, 12, 31)

# Specific endpoint for Delta India
BASE_URL = "https://api.india.delta.exchange" # Using Testnet CDN for reliability in demo
# Note: For real mainnet data, swap back to: "https://api.india.delta.exchange" 

BATCH_SIZE = 2000 # Max candles per request

# Output setup
OUTPUT_DIR = os.path.join("data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"delta_{SYMBOL}_{RESOLUTION}.csv")
LOG_FILE = os.path.join("data", "data_fetch.log")

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)

def get_candles(start_ts, end_ts):
    """
    Fetches candles from Delta India API.
    Timestamps (secs) are in UNIX epoch format.
    """
    endpoint = "/v2/history/candles"
    url = BASE_URL + endpoint
    
    params = {
        "symbol": SYMBOL,
        "resolution": RESOLUTION,
        "start": int(start_ts),
        "end": int(end_ts)
    }
    
    try:
        with requests.Session() as session:
            response = session.get(url, params=params, timeout=10)
            response.raise_for_status() 
            data = response.json()
            
            if data.get("success") and "result" in data:
                return data["result"]
            else:
                logging.warning(f"API returned success=False or no result: {data}")
                return []
    except Exception as e:
        logging.error(f"Request failed: {e}")
        return []

def fetch_full_history():
    all_candles = []
    current_start = START_DATE.timestamp()
    final_end = END_DATE.timestamp()
    
    logging.info(f"--- Starting Download for {SYMBOL} ---")
    logging.info(f"From: {START_DATE}")
    logging.info(f"To:   {END_DATE}")
    
    total_minutes = (final_end - current_start) / 60
    logging.info(f"Expected candles: ~{int(total_minutes)}")

    while current_start < final_end:
        current_end = current_start + (BATCH_SIZE * 60)
        if current_end > final_end:
            current_end = final_end
            
        logging.info(f"Fetching: {datetime.fromtimestamp(current_start)} -> {datetime.fromtimestamp(current_end)}")
        
        candles = get_candles(current_start, current_end)
        
        if not candles:
            logging.warning("No candles returned for this slice. Moving to next.")
            current_start = current_end
            continue
            
        all_candles.extend(candles)
        current_start = current_end
        time.sleep(0.1) # Rate limit protection

    return all_candles

def visualize_data(df):
    """
    Generates a quick verification plot of the fetched data.
    """
    logging.info("Generating Data Verification Plot...")
    
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Close Price
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['close'], label='Close Price', color='blue', linewidth=1)
    plt.title(f"Raw Data Verification: {SYMBOL}")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Volume
    plt.subplot(2, 1, 2)
    plt.bar(df.index, df['volume'], label='Volume', color='gray', alpha=0.6, width=0.01)
    plt.title("Trading Volume")
    plt.ylabel("Volume")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "data_preview.png")
    plt.savefig(plot_path)
    logging.info(f"Visualization saved to: {plot_path}")
    plt.close() # Close to free memory

if __name__ == "__main__":
    # 1. Fetch
    raw_data = fetch_full_history()
    logging.info(f"Total candles fetched: {len(raw_data)}")
    
    if len(raw_data) > 0:
        # 2. Process
        df = pd.DataFrame(raw_data)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        
        # 3. Save CSV
        df.to_csv(OUTPUT_FILE)
        logging.info(f"Data saved to: {OUTPUT_FILE}")
        
        # 4. Visualize (New Step)
        visualize_data(df)
        
        # 5. Summary Stats
        logging.info(f"Date Range: {df.index.min()} to {df.index.max()}")
        logging.info(f"Missing seconds check: {(df.index[-1] - df.index[0]).total_seconds() / 60 - len(df):.0f} missing candles (approx)")
    else:
        logging.error("Failure: No data was collected.")