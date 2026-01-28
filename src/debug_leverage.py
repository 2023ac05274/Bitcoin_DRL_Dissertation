import time
import hmac
import hashlib
import requests
import json

# --- CONFIGURATION ---
BASE_URL = "https://cdn-ind.testnet.deltaex.org"
API_KEY = "6cWxkYDq6xydqkqsHP0LHnYOjeXNHL"       # <--- PASTE KEYS AGAIN
API_SECRET = "j23ZPsTrZJgMZmLuZyJGAONJsVN1swdZbNDFRhmXwEnIJK6gJgEioP3rPjsQ" # <--- PASTE KEYS AGAIN
SYMBOL = "BTCUSD"

def generate_signature(method, timestamp, path, query_string, payload):
    signature_data = method + timestamp + path + query_string + payload
    return hmac.new(API_SECRET.encode(), signature_data.encode(), hashlib.sha256).hexdigest()

def get_product_id():
    try:
        response = requests.get(f"{BASE_URL}/v2/products")
        for p in response.json()['result']:
            if p['symbol'] == SYMBOL: return p['id']
    except: return None

def try_leverage_endpoint(path_variant):
    pid = get_product_id()
    if not pid: 
        print("Product ID not found.")
        return

    print(f"\n--- Testing Endpoint: {path_variant} ---")
    
    timestamp = str(int(time.time()))
    payload = json.dumps({
        "product_id": int(pid),
        "leverage": "5" # Trying to set to 5x
    })
    
    signature = generate_signature("POST", timestamp, path_variant, "", payload)
    headers = {
        "api-key": API_KEY, "timestamp": timestamp, 
        "signature": signature, "Content-Type": "application/json"
    }
    
    try:
        # Full URL
        full_url = BASE_URL + path_variant
        print(f"POST {full_url}")
        
        resp = requests.post(full_url, headers=headers, data=payload, timeout=5)
        
        print(f"Status Code: {resp.status_code}")
        print(f"Response: {resp.text}")
        
        if resp.status_code == 200:
            print(">>> SUCCESS! This is the correct endpoint.")
            return True
        else:
            print(">>> Failed.")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("starting Leverage Endpoint Discovery...")
    
    # VARIANT 1: Standard Global Endpoint
    try_leverage_endpoint("/v2/orders/leverage")
    
    # VARIANT 2: Product Config Endpoint
    try_leverage_endpoint("/v2/products/leverage")
    
    # VARIANT 3: Legacy Endpoint
    try_leverage_endpoint("/orders/leverage")