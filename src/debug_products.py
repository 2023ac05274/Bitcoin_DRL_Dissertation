import requests

# Testnet URL
BASE_URL = "https://cdn-ind.testnet.deltaex.org"

def list_btc_products():
    print("Fetching Product List from Delta India Testnet...")
    try:
        response = requests.get(f"{BASE_URL}/v2/products")
        data = response.json()
        
        found = False
        print(f"{'ID':<10} | {'Symbol':<15} | {'Type':<20} | {'Description'}")
        print("-" * 80)
        
        for p in data['result']:
            if "BTC" in p['symbol'] and p['contract_type'] in ['perpetual_futures', 'futures']:
                print(f"{p['id']:<10} | {p['symbol']:<15} | {p['contract_type']:<20} | {p['description']}")
                found = True
                
        if not found:
            print("No BTC contracts found. Check API URL.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_btc_products()