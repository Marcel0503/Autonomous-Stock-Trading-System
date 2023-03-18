# Configuration file, storing the essential API keys.
# To successfully run the script one needs to create an account at https://app.alpaca.markets/signup
# and enter the individual API key and secret API key.
alpaca_market_api_key = ""
alpaca_market_secret_api_key = ""

# Determines whether real money or paper money is used.
paper_trade = True

# URLs to the default alpaca.markets API and the paper alpaca.markets API
alpaca_api_url = "https://api.alpaca.markets"
alpaca_paper_api_url = "https://paper-api.alpaca.markets"

# Important file and directory paths
data_points_file_path = "../data_points/data_points_with_extracted_features_and_labels"
daily_stock_data_directory_path = "../daily_stock_data/"
models_file_path = "../models/knn.json"