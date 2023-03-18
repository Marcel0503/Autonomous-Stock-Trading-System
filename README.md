# Autonomous-Stock-Trading-System
## Setup
To successfully run the script an alpaca.markets account is needed (https://app.alpaca.markets/signup).
In the main/config.py file, one has to enter the API key and the secret API key from one's individual alpaca.markets account.
In this configuration file, one also has the option to choose whether the API
should use real money for the investments or simulate the investments with paper money.
This is determined by the paper_trade variable. By default, the system will use paper money.

## Run
To run the Python script, one has to execute the main/main.py file.

# Structure
The main folder contains the Python modules.<br>
The system logs relevant information via text files in the logs directory.<br>
The daily_stock_data folder contains the historic market data of 6,457 stocks in daily intervals.<br>
The data_points folder contains a pandas dataframe consisting of the data points with labels extracted from the historic stock data.<br>
The models folder stores the trained machine-learning models, which are then used to make predictions.
