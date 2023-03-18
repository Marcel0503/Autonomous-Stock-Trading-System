import threading, feature_extraction
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta

api = None


# Retrieves the data of all currently active and tradable stocks at the nasdaq stock market
# and extracts features from this data according to the schema used for the training data.
def extract_stock_features():
    recent_bars = {}
    start = get_time(50)
    end = get_time(1)
    threads = set()
    active_assets = api.list_assets(status='active', asset_class='us_equity')
    tradable_nasdaq_assets = [a.symbol for a in active_assets if a.tradable and a.exchange == 'NASDAQ']
    # Multi-threading to increase performance
    syms_per_request = 1001
    for i in range((len(tradable_nasdaq_assets) - 1) // syms_per_request + 1):
        thread = threading.Thread(target=add_set_of_syms_bars,
                                  args=(
                                      recent_bars,
                                      tradable_nasdaq_assets[i * syms_per_request:(i + 1) * syms_per_request], start,
                                      end,))
        thread.start()
        threads.add(thread)
    for thread in threads:
        thread.join()
    features_values = {}
    for key in recent_bars:
        # OHLC values for the last day the stock market was open
        most_recent_bar = recent_bars[key][len(recent_bars[key]) - 1]
        # OHLC values for the second last day the stock market was open
        second_most_recent_bar = recent_bars[key][len(recent_bars[key]) - 2]
        # Prefiltering to only allow stock with previous days volume over 5000
        # and price increase yesterday of more than 20%
        if most_recent_bar.v <= 5000 or second_most_recent_bar.v <= 5000 or most_recent_bar.c <= 1.2 * most_recent_bar.o:
            continue
        # Extracts the features from the recent data
        features = make_features(recent_bars[key])
        if features:
            # Stores those features in a dictionary where key is the stock's symbol
            features_values[key] = features
    return features_values


# Makes features for a certain stock course according to the schema used for the training data
# based on the 21 last days the stock market was open.
def make_features(bars):
    length = len(bars)
    if length < 21:
        return None
    recent_values = []
    for bar in bars[length - 21:length]:
        recent_values.insert(0, (bar.o, bar.h, bar.l, bar.c, bar.v))
    return feature_extraction.make_features(recent_values)


# Returns the datetime of the day that was timedelta_days ago.
# For timedelta_days = 1, yesterday is returned in a specific datetime-format.
def get_time(timedelta_days):
    time = datetime.now() - timedelta(days=timedelta_days)
    year = str(time.year)
    month = str(time.month)
    day = str(time.day)
    # In case month is a single digit, a 0 is added in front.
    if len(month) == 1:
        month = '0' + month
    # In case day is a single digit, a 0 is added in front.
    if len(day) == 1:
        day = '0' + day
    time = year + '-' + month + '-' + day + 'T23:59:00+00:00'
    return time


# Adds the recent stock data for the given time period for the given stock symbols to the dictionary recent_bars.
def add_set_of_syms_bars(recent_bars, syms, start, end):
    bars = api.get_bars(symbol=syms, timeframe=TimeFrame(1, TimeFrameUnit.Day),
                        start=start, end=end)
    # Adds all bars from one stock into a list. All lists are stored in the recent_bars dictionary.
    for bar in bars:
        if recent_bars.get(bar.S):
            # List already exists, add bar to the list
            recent_bars[bar.S].append(bar)
        else:
            # Initialize list
            recent_bars[bar.S] = [bar]