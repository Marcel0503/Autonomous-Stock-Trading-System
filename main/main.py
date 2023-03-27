import logging, time, ml_prediction, trader, recent_market_data, config
from datetime import datetime, timedelta
from pytz import timezone
from alpaca_trade_api.rest import REST


# Calculates the time in seconds till three hours before the stock market opens again.
def calculate_seconds_to_sleep():
    clock = api.get_clock()
    until = clock.next_open
    until = datetime(until.year, until.month, until.day, until.hour, until.minute, until.second) - timedelta(hours=3)
    tz = timezone('US/Eastern')
    now = datetime.now(tz)
    now = datetime(now.year, now.month, now.day, now.hour, now.minute, now.second)
    seconds_to_sleep = (until - now).total_seconds()
    return seconds_to_sleep


# Makes the script sleep until three hours before the market opens again.
def sleep_till_three_hours_before_next_market_open():
    seconds_to_sleep = calculate_seconds_to_sleep()
    if seconds_to_sleep < 10.800:
        # Is executed in case it is less than three hours until the next market opening
        # The script will then sleep until the next day,
        # since it cannot guarantee that the machine learning predictions will be finished in time.
        time.sleep(10.900)
        seconds_to_sleep = calculate_seconds_to_sleep()
    try:
        logging.info("Sleeping for " + str(seconds_to_sleep) + " seconds")
        time.sleep(seconds_to_sleep)
    except ValueError as e:
        logging.error(e)


# Makes the script sleep until three minutes before the market closes.
def sleep_till_three_min_before_market_close():
    clock = api.get_clock()
    until = clock.next_close

    until = datetime(until.year, until.month, until.day, until.hour, until.minute, until.second) - timedelta(minutes=3)
    # Timezone of the API
    tz = timezone('US/Eastern')
    now = datetime.now(tz)
    # Creates an offset-naive datetime
    now = datetime(now.year, now.month, now.day, now.hour, now.minute, now.second)

    try:
        logging.info("Sleeping until " + str(until) + " US/Eastern time")
        time.sleep((until - now).total_seconds())
    except ValueError as e:
        logging.error(e)


def run():
    logging.info('Start ' + datetime.now().isoformat())

    # Sets the api reference
    recent_market_data.api = api
    trader.api = api

    while True:
        sleep_till_three_hours_before_next_market_open()
        # Dictionary containing the extracted features for every stock on the market
        recent_market_data_feature_values = recent_market_data.extract_stock_features()
        # Ranking based on the prediction of the machine learning component
        most_likely_ranking = ml_prediction.predict_most_likely_profitable(recent_market_data_feature_values)
        logging.info("most_likely_ranking: " + str(most_likely_ranking))
        syms = select_stocks_based_on_ranking(most_likely_ranking)
        logging.info("Selected stocks: " + str(most_likely_ranking))

        # Buys the selected stocks and sets automatic sell points (take profit, stop loss)
        trader.buy_stocks(syms)
        sleep_till_three_min_before_market_close()
        # Sells all stocks that the system still holds
        trader.sell_stocks()
        logging.info("End of day " + datetime.now().isoformat() + "\n")


# Determines in what stocks the system will invest based on the most likely ranking
def select_stocks_based_on_ranking(most_likely_ranking):
    syms = []
    for key, value in most_likely_ranking.items():
        if len(syms) >= 5:
            break
        if value >= 1.021:
            syms.append((key, value))
    return syms


if __name__ == "__main__":
    api_key = config.alpaca_market_api_key
    api_secret_key = config.alpaca_market_secret_api_key

    # Sets up the logging configuration
    now = datetime.now()
    logging.basicConfig(
        filename="../logs/" + str(now.day) + "-" + str(now.month) + "-" + str(
            now.year) + '.txt', filemode='a', format='%(levelname)s: %(message)s'
        , level=logging.INFO)

    # In case an api key is missing
    if len(api_key) == 0 or len(api_secret_key) == 0:
        logging.error("Missing API key!")
        exit()

    # Depending on whether real money or paper money is used for the trading a different API url is needed.
    if config.paper_trade:
        base_url = config.alpaca_paper_api_url
    else:
        base_url = config.alpaca_api_url
    # Authentication
    api = REST(key_id=api_key, secret_key=api_secret_key, base_url=base_url)

    # Runs the trading process
    run()