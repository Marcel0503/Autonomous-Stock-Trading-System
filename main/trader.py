import logging, math

api = None


# Returns the balance of the connected account.
def get_cash():
    account = api.get_account()
    return float(account.cash)


# Buys shares of the given stocks.
# For each stock, 19% of the current account's balance is used to buy corresponding shares.
def buy_stocks(syms):
    if len(syms) == 0:
        return
    cash = get_cash()
    logging.info("Cash: " + str(cash))
    cash_for_one_trade = math.floor(cash * 0.19)
    for sym, exp_profit in syms:
        bar = api.get_latest_bar(sym)

        price = bar.c
        quantity = int(cash_for_one_trade // price)
        # Stop Loss price at which the API will automatically sell all shares if the stock price falls below this mark
        stop_loss = {'stop_price': price * 0.98}
        # Take Profit price at which the API will automatically sell all shares if the price
        take_profit = {'limit_price': price * exp_profit}
        logging.info(
            "Buying " + sym + ": total: " + str(quantity * price) + "; quantity: " + str(quantity) + "; price: " + str(
                price) + "; stop_loss: " + str(
                stop_loss['stop_price']) + "; take_profit: " + str(take_profit['limit_price']))
        api.submit_order(symbol=sym, qty=quantity, side="buy", take_profit=take_profit, stop_loss=stop_loss)


# Cancels any still remaining buy orders in case some of them were not executed as expected
# and sells all shares the system still holds.
def sell_stocks():
    logging.info("Closing all positions")
    api.cancel_all_orders()
    api.close_all_positions()