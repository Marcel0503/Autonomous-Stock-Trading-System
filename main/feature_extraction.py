import csv, os, config
import numpy as np
import pandas as pd


# Creates data points by going through all historic data for each stock and extracting the relevant features
# and the corresponding label for every documented day for which the stock price on the previous day increased
# by more than 20%. Thereby, making the predictions of the machine learning component more accurate.
def extract_features_from_daily_stock_data():
    syms = get_stock_syms()
    read_path = config.daily_stock_data_directory_path
    # Stores the extracted features
    features = []
    # Stores the corresponding labels
    labels = []
    for sym in syms:
        with open(read_path + sym + '_d.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            consider_investing = False
            # Stores the price to which a stock would be bought for
            recent_values = []
            for row in reader:
                # open, high, low, close, volume value of the current day
                o = float(row[1])
                h = float(row[2])
                l = float(row[3])
                c = float(row[4])
                v = int(row[5])
                recent_values.insert(0, (o, h, l, c, v))

                if len(recent_values) <= 21:
                    continue
                if len(recent_values) > 22:
                    recent_values.pop()

                if consider_investing:
                    consider_investing = False
                    # Since the system would buy stocks as soon as the stock market opens,
                    # the open value is seen as the buy price.
                    buy_price = o
                    # Extracts the features from the previous 21 OHLC values
                    feature = make_features(
                        recent_values[1:])
                    features.append(feature)
                    # The highest price increase is used as the label
                    labels.append(h / buy_price)

                # If the stock's price increased by more than 20% and the volume on the last two days
                # was larger than 5000.
                # This is due to the fact, that investing in stocks with a volume below 5000
                # comes with a much higher risk.
                if c > 1.2 * o and recent_values[0][4] > 5000 and recent_values[1][4] > 5000:
                    consider_investing = True

    store_data_points(features, labels)
    return features, labels


# Retrieves the symbols of all stocks stored in the corresponding directory.
def get_stock_syms():
    read_path = config.daily_stock_data_directory_path
    filenames = os.listdir(read_path)
    syms = set()
    for name in filenames:
        syms.add(name[0:len(name) - 6])
    return syms


# Extracts the features from the given OHLC values.
def make_features(recent_values):
    latest_day = recent_values[0]
    o = latest_day[0]
    h = latest_day[1]
    l = latest_day[2]
    c = latest_day[3]
    v = latest_day[4]

    # Relative difference between high and low
    h_l = h / l - 1
    # Relative difference between close and open
    c_o = c / o - 1

    # Moving average over the last seven close values
    ma7 = 0
    for i in range(7):
        ma7 += recent_values[i][3]
    ma7 /= 7

    # Standard-deviation over the last seven close values
    std7 = 0
    for i in range(7):
        std7 += np.square(recent_values[i][3] - ma7)
    std7 /= 6
    std7 = np.sqrt(std7)

    # Moving average over the last 14 close values
    ma14 = 0
    for i in range(14):
        ma14 += recent_values[i][3]
    ma14 /= 14

    # Moving average over the last 21 close values
    ma21 = 0
    for i in range(21):
        ma21 += recent_values[i][3]
    ma21 /= 21

    # Extracted features
    return [h_l, c_o, ma7, ma14, ma21, std7 / ma7, v]


# Stores the extracted features with the corresponding label using a pandas dataframe.
def store_data_points(features, labels):
    columns = ["h/l", "o/c", "ma7", "ma14", "ma21", "7std", "v", "label"]
    # Stores all data points
    data = []
    for i in range(len(features)):
        x = features[i]
        row = []
        for j in range(len(x)):
            row.append(x[j])
        # Appends the corresponding label to the data point
        row.append(labels[i])
        # Appends the data point to the data list.
        data.append(row)
    # Puts the data in a pandas dataframe
    df = pd.DataFrame(data, columns=columns)
    # Stores the dataframe at the given path
    df.to_pickle(config.data_points_file_path)


# Loads the stored data points.
# Returns a list with the extracted features and a second list with the corresponding labels.
def load_data_points():
    df = pd.read_pickle(
        config.data_points_file_path)
    df_dict = df.to_dict("records")
    X = []
    y = []
    for row in df_dict:
        x = []
        for item in row.items():
            if item[0] == "label":
                y.append(item[1])
            else:
                x.append(item[1])
        X.append(x)
    return X, y