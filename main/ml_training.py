import feature_extraction, json, config
import numpy as np
from sklearn.model_selection import train_test_split


# Determines the optimal k for a single KNN model.
def train_single_knn_np(profit_boundary, remake_features=False):
    k_values = range(5, 51)

    if not remake_features:
        try:
            # Loads the already extracted features
            X, y = feature_extraction.load_data_points()
        except FileNotFoundError:
            # In case already extracted features do not exist at the expected file path,
            # the feature extraction process is repeated.
            remake_features = True
    if remake_features:
        # Extracts features from historic stock data
        X, y = feature_extraction.extract_features_from_daily_stock_data()

    # Sets the labels to either 1 or 0 depending on whether the stock price increased above the given profit_boundary
    for i in range(len(y)):
        if y[i] >= profit_boundary:
            y[i] = 1
        else:
            y[i] = 0

    # Splits the data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Weights for each dimension based on the difference between the largest and smallest value
    # of all data points in the training data.
    weights = determine_weights(X_train)
    # Apply weights
    X_train = np.divide(X_train, weights)
    X_test = np.divide(X_test, weights)

    # Bias based on the amount of positive and negative data points in the training data
    bias = [0, 0]
    for label in y_train:
        bias[label] += 1

    k_with_max_precision = max(k_values)
    counter_tp = {}
    counter_fp = {}
    for k in k_values:
        counter_tp[k] = 0
        counter_fp[k] = 0

    # For each data point in the test set
    for i in range(len(X_test)):
        # List that will contain the labels for all data points in the training data
        # except the label for the currently inspected data point instance0.
        modified_y_train = y_train
        instance0 = np.array(X_test[i])
        instance0_label = y_test[i]

        # Calculates the distance for all objects in the training data set to instance0, a data point in the test set.
        h = np.subtract(X_train, instance0)
        h = np.multiply(h, h)
        sums = np.sum(h, axis=1)
        distances = np.sqrt(sums)

        nearest_labels = []
        for j in range(k_with_max_precision):
            # Returns the index of the data point with the smallest distance, appends the label of this data point to
            # the nearest_labels list and then removes the distance of this point from the distances list and the
            # label of this point from the modified_y_train list, so that in the next iteration index will be the
            # index of the data point with the next smallest distance.
            result = np.where(distances == np.amin(distances))
            index = result[0][0]
            nearest_labels.append(modified_y_train[index])
            distances = np.delete(distances, index)
            modified_y_train = np.delete(modified_y_train, index)

        # In case there are not enough data points in the training set
        if k_with_max_precision > len(nearest_labels):
            raise ValueError

        # Counters for each considered value for the parameter k.
        # The first value counts how many of the k nearest neighbors of instance0 are unprofitable,
        # the second value counts how many of the k nearest neighbors are profitable.
        counters = {}
        for k in k_values:
            counters[k] = [0, 0]

        # Counts for each considered value for the parameter k how many of the k nearest neighbours of instance0
        # are profitable and how many are not.
        for j in range(k_with_max_precision):
            for k in k_values:
                if j < k:
                    label = nearest_labels[j]
                    counters[k][label] += 1

        for k in k_values:
            # Determines for each considered value for k if it would make a positive prediction
            if counters[k][1] * bias[0] > counters[k][0] * bias[1]:
                if instance0_label == 1:
                    # Prediction is correct
                    counter_tp[k] += 1
                else:
                    # Prediction is false
                    counter_fp[k] += 1

    # Determines for which k the highest precision is achieved
    max_precision = 0
    k_with_max_precision = None
    tp_for_max_precision = 0
    for k in k_values:
        # To avoid division by 0
        if counter_tp[k] == 0 and counter_fp[k] == 0:
            continue
        precision = counter_tp[k] / (counter_tp[k] + counter_fp[k])
        if precision > max_precision:
            # new max_precision found
            k_with_max_precision = k
            max_precision = precision
            tp_for_max_precision = counter_tp[k]
    # Hyperparameters for this model are updated in case this configuration is better than the already existing one
    update_hyperparameters(profit_boundary, k_with_max_precision, max_precision, tp_for_max_precision)


# Checks if the new hyperparameters are better than the current selected ones,
# if so the new hyperparameters are stored in the corresponding file.
def update_hyperparameters(profit_boundary, max_k, max_precision, tp_for_max_precision):
    profit_boundary = str(profit_boundary)
    file_path = config.models_file_path
    try:
        input_file = open(file_path)
        model_data = json.load(input_file)
        input_file.close()

        if model_data.get(profit_boundary):
            # A configuration for this model already exists
            old_precision = model_data[profit_boundary]["Precision"]
            # New configuration is better than existing one
            if max_precision > old_precision:
                model_data[profit_boundary] = {"Optimal k": max_k, "Precision": max_precision,
                                               "TP": tp_for_max_precision}
                store_hyperparameters(model_data, file_path)
        else:
            # No existing configuration is stored for this model. Therefore. the given configuration is stored.
            model_data[profit_boundary] = {"Optimal k": max_k, "Precision": max_precision,
                                           "TP": tp_for_max_precision}
            store_hyperparameters(model_data, file_path)
    except FileNotFoundError:
        # New file is created
        model_data = {
            profit_boundary: {"Optimal k": max_k, "Precision": max_precision, "TP": tp_for_max_precision}}
        store_hyperparameters(model_data, file_path)


# Stores the given model_data in a JSON file.
def store_hyperparameters(model_data, file_path):
    out_file = open(file_path, 'w')
    json.dump(model_data, out_file)
    out_file.close()


# Weights for the dimension used by the KNN model.
# For each dimension, the difference between the largest and smallest value in the training set is used as weight.
def determine_weights(X_train):
    max_values = []
    min_values = []
    for data_point in X_train:
        for dimension in range(len(data_point)):
            # In case the max_value in the max_values list is not yet instantiated for this dimension
            if len(max_values) <= dimension:
                max_values.append(data_point[dimension])
            # In case the min_value in the min_values list is not yet instantiated for this dimension
            if len(min_values) <= dimension:
                min_values.append(data_point[dimension])
                continue
            # In case value of the currently inspected data point is larger than the current max_value,
            # the value of the data point is set as the new max_value.
            if data_point[dimension] > max_values[dimension]:
                max_values[dimension] = data_point[dimension]
            # In case value of the currently inspected data point is less than the current min_value,
            # the value of the data point is set as the new min_value.
            if data_point[dimension] < min_values[dimension]:
                min_values[dimension] = data_point[dimension]
    weights = []
    for dimension in range(len(max_values)):
        dif = max_values[dimension] - min_values[dimension]
        weights.append(dif)
    return weights


# Starts the training process, in which optimal k is determined for each of the 19 models.
# Each model predicts whether a stock will increase above an individual profit_boundary during the next day.
def train_knn():
    for profit_boundary in [1.011, 1.016, 1.021, 1.026, 1.031, 1.036, 1.041, 1.046, 1.051, 1.056, 1.061, 1.066, 1.071,
                            1.076, 1.081, 1.086, 1.091, 1.096, 1.101]:
        train_single_knn_np(profit_boundary=profit_boundary)