import json, ml_training, feature_extraction, config
import numpy as np

already_calculated_weights = {}
already_calculated_biases = {}
optimal_k_values = {}


# Returns the determined weights of the training data.
# For each dimension, the difference between the largest and smallest value in the training set is used as weight.
def get_weights(profit_boundary, X):
    if already_calculated_weights.get(profit_boundary):
        weights = already_calculated_weights[profit_boundary]
    else:
        weights = ml_training.determine_weights(X)
        already_calculated_weights[profit_boundary] = weights
    return weights


# Bias based on the amount of positive and negative data points in the data. The bias is only calculated once,
# for following predictions the values stored in the already_calculated_biases is used.
def get_bias(profit_boundary, y):
    if already_calculated_biases.get(profit_boundary):
        bias = already_calculated_biases[profit_boundary]
    else:
        bias = [0, 0]
        for label in y:
            bias[label] += 1
        already_calculated_biases[profit_boundary] = bias
    return bias


# Loads the hyperparameters for all KNN models and stores them in the optimal_k_values dictionary
def load_hyperparameters():
    file_path = config.models_file_path
    input_file = open(file_path)
    hyperparameters = json.load(input_file)
    input_file.close()

    for items in hyperparameters.items():
        # items[0] is the individual profit_boundary
        optimal_k_values[items[0]] = items[1]["Optimal k"]


# Makes a prediction whether the stock whose features are given as the data_point list will increase above the given
# profit_boundary.
def predict(profit_boundary, data_point):
    profit_boundary = str(profit_boundary)
    if not optimal_k_values.get(profit_boundary):
        return 0
    k = optimal_k_values[profit_boundary]

    X, y = feature_extraction.load_data_points()
    # Sets the labels to either 1 or 0 depending on whether the stock price increased above the given profit_boundary
    for i in range(len(y)):
        if y[i] >= profit_boundary:
            y[i] = 1
        else:
            y[i] = 0

    # Weights for each dimension based on the difference between the largest and smallest value
    # of all data points in the training data.
    weights = get_weights(profit_boundary, X)
    # Applies weights to the existing data and the given data_point
    X = np.divide(X, weights)
    data_point = np.divide(data_point, weights)

    # Bias based on the amount of positive and negative data points in the data
    bias = get_bias(profit_boundary, y)

    # Calculate distances from the given data_point to all data points in the existing data
    h = np.subtract(X, data_point)
    h = np.multiply(h, h)
    sums = np.sum(h, axis=1)
    distances = np.sqrt(sums)

    # Determines the labels of the k nearest neighbors
    nearest_labels = []
    for j in range(k):
        result = np.where(distances == np.amin(distances))
        index = result[0][0]
        nearest_labels.append(y[index])
        distances = np.delete(distances, index)
        y = np.delete(y, index)

    # Counts the amount of negative and positive labels in the nearest_labels list
    counters = [0, 0]
    for label in nearest_labels:
        counters[label] += 1

    # Makes a prediction based on the labels of the k nearest neighbors multiplied with the previously determined bias
    if counters[1] * bias[0] > counters[0] * bias[1]:
        return 1
    else:
        return 0


# Returns a dictionary, ranking the given data points representing stocks based on the expected stock price increase.
def predict_most_likely_profitable(data_points):
    load_hyperparameters()

    profit_boundaries = [1.011, 1.016, 1.021, 1.026, 1.031, 1.036, 1.041, 1.046, 1.051, 1.056, 1.061, 1.066, 1.071,
                         1.076, 1.081, 1.086, 1.091, 1.096, 1.101]
    expected_increase = {}
    for stock_sym in data_points:
        data_point = data_points[stock_sym]
        for profit_boundary in profit_boundaries:
            if predict(profit_boundary, data_point) == 1:
                # Individual model predicts stock it will increase above the specific profit_boundary
                expected_increase[stock_sym] = profit_boundary
            else:
                # Individual model makes negative prediction
                break
    # Sorts the ranking by the expected profit
    most_likely_ranking = {k: v for k, v in sorted(expected_increase.items(), key=lambda item: item[1], reverse=True)}
    return most_likely_ranking