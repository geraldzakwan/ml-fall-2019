# TO DO:
# 1. Create linear regression -> confirm if it is needed to be from scratch or we can use sklearn
# 2. The function is called eta, compute square loss to the test data
# 3. Second function is classifier, basically takes threshold and output binary class, compute error & FP rate

import numpy as np

def load_csv_to_ndarray(filepath):
    with open(filepath, 'r') as f:
        header = f.readline()

    data = np.genfromtxt(filepath, dtype=float, delimiter=',')

    return header.strip('\n').split(','), data[1:len(data)]

def create_feature_matrix(data):
    return data[:, 1:len(data[0])]

def create_label_vector(data):
    return data[:, 0]

def add_bias(feature_matrix):
    # Create (1.0, ..., 1.0) of length equals to feature_matrix rows
    bias = np.full(len(feature_matrix), 1.0)
    # Add bias as the first column of feature_matrix
    return np.append(bias[:, None], feature_matrix, 1)

def calculate_params(feature_matrix, label_vector):
    feature_matrix_transposed = feature_matrix.T

    feature_matrix_matmul = np.dot(feature_matrix_transposed, feature_matrix)

    try:
        inverse_matmul = np.linalg.inv(feature_matrix_matmul)
    except numpy.linalg.LinAlgError:
        print('MATRIX NOT INVERTIBLE')
        raise Exception

    return np.dot(np.dot(inverse_matmul, feature_matrix_transposed), label_vector)

def predict_values(model_params, feature_matrix):
    return np.dot(feature_matrix, model_params)

def calculate_squared_loss(prediction_values, label_vector, average=True):
    total_squared_loss = np.sum((prediction_values - label_vector) ** 2)

    if average:
        return total_squared_loss / len(label_vector)

    return total_squared_loss

def predict_label(predict_values, threshold):
    return [1.0 if value > threshold else 0.0 for value in predict_values]

def calculate_error_rate(predictions, label_vector):
    verdicts = [1 if y_pred == y else 0 for y_pred, y in zip(predictions, label_vector)]

    return np.sum(verdicts) / len(label_vector)

def calculate_false_positive_rate(predictions, label_vector):
    total_false_positives = np.sum([1 if y_pred == 1.0 and y == 0.0 else 0 for y_pred, y in zip(predictions, label_vector)])
    total_negatives = np.sum([1 if y == 0.0 else 0 for y in label_vector])

    return total_false_positives / total_negatives

if __name__=='__main__':
    columns, train_data = load_csv_to_ndarray('data/compas-train.csv')

    x_train = create_feature_matrix(train_data)
    x_train = add_bias(x_train)
    print(x_train[0:10])

    y_train = create_label_vector(train_data)

    ols_params = calculate_params(x_train, y_train)
    print(ols_params)

    train_prediction_values = predict_values(ols_params, x_train)

    train_predictions = predict_label(train_prediction_values, 0.5)

    train_error_rate = calculate_error_rate(train_predictions, y_train)
    print('Train error rate: ' + str(train_error_rate))

    columns, test_data = load_csv_to_ndarray('data/compas-test.csv')

    x_test = create_feature_matrix(test_data)
    x_test = add_bias(x_test)

    y_test = create_label_vector(test_data)

    prediction_values = predict_values(ols_params, x_test)

    average_squared_loss = calculate_squared_loss(prediction_values, y_test)
    print('Test averaged squared loss: ' + str(average_squared_loss))

    predictions = predict_label(prediction_values, 0.5)

    test_error_rate = calculate_error_rate(predictions, y_test)
    print('Test error rate: ' + str(test_error_rate))

    test_false_positive_rate = calculate_false_positive_rate(predictions, y_test)
    print('Test false positive rate: ' + str(test_false_positive_rate))
