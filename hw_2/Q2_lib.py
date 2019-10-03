# TO DO:
# 1. Create linear regression -> confirm if it is needed to be from scratch or we can use sklearn
# 2. The function is called eta, compute square loss to the test data
# 3. Second function is classifier, basically takes threshold and output binary class, compute error & FP rate

import numpy as np
import statsmodels.api as sm

def load_csv_to_ndarray(filepath):
    with open(filepath, 'r') as f:
        header = f.readline()

    data = np.genfromtxt(filepath, dtype=float, delimiter=',')

    return header.strip('\n').split(','), data[1:len(data)]

def create_feature_matrix(data):
    return data[:, 1:len(data[0])]

def create_label_vector(data):
    return data[:, 0]

def add_intercept(feature_matrix):
    # Create (1.0, ..., 1.0) of length equals to feature_matrix rows
    intercept = np.full(len(feature_matrix), 1.0)
    # Add intercept as the first column of feature_matrix
    return np.append(intercept[:, None], feature_matrix, 1)

def calculate_intercept(model_params, feature_matrix, label_vector):
    diff_vector = label_vector - np.dot(feature_matrix, model_params)

    return np.sum(diff_vector) / len(diff_vector)

def train_params(feature_matrix, label_vector):
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
    # x_train = sm.add_constant(x_train)
    y_train = create_label_vector(train_data)

    model = sm.OLS(y_train, x_train)
    results = model.fit()

    print(results.summary())

    print(results.mse_total)

    ypred = results.predict(x_train[0:10])
    print(ypred)
