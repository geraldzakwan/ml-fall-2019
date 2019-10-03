import numpy as np

def load_csv_to_ndarray(filepath):
    # Read using numpy lib
    data = np.genfromtxt(filepath, dtype=float, delimiter=',')

    # Return all rows except the first one
    return data[1:len(data)]

def create_label_vector(data):
    # Extract first column as label vector -> shape (4920,)
    return data[:, 0]

def create_feature_matrix(data):
    # Extract the rest of the columns as feature matrix -> shape (4920, 8)
    return data[:, 1:len(data[0])]

def add_intercept(feature_matrix):
    # Create (1.0, ..., 1.0) of length equals to feature_matrix rows
    intercept = np.full(len(feature_matrix), 1.0)

    # Add that vector as the first column of feature_matrix
    return np.append(intercept[:, None], feature_matrix, 1)

# Main training function, basically following this equation:
# b = inverse(X'X)X'y where:
# 1. b -> the params we seek
# 2. X -> feature matrix from train data (X' denotes the transpose)
# 3. y -> label from train data
def train_params(feature_matrix, label_vector):
    # Get X'
    feature_matrix_transposed = feature_matrix.T

    # Compute X'X
    feature_matrix_matmul = np.dot(feature_matrix_transposed, feature_matrix)

    try:
        # Compute inverse(X'X), in this case, X'X matrix is invertible
        inverse_matmul = np.linalg.inv(feature_matrix_matmul)
    except numpy.linalg.LinAlgError:
        print('MATRIX NOT INVERTIBLE')
        raise Exception

    # Compute inverse(X'X)X'y
    return np.dot(np.dot(inverse_matmul, feature_matrix_transposed), label_vector)

def predict_values(model_params, feature_matrix):
    # Compute X_hat * b to get y_hat
    return np.dot(feature_matrix, model_params)

def calculate_squared_loss(prediction_values, label_vector, mean=True):
    # Calculate sigma (y_hat - y)^2
    total_squared_loss = np.sum((prediction_values - label_vector) ** 2)

    # Divide error by total observations to get mean squared loss
    if mean:
        return total_squared_loss / len(label_vector)

    return total_squared_loss

def predict_label(predict_values, threshold):
    # Return 1 if y_hat is greater than threshold, return 0 otherwise
    return [1 if value > threshold else 0 for value in predict_values]

def calculate_error_rate(predictions, label_vector):
    verdicts = [1 if y_pred == y else 0 for y_pred, y in zip(predictions, label_vector)]

    # Return fraction of wrongly classified instances
    return 1 - (np.sum(verdicts) / len(label_vector))

def calculate_false_positive_rate(predictions, label_vector):
    # Count how many prediction says 1 when it is actually 0
    false_positives = np.sum([1 if y_pred == 1.0 and y == 0.0 else 0 for y_pred, y in zip(predictions, label_vector)])

    # Count how many 0 labels (negative cases)
    total_negatives = np.sum([1 if y == 0.0 else 0 for y in label_vector])

    # Divide false positives count by total negative cases
    return false_positives / total_negatives

if __name__=='__main__':
    train_data = load_csv_to_ndarray('data/compas-train.csv')

    x_train = create_feature_matrix(train_data)
    x_train = add_intercept(x_train)
    y_train = create_label_vector(train_data)

    ols_params = train_params(x_train, y_train)

    # train_prediction_values = predict_values(ols_params, x_train)
    #
    # train_mean_squared_loss = calculate_squared_loss(train_prediction_values, y_train)
    # print('-------------------------------------------------')
    # print('Train mean squared loss: ' + str(train_mean_squared_loss))
    # print()
    #
    # train_predictions = predict_label(train_prediction_values, 0.5)
    #
    # train_error_rate = calculate_error_rate(train_predictions, y_train)
    # print('Train error rate: ' + str(train_error_rate))
    # print()
    #
    # train_false_positive_rate = calculate_false_positive_rate(train_predictions, y_train)
    # print('Train false positive rate: ' + str(train_false_positive_rate))
    # print('-------------------------------------------------')

    test_data = load_csv_to_ndarray('data/compas-test.csv')

    x_test = create_feature_matrix(test_data)
    x_test = add_intercept(x_test)
    y_test = create_label_vector(test_data)

    prediction_values = predict_values(ols_params, x_test)

    mean_squared_loss = calculate_squared_loss(prediction_values, y_test)
    print('-------------------------------------------------')
    print('Test mean squared loss: ' + str(mean_squared_loss))
    print()

    predictions = predict_label(prediction_values, 0.5)

    test_error_rate = calculate_error_rate(predictions, y_test)
    print('Test error rate: ' + str(test_error_rate))
    print()

    test_false_positive_rate = calculate_false_positive_rate(predictions, y_test)
    print('Test false positive rate: ' + str(test_false_positive_rate))
    print('-------------------------------------------------')
