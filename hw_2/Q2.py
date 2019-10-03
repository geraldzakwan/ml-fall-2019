import numpy as np

# TO DO:
# 1. Create linear regression -> confirm if it is needed to be from scratch or we can use sklearn
# 2. The function is called eta, compute square loss to the test data
# 3. Second function is classifier, basically takes threshold and output binary class, compute square loss

def load_csv_to_ndarray(filepath):
    with open(filepath, 'r') as f:
        header = f.readline()

    data = np.genfromtxt(filepath, dtype=float, delimiter=',')

    return header.strip('\n').split(','), data[1:len(data)]

def create_feature_matrix(data):
    return data[:, 1:len(data[0])]

def create_label_vector(data):
    return data[:, 0]

def calculate_params(feature_matrix, label_vector):
    feature_matrix_transposed = feature_matrix.T

    feature_matrix_matmul = np.dot(feature_matrix_transposed, feature_matrix)

    try:
        inverse_matmul = np.linalg.inv(feature_matrix_matmul)
    except numpy.linalg.LinAlgError:
        print('MATRIX NOT INVERTIBLE')
        raise Exception

    return np.dot(np.dot(inverse_matmul, feature_matrix_transposed), label_vector)

if __name__=='__main__':
    columns, train_data = load_csv_to_ndarray('data/compas-train.csv')

    x_train = create_feature_matrix(train_data)
    y_train = create_label_vector(train_data)

    ols_params = calculate_params(x_train, y_train)

    print(ols_params)
