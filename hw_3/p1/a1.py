import numpy as np
import matplotlib.pyplot as plt

def load_csv_to_ndarray(filepath):
    # Read using numpy lib and return all rows, no header in this case
    return np.genfromtxt(filepath, dtype=float, delimiter=',')

def transform_feature(one_dim_feature_vector):
    transformed_feature_vector = np.zeros((len(one_dim_feature_vector), 2*128))

    for i in range(0, len(one_dim_feature_vector)):
        feature = one_dim_feature_vector[i]
        transformed_feature = np.zeros(2*128)

        for j in range(0, 128):
            transformed_feature[2*j] = np.cos((j+1) * feature)

            transformed_feature[(2*j)+1] = np.sin((j+1) * feature)

        transformed_feature_vector[i] = transformed_feature

    return transformed_feature_vector

def create_weights_diagonal_matrix(squared=False):
    # Shape (256,)
    weights = np.zeros(2*128)

    for j in range(0, 128):
        weights[2*j] = np.ceil((j+1)/2)
        if squared:
            weights[2*j] = weights[2*j] ** 2

        weights[(2*j)+1] = np.ceil((j+1)/2)
        if squared:
            weights[(2*j)+1] = weights[(2*j)+1] ** 2

    return weights, np.diag(weights)

# Use np.linalg.lstsq to compute the minimum Euclidean norm solution
def compute_min_euclidean_norm_solution(feature_matrix, label_vector):
    # Get X'
    feature_matrix_transposed = feature_matrix.T

    # Compute X'X (covariance matrix)
    feature_matrix_matmul = np.dot(feature_matrix_transposed, feature_matrix)

    # Basically, lstsq(M_1, M_2) will return a solution, let say M_sol
    # M_sol has the least squares or l2 norms amongst other solutions (there can also be no solution)
    # M_sol fulfills the equation: np.dot(M_1, M_sol) = M_2
    # In our case, M_1 would be X'X, M_2 would be X'y and M_sol would be the weights we seek
    return np.linalg.lstsq(feature_matrix_matmul, np.dot(feature_matrix_transposed, label_vector), rcond=None)[0]

def run_experiment_1a():
    train_df = load_csv_to_ndarray('hw3p1_train.csv')
    test_df = load_csv_to_ndarray('hw3p1_test.csv')

    one_dim_feature_vector = train_df[:,0]
    label_vector = train_df[:,1]

    feature_matrix = transform_feature(one_dim_feature_vector)

    sol = compute_min_euclidean_norm_solution(feature_matrix, label_vector)

    test_feature_matrix = transform_feature(test_df)
    test_pred_result = np.dot(test_feature_matrix, sol)

    # By default, Mac OS X can't directly render matplotlib
    # To use matplotlib, please use Jupyter Notebook or set the backend properly (https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python)

    # fig = plt.figure(figsize=(10, 10))
    #
    # plt.title('Min Euclidean Plot')
    # plt.plot(test_df, test_pred_result, color='pink', linewidth=1, marker='o', markersize=3, mfc='white', mec='black')
    #
    # plt.savefig('1a.png', bbox_inches='tight')

if __name__ == '__main__':
    run_experiment_1a()
