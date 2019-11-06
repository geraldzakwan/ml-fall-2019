import numpy as np
import scipy
from scipy import optimize

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

# Use np.linalg.lstsq to compute the minimum Euclidean norm solution
def compute_min_euclidean_norm_solution(feature_matrix, label_vector, use_pinv=False):
    if use_pinv:
        return np.dot(np.linalg.pinv(feature_matrix), label_vector)

    # Get X'
    feature_matrix_transposed = feature_matrix.T

    # Compute X'X (covariance matrix)
    feature_matrix_matmul = np.dot(feature_matrix_transposed, feature_matrix)

    # Basically, lstsq(M_1, M_2) will return a solution, let say M_sol
    # M_sol has the least squares or l2 norms amongst other solutions (there can also be no solution)
    # M_sol fulfills the equation: np.dot(M_1, M_sol) = M_2
    # In our case, M_1 would be X'X, M_2 would be X'y and M_sol would be the weights we seek
    return np.linalg.lstsq(feature_matrix_matmul, np.dot(feature_matrix_transposed, label_vector), rcond=None)[0]

def create_weights_diagonal_matrix(squared=False):
    weights = np.zeros(2*128)

    for j in range(0, 128):
        weights[2*j] = np.ceil((j+1)/2)
        if squared:
            weights[2*j] = weights[2*j] ** 2

        weights[(2*j)+1] = np.ceil((j+1)/2)
        if squared:
            weights[(2*j)+1] = weights[(2*j)+1] ** 2

    return weights, np.diag(weights)

def count_nonzero_entries(weights, threshold=1.e-15):
    count = 0

    for weight in weights:
        if abs(weight) < threshold:
            count = count + 1

    return count

def calculate_squared_loss(prediction_values, label_vector, mean=True):
    # Calculate sigma (y_hat - y)^2
    total_squared_loss = np.sum((prediction_values - label_vector) ** 2)

    # Divide error by total observations to get mean squared loss
    if mean:
        return total_squared_loss / len(label_vector)

    return total_squared_loss

def run_experiment_1a(use_pinv=False):
    print('Experiment 1a: ')
    print()

    train_df = load_csv_to_ndarray('hw3p1_train.csv')
    test_df = load_csv_to_ndarray('hw3p1_test.csv')

    one_dim_feature_vector = train_df[:,0]
    label_vector = train_df[:,1]

    feature_matrix = transform_feature(one_dim_feature_vector)

    sol = compute_min_euclidean_norm_solution(feature_matrix, label_vector, use_pinv)

    test_feature_matrix = transform_feature(test_df)
    test_pred_result = np.dot(test_feature_matrix, sol)

    print('Mean squared loss: ')
    print(calculate_squared_loss(np.dot(feature_matrix, sol), label_vector))
    print()

    print('Number of nonzero entries:')
    print(count_nonzero_entries(sol))
    print()

    print('--------------------------------')

    # # By default, Mac OS X can't directly render matplotlib
    # # To use matplotlib, please use Jupyter Notebook or set the backend properly (https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python)
    # import matplotlib.pyplot as plt
    #
    # fig = plt.figure(figsize=(10, 10))
    #
    # plt.title('Minimum Euclidean Plot')
    # plt.plot(test_df, test_pred_result, color='pink', linewidth=1, marker='o', markersize=3, mfc='white', mec='black')
    #
    # plt.savefig('1a.png', bbox_inches='tight')
    #
    # plt.show()

def run_experiment_1b(use_pinv=False):
    print('Experiment 1b: ')
    print()

    train_df = load_csv_to_ndarray('hw3p1_train.csv')
    test_df = load_csv_to_ndarray('hw3p1_test.csv')

    one_dim_feature_vector = train_df[:,0]
    label_vector = train_df[:,1]

    feature_matrix = transform_feature(one_dim_feature_vector)

    _, weights_diagonal_matrix = create_weights_diagonal_matrix()

    # Stack feature_matrix with weights_diagonal
    weighted_feature_matrix = np.vstack((feature_matrix, weights_diagonal_matrix))
    print(weighted_feature_matrix.shape)

    # Pad label_vector with zeros
    padded_label_vector = np.zeros(2*128 + len(label_vector))
    padded_label_vector[0:len(label_vector)] = label_vector

    sol = compute_min_euclidean_norm_solution(weighted_feature_matrix, padded_label_vector, use_pinv)

    test_feature_matrix = transform_feature(test_df)
    test_pred_result = np.dot(test_feature_matrix, sol)

    print('Mean squared loss: ')
    print(calculate_squared_loss(np.dot(feature_matrix, sol), label_vector))
    print()

    print('Number of nonzero entries:')
    print(count_nonzero_entries(sol))
    print()

    print('--------------------------------')

    # # By default, Mac OS X can't directly render matplotlib
    # # To use matplotlib, please use Jupyter Notebook or set the backend properly (https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python)
    # import matplotlib.pyplot as plt
    #
    # fig = plt.figure(figsize=(10, 10))
    #
    # plt.title('Minimum Weighted Euclidean Plot')
    # plt.plot(test_df, test_pred_result, color='pink', linewidth=1, marker='o', markersize=3, mfc='white', mec='black')
    #
    # plt.savefig('1b.png', bbox_inches='tight')
    #
    # plt.show()

def run_experiment_1c():
    print('Experiment 1c: ')
    print()

    train_df = load_csv_to_ndarray('hw3p1_train.csv')
    test_df = load_csv_to_ndarray('hw3p1_test.csv')

    one_dim_feature_vector = train_df[:,0]
    label_vector = train_df[:,1]

    feature_matrix = transform_feature(one_dim_feature_vector)

    # 256
    weights, _ = create_weights_diagonal_matrix()

    # 512
    appended_weights = np.zeros(2*2*128)

    # First 256 is for w, last 256 for v
    # Give zero weights to w cause we don't give a shit about sum(w)
    # We care about sum(abs(w)), i.e. sum(v)
    appended_weights[2*128:2*2*128] = weights

    # Equality constraint

    # A^tA
    A_eq = np.dot(feature_matrix.transpose(), feature_matrix)
    # Add 256 more zero columns to fit for v
    A_eq = np.hstack((A_eq, np.zeros((2*128, 2*128))))
    print(A_eq.shape)
    # Label vector stay as is, A^tb will have shape 256
    b_eq = np.dot(feature_matrix.transpose(), label_vector)
    print(b_eq.shape)

    # Inequality constraint

    # Each pair of v and w needs to follow: - w - v <= 0 and w - v <= 0
    A_ub = np.zeros((512, 512))
    for i in range(0, int(len(A_ub)/2)):
        # -w_j - v_j
        A_ub[2*i][i] = -1
        A_ub[2*i][i+256] = -1

        # w_j - v_j
        A_ub[(2*i)+1][i] = 1
        A_ub[(2*i)+1][i+256] = -1

    # Upper bound <= 0
    b_ub = np.zeros(512)

    # No need to set bound for v, because the default is nonnegative
    # Set bound for w from -inf to inf (None, None)
    # Set bound for v from 0 to inf (0, None)
    bounds = []
    for i in range(0, 256):
        bounds.append((None, None))
    for i in range(0, 256):
        bounds.append((0, None))

    res = scipy.optimize.linprog(appended_weights, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='revised simplex')

    dantzig_sol = res.x[0:256]

    print('Mean squared loss: ')
    print(calculate_squared_loss(np.dot(feature_matrix, dantzig_sol), label_vector))
    print()

    print('Number of nonzero entries:')
    print(count_nonzero_entries(dantzig_sol))
    print()

    print('--------------------------------')

    # # By default, Mac OS X can't directly render matplotlib
    # # To use matplotlib, please use Jupyter Notebook or set the backend properly (https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python)
    # import matplotlib.pyplot as plt
    #
    # test_feature_matrix = transform_feature(test_df)
    # test_pred_result = np.dot(test_feature_matrix, dantzig_sol)
    #
    # fig = plt.figure(figsize=(10, 10))
    #
    # plt.title('Dantzig Selector Euclidean Plot')
    # plt.plot(test_df, test_pred_result, color='pink', linewidth=1, marker='o', markersize=3, mfc='white', mec='black')
    #
    # plt.savefig('1c.png', bbox_inches='tight')
    #
    # plt.show()

if __name__ == '__main__':
    run_experiment_1a(False)
    run_experiment_1a(True)

    run_experiment_1b(False)
    run_experiment_1b(True)

    run_experiment_1c()
