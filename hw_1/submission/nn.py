from helper import *

def nn(X, Y, test):
    start = time.time()
    preds = []

    distances_matrix = calculate_distances_matrix(test, X)
    for row in distances_matrix:
        min_idx = np.argmin(row)
        preds.append(Y[min_idx])

    print(str(time.time()-start) + ' seconds')
    return preds

if __name__ == '__main__':
    ocr = load_ocr()

    mean_errors = []
    std_errors = []

    num_trials = 10
    train_sizes = [1000, 2000, 4000, 8000]

    for n in train_sizes:
        test_err = np.zeros(num_trials)
        print('Executing for n = ' + str(n))

        for trial in range(num_trials):
            sel = sample(range(len(ocr['data'].astype('float'))),n)
            preds = nn(ocr['data'].astype('float')[sel], ocr['labels'][sel], ocr['testdata'].astype('float'))
            test_err[trial] = np.mean(preds != ocr['testlabels'])

        mean_errors.append(np.mean(test_err))
        std_errors.append(np.std(test_err))

        print('--------------------')

    print('train_size, mean_error, standard_deviation')
    for i in range(0, 4):
        print(str(train_sizes[i]) + ', ' + str(mean_errors[i]) + ', ' + str(std_errors[i]))

    print('--------------------')

    # By default, Mac OS X can't directly render matplotlib
    # To use matplotlib, please use Jupyter Notebook or set the backend properly (https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python)
    # plt = plot_learning_curve('Nearest Neighbor Learning Curve', train_sizes, mean_errors, std_errors)
    # plt.savefig('nearest_neighbor_learning_curve.png', bbox_inches='tight')
