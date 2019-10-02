from helper import *
from sklearn.model_selection import KFold

def get_knn_label(dist_vector, label_vector, k):
    k_smallest_indexes = np.argpartition(dist_vector, k)[:k]
    count_dict = {}
    for idx in k_smallest_indexes:
        label = label_vector[idx][0]
        if label not in count_dict:
            count_dict[label] = 1
        else:
            count_dict[label] = count_dict[label] + 1
    # Note that if k is even and there are more than one labels with highest count,
    # the 'lowest' label is taken, e.g. label '1' is more favorable than lable '2'
    # max_value = max(count_dict.values())
    # max_keys = [k for k, v in count_dict.items() if v == max_value]
    # if(len(max_keys)==1):
    #     final_label = max_keys[0]
    # else:
    #     final_label = sorted(max_keys)[0]
    final_label = max(count_dict, key=lambda key: count_dict[key])
    return np.array([final_label])

def knn(X, Y, test, k):
    start = time.time()
    preds = []

    distances_matrix = calculate_distances_matrix(test, X)
    for row in distances_matrix:
        preds.append(get_knn_label(row, Y, k))

    print(str(time.time()-start) + " seconds")
    return preds

if __name__ == '__main__':
    ocr = load_ocr()

    train_data = ocr['data'].astype('float')
    labels = ocr['labels']
    splitted_data = []
    splitted_labels = []

    kf = KFold(n_splits=10)
    for train, validation in kf.split(train_data):
        splitted_data.append((train_data[train], train_data[validation]))
        splitted_labels.append((labels[train], labels[validation]))

    mean_errors = []
    for k in range(1, 11):
        print('Executing for k = ' + str(k))

        test_err = np.zeros(10)
        for i in range(0, len(splitted_data)):
            preds = knn(splitted_data[i][0], splitted_labels[i][0], splitted_data[i][1], k)
            test_err[i] = np.mean(preds != splitted_labels[i][1])

        mean_errors.append(np.mean(test_err))

        print('--------------------')

    for k in range(1, 11):
        print("k = " + str(k) + ", cross validation error rate = " + str(mean_errors[k-1]))

    optimal_k = np.argmin(mean_errors) + 1
    print('Best model achieved for k=' + str(optimal_k))
    print('--------------------')

    test_preds = knn(train_data, labels, ocr['testdata'].astype('float'), optimal_k)
    optimal_k_test_error = np.mean(test_preds != ocr['testlabels'])

    print('For k=' + str(optimal_k) + ', test error rate is ' + str(optimal_k_test_error))
    print('--------------------')
