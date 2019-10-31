from helper import load_csv_to_ndarray

if __name__ == '__main__':
    train_df = load_csv_to_ndarray('hw3p1_train.csv')
    test_df = load_csv_to_ndarray('hw3p1_test.csv')

    print(train_df.shape)
    print(test_df.shape)
