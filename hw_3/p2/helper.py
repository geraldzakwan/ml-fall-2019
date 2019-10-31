from scipy.io import loadmat

def load_news_data(filepath):
    news = loadmat(filepath)

    train_data = news['data']
    train_labels = news['labels']

    test_data = news['testdata']
    test_labels = news['testlabels']

    return train_data, train_labels, test_data, test_labels

def create_dictionary(filepath):
    with open(filepath, 'r') as f:
        list_of_words = f.readlines()

    return list_of_words
