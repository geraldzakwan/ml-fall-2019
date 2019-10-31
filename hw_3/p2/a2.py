from helper import load_news_data, create_dictionary

if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = load_news_data('news.mat')

    # Convert train and test data to np array
    train_data = train_data.todense()
    # print(train_data.shape)
    # (11269, 61188)

    # So, basically 61188 is the vocab size
    # It is a one hot encoding
    # 11269 is the total sentence

    # Convert test data to np array
    test_data = test_data.todense()

    print(test_data.shape)
    # (7505, 61188)

    word_dict = create_dictionary('news.vocab')
    # print(len(word_dict))
    # 61188

    
