from scipy.io import loadmat
news = loadmat('news.mat')

train_data = news['data']
train_labels = news['labels']

test_data = news['testdata']
test_labels = news['testlabels']

print(type(train_data))
print(type(train_labels))

print(type(test_data))
print(type(test_labels))
