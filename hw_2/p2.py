from helper import Helper

threshold = 0.5

# TO DO:
# 1. Create linear regression -> confirm if it is needed to be from scratch or we can use sklearn
# 2. The function is called eta, compute square loss to the test data
# 3. Second function is classifier, basically takes threshold and output binary class, compute square loss



if __name__=='__main__':
    print(Helper.load_csv('data/compas-train.csv'))
