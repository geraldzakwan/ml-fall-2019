# Identity
Name: Geraldi Dzakwan
UNI : gd2551

# Environment

- Please use Python 3
- Please put 'ocr.mat' file inside 'data' folder
- Pleas run pip install -r requirements.txt to install all the needed libraries, in this case:
  1. numpy, for vector/matrix operations
  2. scipy, for loading data
  3. matplotlib, for plotting
  - By default, Mac OS X can't directly render matplotlib
  - To use matplotlib, please use Jupyter Notebook or set the backend properly (https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python)
  4. sklearn, only for cross validation

# module helper.py

Contains 3 functions:

1. load_ocr -> To load 'ocr.mat'
2. calculate_distances_matrix -> Explained in the write-up
3. plot_learning_curve -> To plot learning curve for problem 2b

# module nn.py

Run this module to get solution for problem 2b, it consists of two components:

1. Function nn is the nearest neighbor algorithm -> Explained in the write-up
2. Main function, which is mostly taken from the skeleton code

# module knn.py

Run this module to get solution for problem 2c, it consists of three components:

1. Function knn is the k-nearest neighbor algorithm -> Explained in the write-up

2. Function get_knn_label -> Explained in the write up

3. Main function, to do cross validation, logging cross validation error rate, get optimal k and logging test error rate
