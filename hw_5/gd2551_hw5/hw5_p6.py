import numpy as np

data = np.genfromtxt('hw5p6data.csv', delimiter=',', names=True, dtype='float')

X = np.array(np.array(list(zip(data['GPA'], data['SAT']))))

y = data['label']

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0, penalty='none', solver='lbfgs').fit(X, y)

y_pred = clf.predict(X)

def compute_error_rate(y, y_pred):
  verdict = y * y_pred
  error = verdict[verdict < 0]
  return len(error)/len(verdict)

print(compute_error_rate(y, y_pred))

# Cyan error
cyan_examples = y[0:500]
cyan_preds = y_pred[0:500]
print(compute_error_rate(cyan_examples, cyan_preds))

# Red error
red_examples = y[500:600]
red_preds = y_pred[500:600]
print(compute_error_rate(red_examples, red_preds))

# Cyan P(f_mle(X) = +1)
cyan_1_total = cyan_preds[cyan_preds == 1]
cyan_prob = len(cyan_1_total)/len(cyan_preds)
print(cyan_prob)

# Red P(f_mle(X) = +1)
red_1_total = red_preds[red_preds == 1]
red_prob = len(red_1_total)/len(red_preds)
print(red_prob)

classification_discrepancy = np.abs(cyan_prob - red_prob)
print(classification_discrepancy)

cyan_positive_idx = np.where(cyan_examples == 1)

cyan_should_be_positive = cyan_preds[cyan_positive_idx]
cyan_false_negative = cyan_should_be_positive[cyan_should_be_positive == -1]
cyan_fn_rate = len(cyan_false_negative)/len(cyan_preds)
print(cyan_fn_rate)

red_positive_idx = np.where(red_examples == 1)

red_should_be_positive = red_preds[red_positive_idx]
red_false_negative = red_should_be_positive[red_should_be_positive == -1]
red_fn_rate = len(red_false_negative)/len(red_preds)
print(red_fn_rate)

#FNR Discrepancy rate
print(abs(cyan_fn_rate-red_fn_rate))

new_X = X

new_X[:,1][500:600] = np.full(100, np.mean(X[:,1][0:600]))
# print(new_X[:,1][500:600])

clf = LogisticRegression(random_state=0, penalty='none', solver='lbfgs').fit(new_X, y)

y_pred = clf.predict(X)

print(compute_error_rate(y, y_pred))

# Cyan error
cyan_examples = y[0:500]
cyan_preds = y_pred[0:500]
print(compute_error_rate(cyan_examples, cyan_preds))

# Red error
red_examples = y[500:600]
red_preds = y_pred[500:600]
print(compute_error_rate(red_examples, red_preds))

cyan_positive_idx = np.where(cyan_examples == 1)
cyan_positive_idx

cyan_should_be_positive = cyan_preds[cyan_positive_idx]
cyan_false_negative = cyan_should_be_positive[cyan_should_be_positive == -1]
cyan_fn_rate = len(cyan_false_negative)/len(cyan_preds)
print(cyan_fn_rate)

red_positive_idx = np.where(red_examples == 1)

red_should_be_positive = red_preds[red_positive_idx]
red_false_negative = red_should_be_positive[red_should_be_positive == -1]
red_fn_rate = len(red_false_negative)/len(red_preds)
print(red_fn_rate)

#FNR Discrepancy rate
print(abs(cyan_fn_rate-red_fn_rate))
