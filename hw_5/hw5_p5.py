import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits

# import matplotlib.pyplot as plt

class OneLayerNet(nn.Module):
  def __init__(self):
    super(OneLayerNet, self).__init__()
    self.d1 = nn.Linear(2, 1, True)

  def forward(self, x):
    x = self.d1(x)
    return x

class TwoLayerNet(nn.Module):
  def __init__(self):
    super(TwoLayerNet, self).__init__()
    self.d1 = nn.Linear(2, 2, True)
    self.d2 = nn.Linear(2, 1, True)

  def forward(self, x):
    x = self.d1(x)
    x = F.relu(x)

    x = self.d2(x)
    return x

class ThreeLayerNet(nn.Module):
  def __init__(self):
    super(ThreeLayerNet, self).__init__()
    self.d1 = nn.Linear(64, 64, True)
    self.d2 = nn.Linear(64, 32, True)
    self.d3 = nn.Linear(32, 1, True)

  def forward(self, x):
    # print(x.size())
    x = self.d1(x)
    x = F.relu(x)
    x = self.d2(x)
    x = F.relu(x)
    x = self.d3(x)

    return x

class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
    self.conv2 = nn.Conv2d(8, 4, kernel_size=3)
    self.d1 = nn.Linear(4, 1, True)

  def forward(self, x):
    # print(x.size())
    # print(x.unsqueeze_(1).size())
    x = x.unsqueeze(1)
    x = self.conv1(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)
    x = self.conv2(x)
    x = F.relu(x)
    x = x.view(-1, 4)
    x = self.d1(x)

    return x

def XOR_data():
  X = torch.tensor([[-1., -1.], [1., -1.], [-1., 1.], [1., 1.]])
  Y = (torch.prod(X, dim=1) < 0.).float()
  return X, Y.view(-1,1)

def digits_data():
  digits, labels = load_digits(return_X_y=True)
  digits = torch.tensor(digits.reshape([-1, 8, 8]), dtype=torch.float)
  labels = torch.tensor(labels.reshape([-1, 1]) % 2 == 1, dtype=torch.float)
  test_digits = digits[:180,:,:]
  test_labels = labels[:180]
  digits = digits[180:,:,:]
  labels = labels[180:]
  return digits, labels, test_digits, test_labels

def gradient_descent(net, X, Y, num_iterations, eta):
  objective_fn = nn.BCEWithLogitsLoss()

  objective_values = []
  error_rates = []

  # with torch.no_grad():
  #   l1 = objective_fn(net(X), Y).item()
  #   err1 = error_rate(net(X), Y).item()

  # objective_values.append(l1)
  # error_rates.append(err1)

  optimizer = torch.optim.SGD(net.parameters(), lr = eta)

  # net.train()

  for epoch in range(num_iterations + 1):
      optimizer.zero_grad()

      loss = objective_fn(net(X), Y)
      err = error_rate(net(X), Y)

      objective_values.append(loss.item())
      error_rates.append(err.item())

      if epoch < num_iterations:
        loss.backward()
        optimizer.step()

  return objective_values, error_rates

def error_rate(Yhat, Y):
  return ((torch.sign(Yhat) > 0).float() != Y).float().mean()

XOR_X, XOR_Y = XOR_data()
digits, labels, test_digits, test_labels = digits_data()

num_iterations = 25
eta = 1.0

# train one-layer net on XOR data
torch.manual_seed(0)
net1 = OneLayerNet()
a, b = gradient_descent(net1, XOR_X, XOR_Y, num_iterations, eta)

print(a[0])
print(b[0])
print(a[-1])
print(b[-1])

# fig = plt.figure(figsize=(10, 10))
#
# plt.title('Objective Values Plot 5a')
# plt.plot(np.arange(num_iterations + 1), a, color='pink', linewidth=1, marker='o', markersize=3, mfc='white', mec='black')
#
# plt.savefig('5a-obj.png', bbox_inches='tight')
#
# plt.show()
#
# fig = plt.figure(figsize=(10, 10))
#
# plt.title('Error Rate Plot 5a')
# plt.plot(np.arange(num_iterations + 1), b, color='pink', linewidth=1, marker='o', markersize=3, mfc='white', mec='black')
#
# plt.savefig('5a-err.png', bbox_inches='tight')
#
# plt.show()

torch.manual_seed(0)
net2 = TwoLayerNet()
a, b = gradient_descent(net2, XOR_X, XOR_Y, num_iterations, eta)

print(a[0])
print(b[0])
print(a[-1])
print(b[-1])

# fig = plt.figure(figsize=(10, 10))
#
# plt.title('Objective Values Plot 5c')
# plt.plot(np.arange(num_iterations + 1), a, color='pink', linewidth=1, marker='o', markersize=3, mfc='white', mec='black')
#
# plt.savefig('5c-obj.png', bbox_inches='tight')
#
# plt.show()
#
# fig = plt.figure(figsize=(10, 10))
#
# plt.title('Error Rate Plot 5c')
# plt.plot(np.arange(num_iterations + 1), b, color='pink', linewidth=1, marker='o', markersize=3, mfc='white', mec='black')
#
# plt.savefig('5c-err.png', bbox_inches='tight')
#
# plt.show()

num_iterations = 500
eta = 0.1

torch.manual_seed(0)
net3 = ThreeLayerNet()
a, b = gradient_descent(net3, digits.view(-1, 64), labels, num_iterations, eta)

print(a[0])
print(b[0])
print(a[-1])
print(b[-1])
print("ThreeLayerNet: Test error rate: {0}".format(error_rate(net3(test_digits.view(-1,64)), test_labels)))

# fig = plt.figure(figsize=(10, 10))
#
# plt.title('Objective Values Plot 5e')
# plt.plot(np.arange(num_iterations + 1), a, color='pink', linewidth=1)
#
# plt.savefig('5e-obj.png', bbox_inches='tight')
#
# plt.show()
#
# fig = plt.figure(figsize=(10, 10))
#
# plt.title('Error Rate Plot 5e')
# plt.plot(np.arange(num_iterations + 1), b, color='pink', linewidth=1)
#
# plt.savefig('5e-err.png', bbox_inches='tight')
#
# plt.show()

torch.manual_seed(0)
net4 = ConvNet()
a, b = gradient_descent(net4, digits, labels, num_iterations, eta)

print(a[0])
print(b[0])
print(a[-1])
print(b[-1])
print("ConvNet: Test error rate: {0}".format(error_rate(net4(test_digits), test_labels)))

# fig = plt.figure(figsize=(10, 10))
#
# plt.title('Objective Values Plot 5h')
# plt.plot(np.arange(num_iterations + 1), a, color='pink', linewidth=1)
#
# plt.savefig('5h-obj.png', bbox_inches='tight')
#
# plt.show()
#
# fig = plt.figure(figsize=(10, 10))
#
# plt.title('Error Rate Plot 5h')
# plt.plot(np.arange(num_iterations + 1), b, color='pink', linewidth=1)
#
# plt.savefig('5h-err.png', bbox_inches='tight')
#
# plt.show()
