from sklearn.datasets import load_diabetes # data set
from sklearn.linear_model import LinearRegression # model fitting
from sklearn.metrics import mean_squared_error, r2_score # mse and r^2
import numpy as np
import matplotlib.pyplot as plt # data visualization

# data set
diabetes = load_diabetes()

# data manipulation
X_ = diabetes.data
y = diabetes.target
X = X_[:, np.newaxis, 2] # only 1 feature (it is simple linear regression)

# shuffling data
n_samples = X.shape[0]
train_size = int(n_samples * 0.8)
indices = np.random.permutation(n_samples)
train_idx = indices[:train_size]
test_idx = indices[train_size:]

# data set splitting
X_train = X[train_idx]
y_train = y[train_idx]
X_test = X[test_idx]
y_test = y[test_idx]

# model fitting
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, y_train)

# y_pred
y_pred = linear_reg_model.predict(X_test)

# mse and r^2
mse = mean_squared_error(y_test, y_pred)
print("mse: ", mse)
r2 = r2_score(y_test, y_pred)
print("r^2: ", r2)

# data visualization
plt.scatter(X_test, y_test, color = "gray")
plt.plot(X_test, y_pred, color = "red")