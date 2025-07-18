from sklearn.datasets import load_diabetes # data set
from sklearn.model_selection import train_test_split # data set splitting
from sklearn.linear_model import LinearRegression # model fitting
from sklearn.metrics import mean_squared_error, r2_score # rmse and r^2

# data set
diabetes = load_diabetes()

# data manipulation
X = diabetes.data
y = diabetes.target

# data set splitting
X_test, X_train, y_test, y_train = train_test_split(X, y, test_size = 0.2, random_state = 42)

# model fitting
multi_linear_reg_model = LinearRegression()
multi_linear_reg_model.fit(X_train, y_train)

# y_pred
y_pred = multi_linear_reg_model.predict(X_test)

# rmse and r^2
rmse = mean_squared_error(y_test, y_pred, squared = False)
print("rmse: ", rmse)
r2 = r2_score(y_test, y_pred)
print("r^2: ", r2)