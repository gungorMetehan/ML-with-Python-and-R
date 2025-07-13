from sklearn.datasets import fetch_california_housing # data set
from sklearn.model_selection import train_test_split, GridSearchCV # data set splitting, model tuning
from sklearn.ensemble import RandomForestRegressor # model fitting
from sklearn.metrics import mean_squared_error # rmse
import numpy as np

# data set
california_housing = fetch_california_housing()
X = california_housing.data         # (20640, 8)
y = california_housing.target       # (20640, )


# data set splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# model fitting
rf_reg = RandomForestRegressor(random_state = 42)
rf_reg.fit(X_train, y_train)

# y_pred
y_pred = rf_reg.predict(X_test)

# rmse
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(rmse)

# model tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5]
    }

grid_search = GridSearchCV(rf_reg, param_grid, cv = 5, scoring = 'neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# best parameters
grid_search.best_params_

# final model with the best parameters
## model fitting
rf_reg2 = RandomForestRegressor(max_depth = None, n_estimators = 300, random_state = 42)
rf_reg2.fit(X_train, y_train)

## y_pred
y_pred2 = rf_reg2.predict(X_test)

## rmse
mse2 = mean_squared_error(y_test, y_pred2)
rmse2 = np.sqrt(mse2)
print(rmse2)