import pandas as pd # data manipulation
import numpy as np
from sklearn.linear_model import Lasso # model fitting
from sklearn.model_selection import train_test_split, GridSearchCV # data set splitting, model tuning
from sklearn.preprocessing import StandardScaler # standardization
from sklearn.metrics import mean_squared_error, r2_score # mse, rmse, r^2
import matplotlib.pyplot as plt # data visualization

# data set
url = 'https://raw.githubusercontent.com/gungorMetehan/ML-with-Python-and-R/refs/heads/main/data-sets/Hitters.csv'
Hitters = pd.read_csv(url, index_col = 0)

# data manipulation
Hitters = Hitters.dropna()
dms = pd.get_dummies(Hitters[["League", "Division", "NewLeague"]])
y = Hitters["Salary"]
X_= Hitters.drop(["Salary", "League", "Division", "NewLeague"], axis = 1).astype("float64")
X = pd.concat([X_, dms[["League_N", "Division_W", "NewLeague_N"]]], axis = 1)

# data set splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# model fitting
lasso_model = Lasso().fit(X_train, y_train)

# lambdas (alphas) data visualization
lambdas = np.logspace(-4, 3, 100)
lasso_model = Lasso()
coefs = []

for i in lambdas:
    lasso_model.set_params(alpha = i)
    lasso_model.fit(X_train, y_train)
    coefs.append(lasso_model.coef_)

ax = plt.gca()
ax.plot(lambdas, coefs)
ax.set_xscale("log")
plt.show()

# model tuning
lasso = Lasso(max_iter = 10000)
parameters = {"alpha": lambdas}
lasso_cv = GridSearchCV(lasso, parameters, scoring = "neg_mean_squared_error", cv = 10)
lasso_cv.fit(X_train, y_train)

# mse
print("Best alpha:", lasso_cv.best_params_["alpha"])
print("Best score (MSE):", -lasso_cv.best_score_)

# final model
best_alpha = lasso_cv.best_params_["alpha"]
lasso_final = Lasso(alpha = best_alpha)
lasso_final.fit(X_train, y_train)

# y_pred2
y_pred2 = lasso_final.predict(X_test)

# rmse and r^2
mse = mean_squared_error(y_test, y_pred2)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred2)
print("Test MSE:", mse)
print("Test RMSE:", rmse)
print("Test R^2:", r2)

# non-zero coefficients
non_zero_coefs = coefs[coefs != 0]
print(f"Number of selected features: {len(non_zero_coefs)} out of {X.shape[1]}")

# final model's coefs
coefs = pd.Series(lasso_final.coef_, index = X.columns)
coefs_sorted = coefs.sort_values()

plt.figure(figsize = (12, 6))
coefs_sorted.plot(kind = "barh", color = "skyblue")
plt.title("Lasso Regression Coefficients (Best Alpha)")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.show()