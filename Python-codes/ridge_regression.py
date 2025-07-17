import pandas as pd # data manipulation
import numpy as np
from sklearn.linear_model import Ridge # model fitting
from sklearn.model_selection import train_test_split, GridSearchCV # data set splitting, model tuning
from sklearn.preprocessing import StandardScaler # standardization
from sklearn.metrics import mean_squared_error, r2_score # rmse, r^2
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
ridge_model = Ridge().fit(X_train, y_train)

# lambdas (alphas) data visualization
lambdas = np.logspace(-4, 3, 100)

ridge_model = Ridge()
coefs = []

for i in lambdas:
    ridge_model.set_params(alpha = i)
    ridge_model.fit(X_train, y_train)
    coefs.append(ridge_model.coef_)

ax = plt.gca()
ax.plot(lambdas, coefs)
ax.set_xscale("log")
plt.show()

# model tuning
ridge = Ridge()
parameters = {"alpha": lambdas}
ridge_cv = GridSearchCV(ridge, parameters, scoring = "neg_mean_squared_error", cv = 10)
ridge_cv.fit(X_train, y_train)

print("Best alpha:", ridge_cv.best_params_["alpha"])
print("Best score (MSE):", -ridge_cv.best_score_)

# final model
best_alpha = ridge_cv.best_params_["alpha"]
ridge_final = Ridge(alpha = best_alpha)
ridge_final.fit(X_train, y_train)

# y_pred
y_pred = ridge_final.predict(X_test)

# rmse and r^2
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("Test MSE:", mse)
print("Test RMSE:", rmse)
print("Test R^2:", r2)

# final model's coefs
coefs = pd.Series(ridge_final.coef_, index = X.columns)
coefs_sorted = coefs.sort_values()

plt.figure(figsize = (12, 6))
coefs_sorted.plot(kind = "barh", color = "skyblue")
plt.title("Ridge Regression Coefficients (Best Alpha)")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.show()