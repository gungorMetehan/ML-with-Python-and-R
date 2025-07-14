import pandas as pd # data manipulation
import numpy as np # sqrt
from sklearn.metrics import mean_squared_error # rmse
from sklearn.model_selection import train_test_split, GridSearchCV # data set splitting, model tuning
from sklearn.preprocessing import StandardScaler # standardization
from sklearn.neural_network import MLPRegressor # model fitting

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

# scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# model fitting
mlp_model = MLPRegressor(random_state = 42).fit(X_train_scaled, y_train)
mlp_model

# y_pred
y_pred = mlp_model.predict(X_test_scaled)

# rmse
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmse

# model tuning
mlp_params = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1],
    "hidden_layer_sizes": [(10,), (50,), (100,), (100,50), (50,100,50)],
    "activation": ["relu", "tanh"],
    "solver": ["adam", "lbfgs"]
    }

mlp_cv_model = GridSearchCV(mlp_model, mlp_params, cv = 10, verbose = 2, n_jobs = -1).fit(X_train_scaled, y_train)
mlp_cv_model.best_params_

mlp_tuned = MLPRegressor(alpha = 0.0001, hidden_layer_sizes = (100, ), solver = "lbfgs", activation = "tanh", random_state = 42).fit(X_train_scaled, y_train)

# y_pred2
y_pred2 = mlp_tuned.predict(X_test_scaled)

# rmse2
rmse2 = np.sqrt(mean_squared_error(y_test, y_pred2))
rmse2
