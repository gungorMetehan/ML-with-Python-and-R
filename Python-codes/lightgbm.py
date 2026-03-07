pip install lightgbm

import pandas as pd # data manipulation
from sklearn.metrics import accuracy_score, confusion_matrix # evaluation metrics
from sklearn.model_selection import train_test_split, GridSearchCV # split and tuning
from lightgbm import LGBMClassifier # LightGBM model

# data set
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
           "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]

df = pd.read_csv(url, names = columns)

# data preparation
y = df["Outcome"] # target variable
X = df.drop("Outcome", axis = 1) # predictor variables

# train - test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# model fitting
lgbm_model = LGBMClassifier(random_state = 42).fit(X_train, y_train)

# prediction
y_pred = lgbm_model.predict(X_test)

# accuracy
accuracy = accuracy_score(y_test, y_pred)
accuracy

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix

# hyperparameter grid
lgbm_params = {
    "learning_rate": [0.01, 0.05, 0.1], # boosting learning rate
    "n_estimators": [100, 200, 500], # number of trees
    "max_depth": [-1, 3, 5], # tree depth limit
    "subsample": [0.8, 1.0], # row sampling ratio
    "colsample_bytree": [0.8, 1.0] # column sampling ratio
}

# grid search cross validation
lgbm_cv_model = GridSearchCV(
    lgbm_model,
    lgbm_params,
    cv = 10,
    verbose = 2,
    n_jobs = -1
).fit(X_train, y_train)

# best parameters
lgbm_cv_model.best_params_

# tuned model fitting
lgbm_tuned = LGBMClassifier(
    learning_rate = lgbm_cv_model.best_params_["learning_rate"],
    n_estimators = lgbm_cv_model.best_params_["n_estimators"],
    max_depth = lgbm_cv_model.best_params_["max_depth"],
    subsample = lgbm_cv_model.best_params_["subsample"],
    colsample_bytree = lgbm_cv_model.best_params_["colsample_bytree"],
    random_state = 42
).fit(X_train, y_train)

# tuned predictions
y_pred2 = lgbm_tuned.predict(X_test)

# tuned accuracy
accuracy2 = accuracy_score(y_test, y_pred2)
accuracy2

# feature importance plot
import matplotlib.pyplot as plt # data visualization

# feature importance values
importance = lgbm_tuned.feature_importances_

feature_imp = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values("Importance", ascending = False)

# horizontal bar plot
plt.figure(figsize = (8, 6))
plt.barh(feature_imp["Feature"], feature_imp["Importance"])
plt.xlabel("Importance Score")
plt.title("LightGBM Feature Importance")
plt.gca().invert_yaxis()
plt.show()