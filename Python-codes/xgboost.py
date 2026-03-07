import pandas as pd # data manipulation
from sklearn.metrics import accuracy_score, confusion_matrix # evaluation metrics
from sklearn.model_selection import train_test_split, GridSearchCV # split and tuning
from xgboost import XGBClassifier # XGBoost model

# data set
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
           "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]

df = pd.read_csv(url, names = columns) # load dataset

# data preparation
y = df["Outcome"] # target variable
X = df.drop("Outcome", axis = 1) # predictors

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# model fitting
xgb_model = XGBClassifier(
    use_label_encoder = False,
    eval_metric = "logloss",
    random_state = 42
).fit(X_train, y_train)

# prediction step
y_pred = xgb_model.predict(X_test)

# accuracy score
accuracy = accuracy_score(y_test, y_pred)
accuracy

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix

# hyperparameter grid
xgb_params = {
    "learning_rate": [0.01, 0.05, 0.1], # boosting rate
    "n_estimators": [100, 200, 500], # tree number
    "max_depth": [3, 4, 6], # tree depth
    "subsample": [0.8, 1.0], # row sampling
    "colsample_bytree": [0.8, 1.0] # column sampling
}

# grid search cv
xgb_cv_model = GridSearchCV(
    xgb_model,
    xgb_params,
    cv = 10,
    verbose = 2,
    n_jobs = -1
).fit(X_train, y_train)

# best parameters
xgb_cv_model.best_params_

# tuned model fitting
xgb_tuned = XGBClassifier(
    learning_rate = xgb_cv_model.best_params_["learning_rate"],
    n_estimators = xgb_cv_model.best_params_["n_estimators"],
    max_depth = xgb_cv_model.best_params_["max_depth"],
    subsample = xgb_cv_model.best_params_["subsample"],
    colsample_bytree = xgb_cv_model.best_params_["colsample_bytree"],
    use_label_encoder = False,
    eval_metric = "logloss",
    random_state = 42
).fit(X_train, y_train)

# tuned predictions
y_pred2 = xgb_tuned.predict(X_test)

# tuned accuracy
accuracy2 = accuracy_score(y_test, y_pred2)
accuracy2

# feature importance plot
import matplotlib.pyplot as plt # visualization

# importance values
importance = xgb_tuned.feature_importances_

# dataframe creation
feature_imp = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values("Importance", ascending = False)

# bar plot
plt.figure(figsize = (8, 6))
plt.barh(feature_imp["Feature"], feature_imp["Importance"])
plt.xlabel("Importance Score")
plt.title("XGBoost Feature Importance")
plt.gca().invert_yaxis()
plt.show()
