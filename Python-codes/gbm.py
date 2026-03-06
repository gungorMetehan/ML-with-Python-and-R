import pandas as pd # data manipulation
from sklearn.metrics import accuracy_score, confusion_matrix # model evaluation metrics
from sklearn.model_selection import train_test_split, GridSearchCV # split and tuning
from sklearn.preprocessing import StandardScaler # feature scaling
from sklearn.ensemble import GradientBoostingClassifier # GBM model

# data set
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
           "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]

df = pd.read_csv(url, names = columns) # load dataset

# data manipulation
y = df["Outcome"] # target variable
X = df.drop("Outcome", axis = 1) # predictor variables

# data set splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train) # scale training data
X_test_scaled = scaler.transform(X_test) # scale test data

# model fitting
gbm_model = GradientBoostingClassifier(random_state = 42).fit(X_train_scaled, y_train)
gbm_model

# prediction
y_pred = gbm_model.predict(X_test_scaled)

# accuracy
accuracy = accuracy_score(y_test, y_pred)
accuracy

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix

# model tuning parameters
gbm_params = {
    "learning_rate": [0.01, 0.05, 0.1], # boosting step size
    "n_estimators": [100, 200, 500], # number of trees
    "max_depth": [2, 3, 4], # tree depth control
    "subsample": [0.8, 1.0] # stochastic boosting
}

# grid search cross validation
gbm_cv_model = GridSearchCV(
    gbm_model,
    gbm_params,
    cv = 10,
    verbose = 2,
    n_jobs = -1
).fit(X_train_scaled, y_train)

# best parameters
gbm_cv_model.best_params_

# tuned model fitting
gbm_tuned = GradientBoostingClassifier(
    learning_rate = gbm_cv_model.best_params_["learning_rate"],
    n_estimators = gbm_cv_model.best_params_["n_estimators"],
    max_depth = gbm_cv_model.best_params_["max_depth"],
    subsample = gbm_cv_model.best_params_["subsample"],
    random_state = 42
).fit(X_train_scaled, y_train)

# tuned predictions
y_pred2 = gbm_tuned.predict(X_test_scaled) # tuned model prediction

# tuned accuracy
accuracy2 = accuracy_score(y_test, y_pred2) # tuned model accuracy
accuracy2

# note. hyperparameter tuning does not guarantee better test performance.

# feature importance - data visualization
import matplotlib.pyplot as plt # visualization

# feature importance values
importance = gbm_tuned.feature_importances_

# convert to dataframe
feature_imp = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values("Importance", ascending = False)

# plotting
plt.figure(figsize = (8, 6))
plt.barh(feature_imp["Feature"], feature_imp["Importance"])
plt.xlabel("Importance Score")
plt.title("GBM Feature Importance")
plt.gca().invert_yaxis()
plt.show()