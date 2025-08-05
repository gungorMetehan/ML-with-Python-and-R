from sklearn.datasets import load_iris # data set
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV # data set splitting, cross-validation
from sklearn.svm import SVC # model fitting
from sklearn.metrics import accuracy_score # accuracy score

# data set
iris = load_iris()
X = iris.data
y = iris.target

# data set splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# model fitting
svm = SVC()

# hyperparameters for model tuning
svm_param_grid = {"C": [0.1, 1, 10, 100],
                 "gamma": [0.1, 0.01, 0.001, 0.0001]}

# grid search
svm_grid_search = GridSearchCV(svm, svm_param_grid, cv = 5, scoring = 'accuracy')
svm_grid_search.fit(X_train, y_train)
y_pred_grid = svm_grid_search.predict(X_test)

print("SVM Grid Search Best Parameters: ", svm_grid_search.best_params_)
print("SVM Grid Search Best Accuracy: ", svm_grid_search.best_score_)
print("Grid Search Test Accuracy: ", accuracy_score(y_test, y_pred_grid))

# randomized search
svm_randomized_search = RandomizedSearchCV(svm, svm_param_grid, cv = 5, n_iter = 15, random_state = 42)
svm_randomized_search.fit(X_train, y_train)
y_pred_random = svm_randomized_search.predict(X_test)

print("SVM Randomized Search Best Parameters: ", svm_randomized_search.best_params_)
print("SVM Randomized Search Best Accuracy: ", svm_randomized_search.best_score_)
print("Randomized Search Test Accuracy: ", accuracy_score(y_test, y_pred_random))