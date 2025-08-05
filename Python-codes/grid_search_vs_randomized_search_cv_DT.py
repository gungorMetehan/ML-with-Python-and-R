from sklearn.datasets import load_iris # data set
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV # data set splitting, cross-validation
from sklearn.tree import DecisionTreeClassifier # model fitting
from sklearn.metrics import accuracy_score # accuracy score

# data set
iris = load_iris()
X = iris.data
y = iris.target

# data set splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# model fitting
DT = DecisionTreeClassifier()

# hyperparameters for model tuning
DT_param_grid = {"max_depth": [3, 5, 7],
                 "max_leaf_nodes": [None, 5, 10, 15, 20],
                 "min_samples_split": [2, 4, 6]}

# grid search
DT_grid_search = GridSearchCV(DT, DT_param_grid, cv = 5, scoring = 'accuracy')
DT_grid_search.fit(X_train, y_train)
y_pred_grid = DT_grid_search.predict(X_test)

print("DT Grid Search Best Parameters: ", DT_grid_search.best_params_)
print("DT Grid Search Best Accuracy: ", DT_grid_search.best_score_)
print("Grid Search Test Accuracy: ", accuracy_score(y_test, y_pred_grid))

# randomized search
DT_randomized_search = RandomizedSearchCV(DT, DT_param_grid, cv = 5, n_iter = 15, random_state = 42)
DT_randomized_search.fit(X_train, y_train)
y_pred_random = DT_randomized_search.predict(X_test)

print("DT Randomized Search Best Parameters: ", DT_randomized_search.best_params_)
print("DT Randomized Search Best Accuracy: ", DT_randomized_search.best_score_)
print("Randomized Search Test Accuracy: ", accuracy_score(y_test, y_pred_random))