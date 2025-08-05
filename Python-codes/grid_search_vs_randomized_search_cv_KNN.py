from sklearn.datasets import load_iris # data set
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV # data set splitting, cross-validation
from sklearn.neighbors import KNeighborsClassifier # model fitting
from sklearn.metrics import accuracy_score # accuracy score
import numpy as np

# data set
iris = load_iris()
X = iris.data
y = iris.target

# data set splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# model fitting
knn = KNeighborsClassifier()

# hyperparameters for model tuning
knn_param_grid = {"n_neighbors": np.arange(2, 31)}

# grid search
knn_grid_search = GridSearchCV(knn, knn_param_grid, cv = 5)
knn_grid_search.fit(X_train, y_train)
y_pred_grid = knn_grid_search.predict(X_test)

print("KNN Grid Search Best Parameters: ", knn_grid_search.best_params_)
print("KNN Grid Search Best Accuracy: ", knn_grid_search.best_score_)
print("Grid Search Test Accuracy: ", accuracy_score(y_test, y_pred_grid))

# randomized search
knn_randomized_search = RandomizedSearchCV(knn, knn_param_grid, cv = 5, n_iter = 15, random_state = 42)
knn_randomized_search.fit(X_train, y_train)
y_pred_random = knn_randomized_search.predict(X_test)

print("KNN Randomized Search Best Parameters: ", knn_randomized_search.best_params_)
print("KNN Randomized Search Best Accuracy: ", knn_randomized_search.best_score_)
print("Randomized Search Test Accuracy: ", accuracy_score(y_test, y_pred_random))