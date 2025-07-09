from sklearn.datasets import load_iris # data set
from sklearn.model_selection import train_test_split # splitting data set
from sklearn.tree import DecisionTreeClassifier, plot_tree # decision tree algorithm
from sklearn.metrics import accuracy_score, confusion_matrix # accuracy
import matplotlib.pyplot as plt # data visualization
from sklearn.inspection import DecisionBoundaryDisplay # data visualization - boundaries
import numpy as np # data visualization


# loading data and data manipulation
iris = load_iris()
# defining features and target variable
X = iris.data
y = iris.target

# splitting data set (train - test, 80%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 12)

# model fitting
dtree_class = DecisionTreeClassifier(criterion = "gini", max_depth = 5, random_state = 12) # criterion = "entropy" is also O.K.
dtree_class.fit(X_train, y_train)

# y_pred
y_pred = dtree_class.predict(X_test)

# accuracy score
accuracy = accuracy_score(y_test, y_pred)
accuracy
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix

# tree
plt.figure(figsize = (15, 10))
plot_tree(dtree_class, filled = True, feature_names = iris.feature_names, class_names = list(iris.target_names))

# feature importances
feature_imp = dtree_class.feature_importances_
feature_names = iris.feature_names
feature_imp_sorted = sorted(zip(feature_imp, feature_names), reverse = True)

for importance, feature_name in feature_imp_sorted:
    print(f"{feature_name}: {importance}")

# visualization - boundaries
n_classes = len(iris.target_names)
plot_col = "ryb"

for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
    X = iris.data[:, pair]
    y = iris.target
    dtree = DecisionTreeClassifier().fit(X, y)
    
    ax = plt.subplot(2, 3, pairidx + 1)
    plt.tight_layout(h_pad = .5, w_pad = .5, pad = 2.5)
    DecisionBoundaryDisplay.from_estimator(dtree,
                                           X,
                                           cmap = plt.cm.RdYlBu,
                                           response_method = "predict",
                                           ax = ax,
                                           xlabel = iris.feature_names[pair[0]],
                                           ylabel = iris.feature_names[pair[1]])
    
    for i, color in zip(range(n_classes), plot_col):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c = color, label = iris.target_names[i],
                    edgecolors = "black")

# model tuning
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [1, 2, 3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
    }

# 5-fold CV with param_grid parameters
grid_search = GridSearchCV(estimator = DecisionTreeClassifier(), 
                           param_grid = param_grid, 
                           cv = 5,  # 5-fold cross-validation
                           scoring = "accuracy")

# fit the GridSearchCV model
dtree_class_grid = grid_search.fit(X_train, y_train)

# find the best parameters
dtree_class_grid.best_params_

# final model
final_model = DecisionTreeClassifier(criterion = "gini", max_depth = 3, min_samples_split = 2).fit(X_train, y_train)
# y_pred
y_pred = final_model.predict(X_test)

# accuracy score
accuracy = accuracy_score(y_test, y_pred)
accuracy
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix