import matplotlib.pyplot as plt # data visualization
from sklearn.datasets import fetch_olivetti_faces # data set
from sklearn.decomposition import PCA # pca for dimensionality reduction
from sklearn.model_selection import train_test_split, GridSearchCV # data set splitting, model tuning
from sklearn.ensemble import RandomForestClassifier # model fitting
from sklearn.metrics import accuracy_score, confusion_matrix # accuracy

# data set
faces = fetch_olivetti_faces()
X = faces.data         # (400, 4096)
y = faces.target       # (400, )
images = faces.images  # (400, 64, 64)

# images in the data set
fig, axes = plt.subplots(2, 5, figsize = (10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i], cmap = 'gray')
    ax.set_title(f"Label: {y[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()

# pca for dimensionality reduction
pca = PCA(n_components = 100, whiten = True, random_state = 42)
X_pca = pca.fit_transform(X)

X_inverse = pca.inverse_transform(X_pca)
images_pca = X_inverse.reshape(-1, 64, 64)

# new faces
fig, axes = plt.subplots(2, 5, figsize = (10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(images_pca[i], cmap = 'gray')
    ax.set_title(f"Label: {y[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()

# data set splitting
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size = 0.2, random_state = 42)

# model fitting
rf = RandomForestClassifier(random_state = 42)
rf.fit(X_train, y_train)

# y_pred
y_pred = rf.predict(X_test)

# accuracy
accuracy = accuracy_score(y_test, y_pred)

# model tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
    }

grid_search = GridSearchCV(
    estimator = RandomForestClassifier(random_state = 42),
    param_grid = param_grid,
    cv = 3,
    n_jobs = -1,
    verbose = 1
    )

grid_search.fit(X_train, y_train)

# best parameters
grid_search.best_params_

# final model with the best parameters
best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)
accuracy_new = accuracy_score(y_test, y_pred_best)
conf_matrix = confusion_matrix(y_test, y_pred_best)