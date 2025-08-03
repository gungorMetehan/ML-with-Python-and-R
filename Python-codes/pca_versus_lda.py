from sklearn.datasets import load_iris # data set
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # model fitting - LDA
from sklearn.decomposition import PCA # model fitting - PCA
import matplotlib.pyplot as plt # data visualization

# data set
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# model fitting
pca = PCA(n_components = 2)
X_pca = pca.fit_transform(X)
lda = LinearDiscriminantAnalysis(n_components = 2)
X_lda = lda.fit_transform(X, y)

# data visualization
colors = ["#58508d", "#ff6361", "#ffa600"]

## PCA
plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color = color, alpha = 0.9, label = target_name)
plt.legend()
plt.title("PCA of Iris Data Set")

## LDA
plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], color = color, alpha = 0.9, label = target_name)
plt.legend()
plt.title("LDA of Iris Data Set")