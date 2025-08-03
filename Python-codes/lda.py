from sklearn.datasets import fetch_openml # data set
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # model fitting
import matplotlib.pyplot as plt # data visualization

# data set
mnist = fetch_openml("mnist_784", version = 1) 
X = mnist.data
y = mnist.target.astype(int)

# model fitting
lda = LinearDiscriminantAnalysis(n_components = 2)
X_lda = lda.fit_transform(X, y)

# data visualization
plt.figure(figsize = (10, 8))
plt.scatter(X_lda[:, 0], X_lda[:, 1], c = y, cmap = "tab10", alpha = 0.5)
plt.title("LDA of MNIST Data Set")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.colorbar(label = "Digits")