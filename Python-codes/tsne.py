from sklearn.datasets import fetch_openml # data set
from sklearn.manifold import TSNE # model fitting
import matplotlib.pyplot as plt # data visualization

# data set
mnist = fetch_openml("mnist_784", version = 1) 
X = mnist.data
y = mnist.target.astype(int)

# model fitting
tsne = TSNE(n_components = 2, perplexity = 30, n_iter = 1000, random_state = 42)
X_tsne = tsne.fit_transform(X)

# data visualization
plt.figure(figsize = (8, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c = y, cmap = "tab10", alpha = 0.75)
plt.colorbar(scatter, label = "Digits")
plt.title("t-SNE of MNIST Data Set")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()