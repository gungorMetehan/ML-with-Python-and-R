from sklearn.datasets import make_blobs # data generation
from sklearn.cluster import AgglomerativeClustering # model fitting
from sklearn.preprocessing import StandardScaler # standardization
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt # data visualization
import numpy as np

# data set
X, _ = make_blobs(n_samples = 500, centers = 4, cluster_std = 0.8, random_state = 42)

# NOTE: Hierarchical clustering is distance-based and scale-sensitive.
# If features are on different scales, scaling is recommended:
# X = StandardScaler().fit_transform(X)

# visualizing original data
plt.figure(figsize = (5, 5))
plt.scatter(X[:, 0], X[:, 1], s = 10, c = "gray")
plt.title("Original Data (Unclustered)")
plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
plt.tight_layout()
plt.show()

# linkage methods to compare
linkage_methods = ["ward", 
                   "single", 
                   "average", 
                   "complete"]

# data visualization
fig, axes = plt.subplots(2, len(linkage_methods), figsize = (16, 8))

# for readability/speed, subsample points for dendrograms
rng = np.random.RandomState(42)
sub_idx = rng.choice(len(X), size = 150, replace = False)
X_sub = X[sub_idx]

for j, method in enumerate(linkage_methods):
    # dendrogram (top row)
    Z = linkage(X_sub, method = method)
    ax_top = axes[0, j]
    dendrogram(Z, no_labels = True, color_threshold = None, ax = ax_top)
    ax_top.set_title(f"{method.capitalize()} Linkage\nDendrogram")
    ax_top.set_xlabel("Subsampled data points")
    ax_top.set_ylabel("Distance")
    # agglomerative clustering (bottom row)
    if method == "ward":
        model = AgglomerativeClustering(n_clusters = 4, linkage = method)
    else:
        model = AgglomerativeClustering(n_clusters = 4, linkage = method, metric = "euclidean")

    labels = model.fit_predict(X)
    ax_bottom = axes[1, j]
    sc = ax_bottom.scatter(X[:, 0], X[:, 1], c = labels, s = 10, cmap = "viridis")
    ax_bottom.set_title(f"{method.capitalize()} Linkage\nClustering (k=4)")
    ax_bottom.set_xlabel("X"); ax_bottom.set_ylabel("Y")

plt.tight_layout()
plt.show()
