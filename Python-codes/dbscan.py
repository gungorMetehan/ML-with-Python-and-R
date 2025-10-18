from sklearn.datasets import make_circles   # for synthetic circular data generation
from sklearn.cluster import DBSCAN          # for density-based clustering
import matplotlib.pyplot as plt             # for visualization

# Generating a non-linearly separable dataset (two concentric circles)
X, _ = make_circles(n_samples = 1200, factor = 0.5, noise = 0.075, random_state = 42)

# Visualizing the original data without clustering
plt.figure(figsize = (6, 6))
plt.scatter(X[:, 0], X[:, 1], s = 10, color = 'gray')
plt.title("Original Data (Unclustered)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Creating and fitting the DBSCAN model
# eps: The maximum distance between two samples to be considered as neighbors
# min_samples: The minimum number of points required to form a dense region (cluster)
dbscan = DBSCAN(eps = 0.1, min_samples = 5)
cluster_labels = dbscan.fit_predict(X)

# data visualization
plt.figure(figsize = (6, 6))
plt.scatter(X[:, 0], X[:, 1], c = cluster_labels, cmap = "viridis", s = 10)
plt.title("DBSCAN Clustering Result")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
