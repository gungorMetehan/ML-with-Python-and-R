from sklearn import datasets, cluster # 4 different data sets & model fitting
from sklearn.preprocessing import StandardScaler # scaling
import numpy as np
import matplotlib.pyplot as plt # data visualization

# 4 different data sets
circle_data = datasets.make_circles(n_samples = 1500, noise = 0.06)
moon_data = datasets.make_moons(n_samples = 1500, noise = 0.05)
blobs_data = datasets.make_blobs(n_samples = 1500)
no_structure = np.random.rand(1500, 2), None

# clustering algorithm names for data visualization
clustering_names = [
    "MiniBatchKMeans",
    "SpectralClustering",
    "Ward",
    "AgglomerativeClustering",
    "DBSCAN",
    "Birch"]

# colors for data visualization
colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

# data sets list for data visualization
datasets_list = [circle_data, moon_data, blobs_data, no_structure]

# data visualization
plt.figure(figsize = (15, 10))
i = 1

for i_dataset, dataset in enumerate(datasets_list):
    X, y = dataset
    X = StandardScaler().fit_transform(X)

    clustering_algos = [
        cluster.MiniBatchKMeans(n_clusters = 2),
        cluster.SpectralClustering(n_clusters = 2),
        cluster.AgglomerativeClustering(n_clusters = 2, linkage = "ward"),
        cluster.AgglomerativeClustering(n_clusters = 2, linkage = "average"),
        cluster.DBSCAN(eps = 0.2),
        cluster.Birch(n_clusters = 2)
    ]

    for name, algo in zip(clustering_names, clustering_algos):
        algo.fit(X)

        if hasattr(algo, "labels_"):
            y_pred = algo.labels_.astype(int)
        else:
            y_pred = algo.predict(X)

        point_colors = np.array(colors)[y_pred]
        plt.subplot(len(datasets_list), len(clustering_algos), i)
        if i_dataset == 0:
            plt.title(name)
        plt.scatter(X[:, 0], X[:, 1], color = point_colors, s = 10)
        plt.xticks([])
        plt.yticks([])
        i += 1

plt.tight_layout()
plt.show()