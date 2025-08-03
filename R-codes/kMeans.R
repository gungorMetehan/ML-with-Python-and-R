library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra) # data visualization

# data set
USArrests

# scaling variables
USArrests <- scale(USArrests)

# correlation between variables
distances <- get_dist(USArrests)
fviz_dist(distances, gradient = list(low = "#97bcda", mid = "white", high = "#de6069"))

# model fitting
kMeans <- kmeans(USArrests, centers = 2, nstart = 25)
kMeans

# data visualization for 2 clusters
fviz_cluster(kMeans, data = USArrests) +
  theme_minimal() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  labs(title = "k-Means Clustering with 2 Clusters")

# trying different number of clusters
kMeans3 <- kmeans(USArrests, centers = 3, nstart = 25)
kMeans4 <- kmeans(USArrests, centers = 4, nstart = 25)
kMeans5 <- kmeans(USArrests, centers = 5, nstart = 25)

# plotting to compare
p1 <- fviz_cluster(kMeans, geom = "point", data = USArrests) + 
  theme_minimal() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  ggtitle("k-Means Clustering with 2 Clusters")
p2 <- fviz_cluster(kMeans3, geom = "point", data = USArrests) + 
  theme_minimal() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  ggtitle("k-Means Clustering with 3 Clusters")
p3 <- fviz_cluster(kMeans4, geom = "point", data = USArrests) + 
  theme_minimal() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  ggtitle("k-Means Clustering with 4 Clusters")
p4 <- fviz_cluster(kMeans5, geom = "point", data = USArrests) + 
  theme_minimal() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  ggtitle("k-Means Clustering with 5 Clusters")

library(gridExtra)
grid.arrange(p1, p2, p3, p4, nrow = 2)

# determining the optimal number of clusters
## elbow method (there are other methods as well)
set.seed(12)
fviz_nbclust(USArrests, kmeans, method = "wss")
## silhouette method (there are other methods as well)
fviz_nbclust(USArrests, kmeans, method = "silhouette")
## gap statistic method (there are other methods as well)
set.seed(12)
gap_stats <- clusGap(USArrests, FUN = kmeans, nstart = 25, K.max = 10, B = 50)
fviz_gap_stat(gap_stats)

# final
## computing k-means clustering with k = 4
set.seed(12)
final_model <- kmeans(USArrests, 4, nstart = 25)

## data visualization
fviz_cluster(final_model, data = USArrests) +
  theme_minimal() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  ggtitle("k-Means Clustering with 4 Clusters [Final Model]")

# prediction
new_obs <- data.frame(Murder = 7.8, Assault = 210, UrbanPop = 65, Rape = 21.5)
center <- attr(USArrests, "scaled:center")
scale_ <- attr(USArrests, "scaled:scale")
new_obs_scaled <- scale(new_obs, center = center, scale = scale_)

## centers
centers <- final_model$centers

## euclid distances
distances <- apply(centers, 1, function(center) {
  sqrt(sum((new_obs_scaled - center)^2))
})

## minimum distance
assigned_cluster <- which.min(distances)
assigned_cluster # the new observation is assigned to cluster number 2