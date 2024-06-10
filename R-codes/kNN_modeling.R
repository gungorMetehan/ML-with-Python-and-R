### K-Nearest Neighbours in R ###

# examining the dataset

data(iris)
str(iris)
summary(iris)

# making this example reproducible
set.seed(12)

# shuffling rows
random <- runif(nrow(iris))
iris_random <- iris[order(random), ]
head(iris_random)

# an user-defined normalization function
normal <- function(x) {
  return(((x - min(x)) / (max(x) - min(x))))
}

# new shuffled-dataset
iris_new <- as.data.frame(lapply(iris_random[, -5], normal))
summary(iris_new)

# graph (correlation)
install.packages("GGally")
library(GGally)
ggpairs(iris_new)

# using 80% of dataset as training set and 20% as test set (120+30)
train_iris <- iris_new[1:120, ]
test_iris <- iris_new[121:150, ]
train_label <- iris_random[1:120, 5]
test_label <- iris_random[121:150, 5]

# kNN modeling
install.packages("class")
library(class)
knn_pred <- knn(train = train_iris, test = test_iris, cl = train_label, k = 3)

# output
table(test_label, knn_pred)

# another table (extra / optional)
install.packages("gmodels")
library(gmodels)
CrossTable(x = test_label, y = knn_pred, prob.chisq = F)

# calculating accuracy
accuracy <- sum(knn_pred == test_label) / length(test_label)
accuracy

## Optimizing k ##
k_values <- 1:15

# calculating accuracy for each k value
accuracy_values <- sapply(k_values, function(k) {
  classifier_knn <- knn(train = train_iris, 
                        test = test_iris, 
                        cl = train_label, 
                        k = k)
  1 - mean(classifier_knn != test_label)
})

accuracy_data <- data.frame(K = k_values, Accuracy = accuracy_values)

# visualization
library(ggplot2)
ggplot(accuracy_data, aes(x = K, y = Accuracy)) +
  geom_line(color = "#CCEAFF", size = .75) +
  geom_point(color = "#418BBF", size = 2.5) +
  labs(title = "Model Accuracy for Different k Values",
       x = "Number of Neighbors (K)",
       y = "Accuracy") +
  theme_classic()

