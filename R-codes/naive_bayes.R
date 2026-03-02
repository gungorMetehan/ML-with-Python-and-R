library(caret)      # train/test split, confusionMatrix
library(e1071)      # Naive Bayes
library(ggplot2)    # visualization
library(reshape2)   # confusion matrix reshaping

# data manipulation
X <- iris[, 1:4]      # features
y <- iris$Species     # target

set.seed(12)

# data set splitting (80% train, 20% test)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)

X_train <- X[train_index, ]
X_test  <- X[-train_index, ]

y_train <- y[train_index]
y_test  <- y[-train_index]

# model fitting (gaussian naive bayes)
naive_b <- naiveBayes(x = X_train, y = y_train)

# prediction
y_pred <- predict(naive_b, X_test)

# confusion matrix
confusionMatrix(y_pred, y_test)

# confusion matrix (heatmap)
conf_mat <- table(Actual = y_test, Predicted = y_pred)
conf_df <- melt(conf_mat)

ggplot(conf_df, aes(x = Predicted, y = Actual, fill = value)) +
  geom_tile() +
  geom_text(aes(label = value), color = "white", size = 5) +
  scale_fill_gradient(low = "#ffd9a1", high = "#d47f00") +
  labs(title = "Confusion Matrix",
       x = "Predicted",
       y = "Actual") +
  theme_minimal()
