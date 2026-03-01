library(tidyverse) # load data manipulation packages
library(caret)     # machine learning utilities
library(pROC)      # ROC and AUC functions
library(nnet)      # multinomial logistic regression

# importing data
df[df == "?"] <- NA # mark missing values
df <- na.omit(df) # remove rows with NA

df <- df %>%
  mutate(across(everything(), as.numeric)) # convert columns numeric

# train - test split
set.seed(12)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
X_test  <- X[-train_index, ]
y_train <- y[train_index]
y_test  <- y[-train_index]

# fitting scaler
scaler <- preProcess(X_train, method = c("center", "scale"))

# scale training and test data
X_train_scaled <- predict(scaler, X_train) 
X_test_scaled  <- predict(scaler, X_test)

# fitting multinomial regression model
model <- multinom(y_train ~ ., data = X_train_scaled)

# predicted classes
y_pred <- predict(model, X_test_scaled)

# predicted probabilities
y_prob <- predict(model, X_test_scaled, type = "probs") 

# confusion matrix
conf_matrix <- confusionMatrix(y_pred, y_test)

# accuracy
accuracy <- conf_matrix$overall["Accuracy"]

# binarize targets for ROC
y_test_bin <- model.matrix(~ y_test - 1)

roc_list <- list()
auc_list <- numeric(ncol(y_test_bin))

for (i in 1:ncol(y_test_bin)) {
  roc_list[[i]] <- roc(
    y_test_bin[, i],
    y_prob[, i]
  )
  auc_list[i] <- auc(roc_list[[i]])
}

# plot ROC curves
plot(
  roc_list[[1]],
  col = 1,
  lwd = 2,
  main = "Multi-Class ROC Curve (One-vs-Rest)"
)

for (i in 2:length(roc_list)) {
  plot(roc_list[[i]], col = i, lwd = 2, add = TRUE)
}

legend(
  "bottomright",
  legend = paste("Class", 0:(length(roc_list) - 1),
                 "AUC =", round(auc_list, 2)),
  col = 1:length(roc_list),
  lwd = 2
)