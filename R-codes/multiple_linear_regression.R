# data set
library(lars)
data(diabetes)                # from lars package
X_full <- diabetes$x          # 10 standardized predictors (matrix)
y <- diabetes$y               # quantitative target (vector)

# convert predictors to a data.frame
X_df <- as.data.frame(X_full)
colnames(X_df) <- paste0("feature_", seq_len(ncol(X_df)))

# shuffling & splitting (80/20)
set.seed(42) # set a seed so that the random split is reproducible

n_samples <- nrow(X_df)
train_size <- floor(0.8 * n_samples)
indices <- sample(n_samples)
train_idx <- indices[1:train_size]
test_idx <- indices[(train_size + 1):n_samples]

df_train <- data.frame(y = y[train_idx], X_df[train_idx, , drop = FALSE])
df_test <- data.frame(y = y[test_idx], X_df[test_idx, , drop = FALSE])

# model fitting
# NOTE: Predictors in this dataset are already standardized (from 'lars' package).
lm_model <- lm(y ~ ., data = df_train)

# prediction
y_pred <- predict(lm_model, newdata = df_test)

# metrics: MSE and R^2 on test set
mse <- mean((df_test$y - y_pred)^2)
ss_res <- sum((df_test$y - y_pred)^2)
ss_tot <- sum((df_test$y - mean(df_test$y))^2)
r2 <- 1 - ss_res / ss_tot

cat("mse: ", mse, "\n")
cat("r^2: ", r2,  "\n")

# simple base R visualizations
# (a) Actual vs Predicted on the test set
plot(df_test$y, y_pred,
     col = "gray25", pch = 16,
     xlab = "Actual Target (test y)",
     ylab = "Predicted Target (??)",
     main = "Multiple Linear Regression: Test Actual vs Predicted")
abline(a = 0, b = 1, col = "red", lwd = 2)  # 45-degree reference line

# (b) Residuals vs Predicted (diagnostic scatter)
residuals_test <- df_test$y - y_pred
plot(y_pred, residuals_test,
     col = "gray25", pch = 16,
     xlab = "Predicted (??)",
     ylab = "Residuals (y - ??)",
     main = "Residuals vs Predicted (Test Set)")
abline(h = 0, col = "red", lwd = 2)