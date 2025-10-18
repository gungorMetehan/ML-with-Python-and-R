# data set
library(lars)
data(diabetes)                # from lars package
X_full <- diabetes$x          # 10 standardized predictors
y <- diabetes$y               # target (quantitative)

# select a single feature (for simple linear regression)
X <- X_full[, 3, drop = FALSE]
colnames(X) <- "feature"

# shuffling & splitting (80/20)
set.seed(42)   # ensures reproducibility of the random split

n_samples <- nrow(X)
train_size <- floor(0.8 * n_samples)
indices <- sample(n_samples)
train_idx <- indices[1:train_size]
test_idx <- indices[(train_size + 1):n_samples]

X_train <- X[train_idx, , drop = FALSE]
y_train <- y[train_idx]
X_test <- X[test_idx, , drop = FALSE]
y_test <- y[test_idx]

# model fitting
df_train <- data.frame(y = y_train, feature = X_train[, 1])
lm_model <- lm(y ~ feature, data = df_train)

# prediction
y_pred <- predict(lm_model, newdata = data.frame(feature = X_test[, 1]))

# metrics: MSE and R^2 on test set
mse <- mean((y_test - y_pred)^2)
ss_res <- sum((y_test - y_pred)^2)
ss_tot <- sum((y_test - mean(y_test))^2)
r2 <- 1 - ss_res / ss_tot

cat("mse: ", mse, "\n")
cat("r^2: ", r2,  "\n")

# baseR: simple data visualization
plot(X_test[, 1], y_test,
     col = "gray", pch = 16,
     xlab = "Feature (Column 3)", ylab = "Target",
     main = "Simple Linear Regression (R)")
abline(lm_model, col = "red", lwd = 2)
