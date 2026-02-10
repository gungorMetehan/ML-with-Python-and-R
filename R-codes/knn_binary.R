# load required dataset
install.packages("mlbench")
library(mlbench)

# load breast_cancer data
data(BreastCancer)
bc <- BreastCancer

str(bc) # inspect data structure
summary(bc) # summary statistics

# remove ID column
if ("Id" %in% names(bc)) bc$Id <- NULL

# handle missing values
bc[bc == "?"] <- NA
bc <- na.omit(bc)

# convert factors to numeric (caution)
bc_num <- as.data.frame(lapply(bc[, -10], function(x) as.numeric(as.character(x))))

# check numeric structure
str(bc_num)

# define normalization function
normal <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

# normalize feature variables
bc_norm <- as.data.frame(lapply(bc_num, normal))

summary(bc_norm) # normalized summary

# define target variable
bc_class <- factor(bc$Class)  # benign / malignant
table(bc_class)

# set random seed
set.seed(12)

# generate train indices
index <- sample(1:nrow(bc_norm), size = 0.8 * nrow(bc_norm))

# split feature data
train_bc <- bc_norm[index, ]
test_bc  <- bc_norm[-index, ]

# split class labels
train_label <- bc_class[index]
test_label  <- bc_class[-index]

# load kNN package
install.packages("class")
library(class)

# fit kNN model
knn_pred <- knn(train = train_bc, test = test_bc, cl = train_label, k = 5))

# confusion matrix
table(test_label, knn_pred)

# load evaluation package
install.packages("gmodels")
library(gmodels)

# detailed confusion table
CrossTable(x = test_label, y = knn_pred, prop.chisq = FALSE)

# compute accuracy score
accuracy <- sum(knn_pred == test_label) / length(test_label)
accuracy

# define k values for plotting
k_values <- 1:20

accuracy_values <- sapply(k_values, function(k) {
  pred <- knn(train_bc, test_bc, train_label, k = k)
  mean(pred == test_label)
})

# store accuracy results
accuracy_data <- data.frame(K = k_values, Accuracy = accuracy_values)

# load plotting library
library(ggplot2)

# plot accuracy curve
ggplot(accuracy_data, aes(x = K, y = Accuracy)) +
  geom_line(color = "#2C7FB8", size = 1) +
  geom_point(color = "#F46D43", size = 2.5) +
  labs(
    title = "kNN Accuracy Across Different K Values",
    subtitle = "Breast Cancer (Binary Classification)",
    x = "Number of Neighbors (K)",
    y = "Accuracy"
  ) +
  theme_classic(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold"),
    plot.subtitle = element_text(color = "grey40"),
    axis.title = element_text(face = "bold")
  )