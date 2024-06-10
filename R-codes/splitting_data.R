### Method 1 ###

# examining the dataset
data(iris)
head(iris); tail(iris)

# missing values?
table(is.na(iris))

# making this example reproducible
set.seed(12)

# using 80% of dataset as training set and 20% as test set
sample <- sample(c(TRUE, FALSE), nrow(iris), replace = TRUE, prob = c(0.8, 0.2))
iris_train  <- iris[sample, ]
iris_test   <- iris[!sample, ]

# let's look at the training and test datasets
View(iris_train)
View(iris_test)



### Method 2 ###

# examining the dataset
data(iris)
head(iris); tail(iris)

# missing values?
table(is.na(iris))

# installing "caTools" package
install.packages("caTools")
library(caTools)

# making this example reproducible
set.seed(12)

# using 80% of dataset as training set and 20% as test set
logic_iris <- sample.split(iris$Species, SplitRatio = .8)
logic_iris
train_iris <- subset(iris, logic_iris == TRUE)
test_iris <- subset(iris, logic_iris == FALSE)

# let's look at the training and test datasets
View(train_iris)
View(test_iris)



### Method 3 ###

# examining the dataset
data(iris)
head(iris); tail(iris)

# missing values?
table(is.na(iris))

# installing "dplyr" package
install.packages("dplyr")
library(dplyr)

# making this example reproducible
set.seed(12)

# creating ID column
iris$id <- 1:nrow(iris)

# using 80% of dataset as training set and 20% as test set
tr_iris <- iris |> sample_frac(0.80)
te_iris  <- anti_join(iris, tr_iris, by = "id")

# let's look at the training and test datasets
View(tr_iris)
View(te_iris)
