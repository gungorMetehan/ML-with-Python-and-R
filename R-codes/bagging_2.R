# loading data set
install.packages("palmepenguins")
library(palmerpenguins)

# viewing the data set
penguins

# missing values?
table(is.na(penguins))

# omitting missing values (there are better solutions)
penguins2 <- na.omit(penguins)
penguins2

# deleting variables that will not be used
penguins3 <- penguins2[, c(-2, -7, -8)]
penguins3

# making this example reproducible
set.seed(12)

# installing and loading necessary packages
install.packages("dplyr")
library(dplyr)    #for data wrangling
install.packages("e1071")
library(e1071)    #for calculating variable importance
install.packages("caret")
library(caret)    #for model fitting
install.packages("rpart")
library(rpart)    #for fitting decision trees
install.packages("ipred")
library(ipred)    #for fitting bagged decision trees

# modeling (1000 trees, out-of-bag error will be obtained (coob = T), 2 observations in a node to split, control parameter = 0)
bagging_model <- bagging(
  formula = species ~ .,
  data = penguins3,
  nbagg = 1000,   
  coob = TRUE,
  control = rpart.control(minsplit = 2, cp = 0)
)

# model
bagging_model

# calculating variable importance
var_importance <- data.frame(variables = names(penguins3[, 2:5]), VI = varImp(bagging_model))
var_importance

# sorting variable importance values descending
VIdata_4plot <- data.frame(variable = row.names(VI), Overall = round(var_importance$Overall, 2))
VIdata_4plot

# visualizing variable importance
install.packages("ggplot2")
library(ggplot2)

ggplot(data = VIdata_4plot, mapping = aes(x = reorder(variable, -Overall), Overall)) + 
  geom_bar(stat = "identity", width = .75, fill = "#74BEFF") + 
  coord_flip() +
  geom_label(label= VIdata_4plot$Overall) +
  xlab("Variable Importance") + 
  ylab("Variables") +
  theme_classic()

# defining new observation
new_obs <- data.frame(bill_length_mm = 39, bill_depth_mm = 18, flipper_length_mm = 180, body_mass_g = 3650)

# using fitted bagged model to predict 'species'
predict(bagging_model, newdata = new_obs)
