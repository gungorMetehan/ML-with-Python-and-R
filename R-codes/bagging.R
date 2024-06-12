# importing data
data(airquality)
str(airquality)

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
  formula = Ozone ~ .,
  data = airquality,
  nbagg = 1000,   
  coob = TRUE,
  control = rpart.control(minsplit = 2, cp = 0)
)

# model
bagging_model

# calculating variable importance
var_importance <- data.frame(variable = names(airquality[, -1]), VI = varImp(bagging_model))
var_importance

# sorting variable importance values descending
VIdata_4plot <- var_importance[order(var_importance$Overall, decreasing = TRUE), ]
VIdata_4plot$Overall <- round(VIdata_4plot$Overall, 2)

# visualizing variable importance
install.packages("ggplot2")
library(ggplot2)

ggplot(data = VIdata_4plot, mapping = aes(x = reorder(variable, -Overall), Overall)) + 
  geom_bar(stat = "identity", width = .75, fill = "#7FEE7C") + 
  coord_flip() +
  geom_label(label= VIdata_4plot$Overall) +
  xlab("Variable Importance") + 
  ylab("Variables") +
  theme_classic()

# defining new observation
new_obs <- data.frame(Solar.R = 310, Wind = 10, Temp = 60, Month = 5, Day = 4)

# using fitted bagged model to predict 'Ozone' value of new observation
predict(bagging_model, newdata = new_obs)
