
from sklearn import datasets
import pandas as pd

# loading iris data set
iris = datasets.load_iris()

# converting bunch to data frame
iris_df = pd.DataFrame(iris.data)
iris_df['Species'] = iris.target

# missing values?
iris_df.isnull().sum()

# importing train_test_split() function
from sklearn.model_selection import train_test_split

# using 80% of dataset as training set and 20% as test set
trainset, testset = train_test_split(iris_df, test_size = 0.2, random_state = 12)

trainset
testset
