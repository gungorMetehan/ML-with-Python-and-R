from sklearn.datasets import load_iris # data set
from sklearn.model_selection import train_test_split # data set splitting

# data set
iris = load_iris()

# data manipulation
X = iris.data
y = iris. target

# data set splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 12)
