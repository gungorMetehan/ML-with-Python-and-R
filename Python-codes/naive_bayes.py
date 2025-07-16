from sklearn.datasets import load_iris # data set
from sklearn.model_selection import train_test_split # data set splitting
from sklearn.naive_bayes import GaussianNB # model fitting
from sklearn.metrics import classification_report, confusion_matrix # accuracy and confusion matrix
import seaborn as sns # confusion matrix
import matplotlib.pyplot as plt # confusion matrix

# data set
iris = load_iris()

# data manipulation
X = iris.data
y = iris. target

# data set splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# model fitting
naive_b = GaussianNB().fit(X_train, y_train)

# y_pred
y_pred = naive_b.predict(X_test)

# accuracy
print(classification_report(y_test, y_pred))

# confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot = True, cmap = 'Blues', fmt = 'g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
