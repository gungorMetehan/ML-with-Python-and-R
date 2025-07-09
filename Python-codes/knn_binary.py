from sklearn.datasets import load_breast_cancer # data set
from sklearn.neighbors import KNeighborsClassifier # knn algorithm
from sklearn.metrics import accuracy_score, confusion_matrix # accuracy
from sklearn.model_selection import train_test_split # splitting data set
from sklearn.preprocessing import StandardScaler # scaling
import pandas as pd # data manipulation
import matplotlib.pyplot as plt # data visualization
import numpy as np # data visualization

# loading data and data manipulation
cancer = load_breast_cancer()
df = pd.DataFrame(data = cancer.data, columns = cancer.feature_names)
df["target"] = cancer.target
# defining features and target variable
X = cancer.data
y = cancer.target
# splitting data set (train - test, 80%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 12)

# scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# model fitting
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# y_pred
y_pred = knn.predict(X_test)

# accuracy score
accuracy = accuracy_score(y_test, y_pred)
accuracy
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix

# hyperparameter tuning
accuracy_vals = []
k_vals = []
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_vals.append(accuracy)
    k_vals.append(k)

# k values data visualization
## plotting
plt.figure(figsize = (10, 6))
plt.plot(k_vals, accuracy_vals, color = "royalblue", marker = "o", linestyle = "-", markersize = 8, linewidth = 2)
## titles and labels
plt.title("Accuracy by Number of Neighbors (K)", fontsize = 14)
plt.xlabel("Number of Neighbors (K)", fontsize = 12)
plt.ylabel("Accuracy", fontsize = 12)
## x-axis with integer ticks only
plt.xticks(ticks = np.arange(1, 21, 1))
## grid
plt.grid(True, linestyle = "--", alpha = 0.7)
## tidy layout
plt.tight_layout()
## showing the plot
plt.show()
