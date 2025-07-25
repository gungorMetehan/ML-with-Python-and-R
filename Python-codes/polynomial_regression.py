import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # data visualization
from sklearn.datasets import fetch_california_housing # data set
from sklearn.linear_model import LinearRegression # model fitting
from sklearn.preprocessing import PolynomialFeatures # degree
from sklearn.model_selection import train_test_split # data set splitting
from sklearn.metrics import mean_squared_error, r2_score # mse & r^2

# data set
california_housing = fetch_california_housing()
X = california_housing.data[:, [0]]             # only 'MedInc'
y = california_housing.target                   # 'MedHouseVal'

# data set splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# model fitting
degree = 2
poly = PolynomialFeatures(degree = degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

# y_pred
y_pred = model.predict(X_test_poly)

# mse & r^2
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.3f}")
print(f"R^2: {r2:.3f}")

# data visualization
sorted_idx = X_test[:, 0].argsort()
X_sorted = X_test[sorted_idx]
y_sorted = y_test[sorted_idx]
y_pred_sorted = y_pred[sorted_idx]

plt.figure(figsize = (10, 6))
plt.scatter(X_test, y_test, color = "gray", edgecolors = "black", alpha = 0.4, label = "Data Points")
plt.plot(X_sorted, y_pred_sorted, color = "red", linewidth = 2, label = f'Polynomial Regression (Degree = {degree})')
plt.xlabel('Average Income')
plt.ylabel('Average House Value')
plt.title('Polynomial Regression: MedInc ~ MedHouseVal')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# predict
random_medinc_values = np.array([[2.5], [5.5], [8.0], [10.0], [12.5]])

## trasformation
random_medinc_poly = poly.transform(random_medinc_values)

## prediction table
predicted_house_values = model.predict(random_medinc_poly)

results_df = pd.DataFrame({
    'Average Income': random_medinc_values.flatten(),
    'Predicted House Value': predicted_house_values.round(3)
    })
print("Prediction Table:")
print(results_df)