import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd


disease = datasets.load_diabetes()


disease_x = disease.data
disease_y = disease.target


X_train, X_test, y_train, y_test = train_test_split(disease_x, disease_y,
                                                    test_size=0.2,random_state=42)


reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

y_predict = reg.predict(X_test)

print("Mean Absolute Error: ", mean_absolute_error(y_test, y_predict))

weights = reg.coef_
bias = reg.intercept_
print(f"Weights:{weights}, Bias:{bias}")

plt.scatter(X_test[:,0],y_test)
plt.plot(X_test, y_predict, color='red')
plt.show()