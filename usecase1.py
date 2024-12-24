import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

cars = pd.read_csv("cars.csv")
print(cars.columns)

plt.figure(figsize=(12,6))
plt.scatter(cars['Model'], cars['Variant'],c="black")
plt.xlabel('Model')
plt.ylabel('Variant')
plt.title('Model vs Variant')
plt.tight_layout()
plt.show()