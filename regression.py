#!/usr/bin/env python

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

df = pd.read_csv('gapminder.csv')
X = df['fertility'].values.reshape(-1, 1)
y = df['life'].values.reshape(-1, 1)

prediction_space = np.linspace(min(X), max(X)).reshape(-1, 1)
reg = LinearRegression()
reg.fit(X, y)
y_pred = reg.predict(prediction_space)

print(reg.score(X, y))

plt.title(f"Linear Regression")
plt.scatter(df['fertility'], df['life'])
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.legend()
plt.xlabel(f"Fertility")
plt.ylabel(f"Life Expectancy")
plt.show()
