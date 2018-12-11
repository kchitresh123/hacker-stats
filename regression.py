#!/usr/bin/env python

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score

import matplotlib.pyplot as plt

# Linear regression over a single feature 'fertility'

df = pd.read_csv('gapminder.csv')
X = df['fertility'].values.reshape(-1, 1)
y = df['life'].values.reshape(-1, 1)
prediction_space = np.linspace(min(X), max(X)).reshape(-1, 1)

reg = LinearRegression()
reg.fit(X, y)

score = reg.score(X, y)
print(f"R^2={score}")

y_pred = reg.predict(prediction_space)

# Linear regression over all features

df = pd.read_csv('gapminder.csv')
X = df.drop(['life', 'Region'], axis=1).values
y = df['life'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
score = np.mean(cross_val_score(reg_all, X, y, cv=10))
print(f"R^2={score}")

y_pred_all = reg_all.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_all))
print(f"Root Mean Squared Error is {rmse}")

X_test_fertility = X_test[:, 1]

plt.title(f"Linear Regression")
plt.scatter(df['fertility'], df['life'])
plt.plot(prediction_space, y_pred, color='black')
plt.scatter(X_test_fertility, y_pred_all, color='red')
plt.legend()
plt.xlabel(f"Fertility")
plt.ylabel(f"Life Expectancy")
plt.show()

# Significant feature detection

lasso = Lasso(alpha=0.4, normalize=True)
lasso.fit(X_train, y_train)

df_columns = df.drop(['life', 'Region'], axis=1).columns

plt.title(f"Life expectancy significant features detection")
plt.plot(range(len(df_columns)), lasso.coef_)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.xlabel(f"Feature")
plt.ylabel(f"Lasso coefficient")
plt.show()
