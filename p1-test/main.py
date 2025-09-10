"""
===========================================
Ordinary Least Squares and Ridge Regression
===========================================

1. Ordinary Least Squares:
   We illustrate how to use the ordinary least squares (OLS) model,
   :class:`~sklearn.linear_model.LinearRegression`, on a single feature of
   the diabetes dataset. We train on a subset of the data, evaluate on a
   test set, and visualize the predictions.

2. Ordinary Least Squares and Ridge Regression Variance:
   We then show how OLS can have high variance when the data is sparse or
   noisy, by fitting on a very small synthetic sample repeatedly. Ridge
   regression, :class:`~sklearn.linear_model.Ridge`, reduces this variance
   by penalizing (shrinking) the coefficients, leading to more stable
   predictions.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Data Loading and Preparation
# ----------------------------
#
# Load the diabetes dataset. For simplicity, we only keep a single feature in the data.
# Then, we split the data and target into training and test sets.
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

print("All imports OK")

X, y = load_diabetes(return_X_y=True)
X = X[:, [2]]  # Use only one feature
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, shuffle=False)

# %%
# Linear regression model
# -----------------------
#
# We create a linear regression model and fit it on the training data. Note that by
# default, an intercept is added to the model. We can control this behavior by setting
# the `fit_intercept` parameter.


regressor = LinearRegression().fit(X_train, y_train)

# %%
# Model evaluation
# ----------------
#
# We evaluate the model's performance on the test set using the mean squared error
# and the coefficient of determination.

y_pred = regressor.predict(X_test)

print(f"Mean squared error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"Coefficient of determination: {r2_score(y_test, y_pred):.2f}")

# %%
# Conclusion
# ----------
#
# - In the first example, we applied OLS to a real dataset, showing
#   how a plain linear model can fit the data by minimizing the squared error
#   on the training set.
#
# - In the second example, OLS lines varied drastically each time noise
#   was added, reflecting its high variance when data is sparse or noisy. By
#   contrast, **Ridge** regression introduces a regularization term that shrinks
#   the coefficients, stabilizing predictions.
#
# Techniques like :class:`~sklearn.linear_model.Ridge` or
# :class:`~sklearn.linear_model.Lasso` (which applies an L1 penalty) are both
# common ways to improve generalization and reduce overfitting. A well-tuned
# Ridge or Lasso often outperforms pure OLS when features are correlated, data
# is noisy, or sample size is small.
