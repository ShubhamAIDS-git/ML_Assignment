# Machine Learning Assignment: Python Code Explanation

## Overview

This document explains the Python code used for dataset exploration, data splitting, and linear regression. The code is divided into three main steps:
1. Dataset Exploration using the Iris dataset.
2. Data Splitting for training and testing.
3. Linear Regression using a simulated `YearsExperience` and `Salary` dataset.

## Step 1: Dataset Exploration

### Loading the Iris Dataset
The Iris dataset is a well-known dataset used in machine learning. It contains 150 samples of iris flowers, with 4 features: sepal length, sepal width, petal length, and petal width. The target variable has three classes representing different species of iris.

```python
from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
iris = load_iris()

# Create a DataFrame from the dataset
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
```

### Displaying the First Five Rows
To get a quick overview of the dataset, the first five rows are displayed.

```python
# Display the first five rows of the dataset
print("First five rows of the Iris dataset:")
print(iris_df.head())
```

### Dataset Shape and Summary Statistics
We check the dataset's shape (i.e., number of rows and columns) and calculate summary statistics for each feature (mean, standard deviation, min, max, etc.).

```python
# Get the dataset’s shape
print("\nDataset shape (rows, columns):", iris_df.shape)

# Calculate summary statistics for each feature
print("\nSummary statistics for each feature:")
print(iris_df.describe())
```

## Step 2: Data Splitting

### Splitting the Dataset
We split the Iris dataset into training and testing sets using an 80-20 split. This means 80% of the data will be used for training the model, and 20% will be used for testing it.

```python
from sklearn.model_selection import train_test_split

# Split the Iris dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
```

### Number of Samples in Each Set
After splitting, we print the number of samples in both the training and testing sets.

```python
# Get the number of samples in both the training and testing sets
print("\nNumber of samples in the training set:", X_train.shape[0])
print("Number of samples in the testing set:", X_test.shape[0])
```

## Step 3: Linear Regression

### Simulating the Dataset
For this task, we simulate a dataset where `YearsExperience` is the feature and `Salary` is the target variable. The `YearsExperience` ranges from 1 to 10 years, and the corresponding `Salary` values are provided.

```python
import numpy as np

# Simulate a dataset with YearsExperience and Salary
years_experience = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)

# Simulated corresponding salaries (in thousands)
salary = np.array([30, 35, 40, 45, 50, 60, 70, 80, 90, 100])
```

### Splitting the Simulated Dataset
Similar to the Iris dataset, we split the simulated dataset into training and testing sets.

```python
# Split this dataset into training and testing sets (80-20 split)
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(years_experience, salary, test_size=0.2, random_state=42)
```

### Fitting the Linear Regression Model
We fit a linear regression model to predict `Salary` based on `YearsExperience`.

```python
from sklearn.linear_model import LinearRegression

# Fit a linear regression model
model = LinearRegression()
model.fit(X_train_lr, y_train_lr)
```

### Model Evaluation
Finally, we predict the `Salary` for the test set and evaluate the model's performance using Mean Squared Error (MSE). The coefficient (slope) and intercept of the model are also printed.

```python
from sklearn.metrics import mean_squared_error

# Predict on the test set
y_pred_lr = model.predict(X_test_lr)

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test_lr, y_pred_lr)

print("\nLinear Regression Model:")
print("Coefficient (Slope):", model.coef_[0])
print("Intercept:", model.intercept_)
print("Mean Squared Error on the Test Set:", mse)
```

## Summary

- **Dataset Exploration:** Provided insights into the Iris dataset, including its shape and summary statistics.
- **Data Splitting:** The Iris dataset was split into training and testing sets.
- **Linear Regression:** A linear regression model was trained on a simulated `YearsExperience` vs. `Salary` dataset, and its performance was evaluated using MSE.
