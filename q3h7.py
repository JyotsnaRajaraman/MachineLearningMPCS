import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Load the Heart dataset from CSV
heart_data = pd.read_csv('Heart.csv')

# Preprocess categorical columns using one-hot encoding
# Replace with your categorical column names
categorical_cols = ['Age', 'Sex', 'ChestPain', 'RestBP', 'Chol',
                    'Fbs', 'RestECG', 'MaxHR', 'ExAng', 'Oldpeak', 'Slope', 'Ca', 'Thal']
encoder = OneHotEncoder(sparse_output=False)
encoded_cols = pd.DataFrame(
    encoder.fit_transform(heart_data[categorical_cols]))
heart_data = pd.concat(
    [heart_data.drop(categorical_cols, axis=1), encoded_cols], axis=1)
# Convert feature names to strings
heart_data.columns = heart_data.columns.astype(str)
# Separate the features (X) and target variable (y)
X = heart_data.drop('AHD', axis=1)
y = heart_data['AHD']


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
# Calculate the number of features
p = X_train.shape[1]


# Set the number of bootstrap samples
B_values = [50, 100, 150, 200, 250, 300]

# Initialize lists to store errors
test_errors_bagging = []
test_errors_random_forest = []
oob_errors_bagging = []
oob_errors_random_forest = []

for B in B_values:
    # Create a BaggingClassifier with DecisionTreeClassifier as the base estimator
    bagging = BaggingClassifier(estimator=DecisionTreeClassifier(
    ), n_estimators=B, random_state=42, oob_score=True)
    bagging.fit(X_train, y_train)

    # Create a RandomForestClassifier with sqrt(p) as the max_features
    random_forest = RandomForestClassifier(
        n_estimators=B, max_features=int(np.sqrt(p)), random_state=42, oob_score=True)
    random_forest.fit(X_train, y_train)

    # Calculate the test error for Bagging
    test_errors_bagging.append(1 - bagging.score(X_test, y_test))

    # Calculate the test error for Random Forest
    test_errors_random_forest.append(1 - random_forest.score(X_test, y_test))

    # Calculate the OOB error for Bagging
    oob_errors_bagging.append(1 - bagging.oob_score_)

    # Calculate the OOB error for Random Forest
    oob_errors_random_forest.append(1 - random_forest.oob_score_)

# Plot the test error and OOB error as a function of B
plt.plot(B_values, test_errors_bagging, color='black', label='Test: Bagging')
plt.plot(B_values, test_errors_random_forest,
         color='orange', label='Test: Random Forest')
plt.plot(B_values, oob_errors_bagging, color='green', label='OOB: Bagging')
plt.plot(B_values, oob_errors_random_forest,
         color='blue', label='OOB: Random Forest')
plt.axhline(y=1 - bagging.score(X_train, y_train), color='red',
            linestyle='--', label='Single Tree Error')
plt.xlabel('Number of Bootstrapped Training Sets (B)')
plt.ylabel('Error Rate')
plt.legend()
plt.title('Bagging and Random Forest Error Rates')
plt.show()
