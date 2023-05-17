import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class AdaBoostClassifier:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.estimators = []
        self.estimator_weights = []
        self.estimator_errors = []

    def fit(self, X, y):
        # Initialize observation weights
        N = X.shape[0]
        sample_weights = np.ones(N) / N

        for _ in range(self.n_estimators):
            # Fit a classifier to the training data using weights
            estimator = DecisionTreeClassifier(max_depth=10)
            estimator.fit(X, y, sample_weight=sample_weights)

            # Predict labels for training examples
            y_pred = estimator.predict(X)

            # Compute the weighted error
            error = np.sum(sample_weights * (y_pred != y)) / \
                np.sum(sample_weights)

            # Compute the estimator weight
            alpha = np.log((1 - error) / error)

            # Update the observation weights
            sample_weights *= np.exp(alpha * (y_pred != y))

            # Normalize the weights
            sample_weights /= np.sum(sample_weights)

            # Store the estimator and its weight and error
            self.estimators.append(estimator)
            self.estimator_weights.append(alpha)
            self.estimator_errors.append(error)

    def predict(self, X):
        # Initialize the predicted labels
        y_pred = np.zeros(X.shape[0])

        for estimator, weight in zip(self.estimators, self.estimator_weights):
            # Predict labels using each estimator
            y_pred += weight * estimator.predict(X)

        # Apply sign function to get the final predicted labels
        y_pred = np.sign(y_pred)

        return y_pred


# Creating a normally distributed array
X = np.random.normal(0, 1, size=[12000, 10])

# Assigning y = 1 if sum(X2j) > χ2 10(0.5), which evaluates to 9.34
# Assigning -1 otherwise
y = (np.sum(X ** 2, axis=1) > 9.34).astype(int)
y = y * 2 - 1

# Splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000)

ada_boost = AdaBoostClassifier(n_estimators=100)
ada_boost.fit(X_train, y_train)

# b)
train_errors = [1 - (1 - error) for error in ada_boost.estimator_errors]
test_errors = []
for estimator in ada_boost.estimators:
    y_pred = estimator.predict(X_test)
    error = np.mean(y_pred != y_test)
    test_errors.append(error)

iterations = range(1, len(train_errors) + 1)

plt.plot(iterations, train_errors, label='Training Error')
plt.plot(iterations, test_errors, label='Test Error')
plt.xlabel('Number of Iterations')
plt.ylabel('Error')
plt.legend()
plt.show()

# (c) Investigating the number of iterations needed for the test error to start rising
threshold = 0.1  # Define a threshold for the test error increase
num_iterations = np.where(np.array(test_errors) > threshold)[0][0] + 1
print("Number of iterations for test error to start rising:", num_iterations)

# (d) Changing the setup of the example and repeating the AdaBoost experiment
X_class1 = np.random.normal(0, 1, size=[12000, 10])
X_class2 = np.random.normal(0, 1, size=[12000, 10])
condition = np.abs(X_class2[:, 1]) > 12
X_class2[condition, :] = np.random.normal(0, 1, size=[np.sum(condition), 10])

X = np.concatenate((X_class1, X_class2))
y = np.concatenate((np.ones(len(X_class1)), -np.ones(len(X_class2))))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000)


ada_boost = AdaBoostClassifier(n_estimators=100)
ada_boost.fit(X_train, y_train)

train_errors = [1 - (1 - error) for error in ada_boost.estimator_errors]
test_errors = []
for estimator in ada_boost.estimators:
    y_pred = estimator.predict(X_test)
    error = np.mean(y_pred != y_test)
    test_errors.append(error)

iterations = range(1, len(train_errors) + 1)

plt.plot(iterations, train_errors, label='Training Error')
plt.plot(iterations, test_errors, label='Test Error')
plt.xlabel('Number of Iterations')
plt.ylabel('Error')
plt.legend()
plt.show()
