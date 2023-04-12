from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

mpl.rc('figure', figsize=[10, 6])


def prediction(X, theta):
    'Return yhat for given inputs X and parameters theta.'
    return (1/(1 + np.exp(-np.matmul(X, theta))))


def loss(X, y, theta, lamb):
    '''
    Return the loss for given inputs X, targets y, parameters theta,
    and regularization coefficient lamb.
    '''
    yhat = prediction(X, theta)
    return (-((y * np.log(yhat)) +
            ((1 - y) * np.log(1-yhat))
    ).mean(axis=0) +
        (lamb) * (theta**2).sum(axis=0)
    )


def gradient_descent(X, y, alpha, lamb, T, theta_init=None):

    theta = theta_init.copy()

    ### YOUR CODE HERE ###
    # Add a column of ones to the left of X for the bias
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    # Initialize theta
    theta = theta_init

    # Initialize an empty list to store the cost values
    cost_history = []

    # Loop over the number of iterations
    for i in range(T):
        # Calculate the predicted labels
        yhat = prediction(X, theta)

        # Calculate the gradient
        gradient = (np.dot(X.T, yhat - y) /
                    X.shape[0]) + (2 * lamb * theta)

        # Update theta
        theta = theta - (alpha * gradient)

        # Calculate the cost (loss + complexity) and append to cost history
        cost = loss(X, y, theta, lamb)
        cost_history.append(cost)

    # Convert cost history to a numpy array
    cost_history = np.array(cost_history)

    # Plot the cost vs iteration number
    plt.plot(np.arange(1, T+1), cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost vs Iteration')
    plt.show()

    return (theta)


def test_gradient_descent():
    X = np.array([[-0.31885351, -1.60298056],
                  [-1.53521787, -0.57040089],
                  [-0.2167283,  0.2548743],
                  [-0.14944994,  2.01078257],
                  [-0.09678416,  0.42220166],
                  [-0.22546156, -0.63794309],
                  [-0.0162863,  1.04421678],
                  [-1.08488033, -2.20592483],
                  [-0.95121901,  0.83297319],
                  [-1.00020817,  0.34346274]])
    y = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    alpha = 0.1
    lamb = 1
    theta_init = np.zeros(X.shape[1]+1)
    assert np.allclose(
        gradient_descent(X, y, alpha, lamb, 1, theta_init),
        np.array([-0.03,  0.0189148,  0.0256793]))
    assert np.allclose(
        gradient_descent(X, y, alpha, lamb, 2, theta_init),
        np.array([-0.05325034,  0.0333282,  0.04540004]))
    assert np.allclose(
        gradient_descent(X, y, alpha, lamb, 3, theta_init),
        np.array([-0.07127091,  0.04431147,  0.06054757]))
    print('test_gradient_descent passed')


def run_prob_4():
    test_gradient_descent()


df = pd.read_csv('wdbc.data', header=None)
base_names = ['radius', 'texture', 'perimeter', 'area', 'smooth', 'compact', 'concav',
              'conpoints', 'symmetry', 'fracdim']
names = ['m' + name for name in base_names]
names += ['s' + name for name in base_names]
names += ['e' + name for name in base_names]
names = ['id', 'class'] + names
df.columns = names
df['color'] = pd.Series([(0 if x == 'M' else 1) for x in df['class']])
my_color_map = mpl.colors.ListedColormap(['r', 'g'], 'mycolormap')


X = df.iloc[:, 2:-1].values  # Input feature matrix
y = df['color'].values  # Output vector
m, n = X.shape  # Number of examples and features
# Normalize the input features
X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# parameters for df
alpha = 0.01  # Learning rate
lamb = 1  # Regularization strength
T = 1000  # Number of iterations
theta_init = np.zeros(n + 1)  # Initial parameter vector


def run_prob_5():
    # Run gradient descent
    theta = gradient_descent(X_norm, y, alpha, lamb, T, theta_init)

    # Print the obtained theta values
    print('Theta values for gradient descent implementation: ', theta)


def run_prob_6():

    # Fit the logistic regression model
    clf = LogisticRegression(penalty='l2', C=1)
    clf.fit(X_norm, y)
    theta = np.concatenate(([clf.intercept_[0]], clf.coef_[0]))

    print('Theta values for scikit-learn LR: ', theta)


c1 = 'mradius'
c2 = 'mtexture'

clf = LogisticRegression(solver='lbfgs')
clf.fit(df[[c1, c2]], df['color'])


def run_prob_7():
    plt.scatter(df[c1], df[c2], c=df['color'], cmap=my_color_map)

    plt.xlabel(c1)
    plt.ylabel(c2)

    x = np.linspace(df[c1].min(), df[c1].max(), 1000)
    y = np.linspace(df[c2].min(), df[c2].max(), 1000)
    xx, yy = np.meshgrid(x, y)
    predicted_prob = clf.predict_proba(
        np.hstack((xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1))
                  ))[:, 1]
    predicted_prob = predicted_prob.reshape(xx.shape)

    plt.contour(xx, yy, predicted_prob, [0.5], colors=['b'])
    plt.title('Scatter Plot with Decision Boundary')
    plt.show()


# To see answers to each question in the homework:

run_prob_4()
# run_prob_5()
# run_prob_6()
# run_prob_7()
