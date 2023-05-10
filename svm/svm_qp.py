import argparse
import cvxopt
import numpy as np
import utils


def predict(X, w, bias):
    """
    Parameters
    ----------
    X: matrix of shape (n, d)
       Training data

    w: matrix of shape (d, 1)
       SVM weight vector

    bias: scalar

    Returns
    -------
    y_pred: matrix of shape (n, 1)
          Predicted values
    """
    return np.sign(X @ w + bias)


def make_P(X, y, C):
    """
    Parameters
    ----------
    X: matrix of shape (n, d)
       Training data

    y: matrix of shape (n, 1)
       Target values

    C: scalar
       Penalty parameter

    Returns
    -------
    P: matrix of shape (n, n)
       positive semidefinite matrix for the quadratic program
    """
    n = X.shape[0]
    K = X @ X.T * y @ y.T
    return cvxopt.matrix(K)


def make_q(n):
    """
    Return the q vector in the standard quadratic program formulation of the SVM dual problem

    Parameters
    ----------
    n: int
       Number of data points

    Returns
    -------
    q: matrix of shape (n, 1)
       positive semidefinite matrix for the quadratic program
    """
    return cvxopt.matrix(-np.ones((n, 1)))


def make_inequality_constraints(X, y, C):
    """
    Return the G, h matrices/vectors in the standard quadratic program formulation
       for the SVM dual problem

    Parameters
    ----------
    X: matrix of shape (n, d)
       Training data

    y: matrix of shape (n, 1)
       Target values

    C: scalar
       Penalty parameter

    Returns
    -------
    G: matrix of shape (m, n)

    h: matrix of shape (m, 1)
    """
    n = X.shape[0]
    G = -np.eye(n)
    h = np.zeros((n, 1))
    return cvxopt.matrix(G * y), cvxopt.matrix(h)


def make_equality_constraints(y):
    """
    Return the A, b matrices/vectors in the standard quadratic program for the SVM dual problem

    Parameters
    ----------
    y: matrix of shape (n, 1)
       Target values

    Returns
    -------
    A: matrix of shape (p, n)

    b: matrix of shape (p, 1)
    """
    return cvxopt.matrix(y.T), cvxopt.matrix(0.0)


def accuracy(X, y, w, bias):
    """
    Compute the accuracy of the prediction rule determined by the
       given weight and bias vector on input data X, y

    Parameters
    ----------
    X: matrix of shape (n, d)
       Training data

    y: matrix of shape (n, 1)
       Target values

    w: matrix of shape (d, 1)
       SVM weight vector

    bias: scalar
          SVM bias term

    Returns
    -------
    acc: float
       accuracy
    """
    y_pred = predict(X, w, bias)
    return np.mean(y_pred == y)


def make_weight_bias(X, y, qp_solution):
    '''
    Given the solution of the SVM dual quadratic program
    construct the corresponding w weight vector

    Parameters
    ----------
    X: matrix of shape (n, d)
       Training data

    y: matrix of shape (n, 1)
       Target values

    qp_solution: output of cvxopt.solvers.qp

    Returns
    -------
    w: vector of shape (d, 1)
       SVM weight vector
    bias: scalar
          bias term
    '''
    alpha = np.array(qp_solution['x']).reshape(-1, 1)
    w = (X.T @ (alpha * y)).reshape(-1, 1)
    support_vectors = alpha > 1e-5
    support_vector_indices = np.where(support_vectors)[0]
    bias = np.mean(y[support_vectors] - X[support_vector_indices] @ w)
    return w, bias


def linear_kernel(X):
    return X @ X.T


def dual_svm(X_train, y_train, X_test, y_test, C):
    '''
    Minimize     1/2 alpha^T P alpha - q^T x
    Subject to   Gx <= h
                Ax  = b

    here alphas = x
    G = X @ X.T
    '''

    # Convert sparse matrix to dense matrix and then to numpy array of floats
    X_train = np.asarray(X_train.toarray(), dtype=float)

    # Construct P
    P = cvxopt.matrix(np.outer(y_train, y_train) *
                      linear_kernel(X_train))

    # Construct q
    q = cvxopt.matrix(np.ones(len(X_train)) * -1)

    # Construct G
    G = cvxopt.matrix(
        np.vstack((np.eye(len(X_train)) * -1, np.eye(len(X_train)))))

    # Construct h
    h = cvxopt.matrix(
        np.hstack((np.zeros(len(X_train)), np.ones(len(X_train)) * C)))

    # Construct A
    A = cvxopt.matrix(y_train.reshape(1, -1))

    # Construct b
    b = cvxopt.matrix(0.0)

    # Solve the optimization problem using cvxopt.solvers.qp function
    sol = cvxopt.solvers.qp(P, q, G, h, A, b)

    weight, bias = make_weight_bias(X_train, y_train, sol)
    test_acc = accuracy(X_test, y_test, weight, bias)
    train_acc = accuracy(X_train, y_train, weight, bias)
    print("C value: " + str(C))
    print("Train acc: {:.3f}".format(train_acc))
    print("Test acc: {:.3f}".format(test_acc))


def main(args):
    # Note that we do not add bias here
    X_train, y_train, X_test, y_test = utils.load_data(
        args.fname, add_bias=False)
    dual_svm(X_train, y_train, X_test, y_test, args.C)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, default='news.mat')
    parser.add_argument('--C', type=float, default=100)
    args = parser.parse_args()
    main(args)
