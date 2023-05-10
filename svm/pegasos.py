import time
import argparse
import numpy as np
import utils


def hinge_loss(w, X, y):
    '''
    Compute the hinge loss:
    hinge loss = \frac{1}{n} \sum_{i=1}^{n} max(0, 1 - y_i w^\top x_i)

    Note: this is mostly used to compute the loss to keep track of
    the training progress so we do not include the regularization term here.

    Parameters
    ----------
    w: matrix of shape (d, 1)
       Weight vector

    X: matrix of shape (n, d)
        Training data

    y: matrix of shape (n, 1)
        Target values

    Returns
    -------
    loss: scalar
          SVM hinge loss
    '''
    n, d = X.shape
    scores = X.dot(w)
    margin = np.maximum(0, 1 - y * scores)
    loss = np.sum(margin) / n
    return loss


def eval_model(w, X, y):
    '''
    Return a tuple of the hinge loss and accuracy

    Parameters
    ----------
    w: matrix of shape (d, 1)
       Weight vector

    X: matrix of shape (n, d)
       Training data

    y: matrix of shape (n, 1)
       Target values

    Returns
    -------
    loss: float
          SVM hinge loss

    acc: float
         accuracy
    '''
    n, d = X.shape
    y_pred = np.sign(X.dot(w))
    y = np.sign(y)
    acc = np.sum(y_pred == y)/n
    loss = hinge_loss(w, X, y)
    return loss, acc


def grad(w, X, y, lamda):
    '''
    Parameters
    ----------
    w: matrix of shape (d, 1)
       Weight vector

    X: matrix of shape (n, d)
        Training data

    y: matrix of shape (n, 1)
        Target values

    lamda: scalar
           learning rate scaling parameter

    Returns
    -------
    grad_w: matrix of shape (d, 1)
            Gradient of of the SVM primal objective with respect to w over the given
            data X, y
    '''
    n = X.shape[0]
    z = y * X.dot(w)
    ind = np.where(z < 1)[0]
    grad_w = -(X[ind].T.dot(y[ind])) / n + lamda * w
    return grad_w


def get_batch(X, y, batch_size):
    '''
    Parameters
    ----------
    X: matrix of shape (n, d)
        Training data

    y: matrix of shape (n, 1)
        Target values

    batch_size: int
                size of batch

    Returns
    -------
    X_batch: matrix of shape (batch_size, d)

    y_batch: matrix of shape (batch_size, 1)
    '''
    idx = np.random.choice(X.shape[0], batch_size, replace=False)
    X_batch = X[idx, :]
    y_batch = y[idx]
    return X_batch, y_batch


def train_svm(args):
    X_train, y_train, X_test, y_test = utils.load_data(
        args.fname, add_bias=True)
    n, d = X_train.shape
    w = np.zeros((d, 1))

    # set hyperparameters
    epochs = args.epochs
    lamda = args.lamda
    batch_size = args.batch_size
    eta = 1 / (lamda * n)

    # initialize variables for monitoring training progress
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    prev_w = np.copy(w)
    delta_w = np.inf
    tol = 1e-4

    # start training loop
    for t in range(1, epochs+1):
        X_batch, y_batch = get_batch(X_train, y_train, batch_size)
        grad_w = grad(w, X_batch, y_batch, lamda)
        w = min(1, np.sqrt(2 * lamda / np.linalg.norm(grad_w, ord=2)**2)
                ) * w - eta * grad_w

        train_loss, train_acc = eval_model(w, X_train, y_train)
        test_loss, test_acc = eval_model(w, X_test, y_test)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        print('Epoch: {} | Train acc: {:.3f} | Test acc: {:.3f}'.format(
            t, train_acc, test_acc))
        prev_w = np.copy(w)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, default='news.mat')
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--lamda', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=15)
    args = parser.parse_args()

    train_svm(args)
