import pdb
import time
import numpy as np

DT = np.float32  # DT refers to data type
eps = 1e-12  # Used for gradient testing

# Utility function for shape inference with broadcasting.
# Understanding the details of this function is not necessary for the homework.


def bcast(a, b):
    xs = np.array(a.shape)
    ys = np.array(b.shape)
    pad = len(xs) - len(ys)
    if pad > 0:
        ys = np.pad(ys, [[pad, 0]], 'constant')
    elif pad < 0:
        xs = np.pad(xs, [[-pad, 0]], 'constant')
    os = np.maximum(xs, ys)
    xred = tuple([idx for idx in np.where(xs < os)][0])
    yred = tuple([idx for idx in np.where(ys < os)][0])
    return xred, yred


def xavier(shape, type, seed=None):
    # Return an np array of given shape in which each element is chosen uniformly
    # at random from the Xavier initialization interval.
    n_in, n_out = shape
    if seed is not None:
        # set seed to fixed number (e.g. layer idx) for predictable results
        np.random.seed(seed)
    # todo initialize uniformly at random from [-sqrt(6/(n_in+n_out)), sqrt(6/(n_in+n_out))]
    limit = np.sqrt(6 / (n_in + n_out))
    if type == "w":
        return np.random.uniform(-limit, limit, size=shape).astype(DT)
    else:
        return np.random.uniform(-limit, limit, size=(n_out,)).astype(DT)


"""
Values
This is used for nodes corresponding to input values or other constants
that are not parameters (i.e., are not updated).
"""


class Value:
    def __init__(self, value=None):
        self.value = DT(value).copy()
        self.grad = None

    def set(self, value):
        self.value = DT(value).copy()


# Parameters
class Param:
    def __init__(self, value):
        self.value = DT(value).copy()
        self.grad = DT(0)


'''
  Class name: Add
  Class usage: add two matrices a, b with broadcasting supported by numpy "+" operation.
  Class function:
      forward: calculate a + b with possible broadcasting
      backward: calculate derivative w.r.t to a and b
'''


class Add:  # Add with broadcasting
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.grad = None if a.grad is None and b.grad is None else DT(0)
        self.value = None

    def forward(self):
        self.value = self.a.value + self.b.value

    def backward(self):
        xred, yred = bcast(self.a.value, self.b.value)
        if self.a.grad is not None:
            self.a.grad = self.a.grad + np.reshape(
                np.sum(self.grad, axis=xred, keepdims=True),
                self.a.value.shape)

        if self.b.grad is not None:
            self.b.grad = self.b.grad + np.reshape(
                np.sum(self.grad, axis=yred, keepdims=True),
                self.b.value.shape)


'''
Class Name: Mul
Class Usage: elementwise multiplication with two matrix 
Class Functions:
    forward: compute the result a*b
    backward: compute the derivative w.r.t a and b
'''


class Mul:  # Multiply with broadcasting
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.grad = None if a.grad is None and b.grad is None else DT(0)
        self.value = None

    def forward(self):
        self.value = self.a.value * self.b.value
        return self.value

    def backward(self):
        xred, yred = bcast(self.a.value, self.b.value)
        if self.a.grad is not None:
            self.a.grad = self.a.grad + np.reshape(
                np.sum(self.grad * self.b.value, axis=xred, keepdims=True),
                self.a.value.shape)

        if self.b.grad is not None:
            self.b.grad = self.b.grad + np.reshape(
                np.sum(self.grad * self.a.value, axis=yred, keepdims=True),
                self.b.value.shape)


'''
Class Name: VDot
Class Usage: matrix multiplication where a is a vector and b is a matrix
    b is expected to be a parameter and there is a convention that parameters come last. 
    Typical usage is a is a feature vector with shape (f_dim, ), b a parameter with shape (f_dim, f_dim2).
Class Functions:
     forward: compute the vector matrix multplication result
     backward: compute the derivative w.r.t a and b, where derivative of a and b are both matrices 
'''


class VDot:  # Matrix multiply (fully-connected layer)
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.grad = None if a.grad is None and b.grad is None else DT(0)
        self.value = None

    def forward(self):
        self.value = np.dot(self.a.value, self.b.value)

    def backward(self):
        if self.a.grad is not None:
            self.a.grad += np.dot(self.b.value, self.grad)
        if self.b.grad is not None:
            self.b.grad += np.outer(self.a.value, self.grad)


'''
Class Name: Sigmoid
Class Usage: compute the elementwise sigmoid activation. Input is vector or matrix. 
    In case of vector, [a_{0}, a_{1}, ..., a_{n}], output is vector [b_{0}, b_{1}, ..., b_{n}] where b_{i} = 1/(1 + exp(-a_{i}))
Class Functions:
    forward: compute activation b_{i} for all i.
    backward: compute the derivative w.r.t input vector/matrix a  
'''


class Sigmoid:
    def __init__(self, a):
        self.a = a
        self.grad = None if a.grad is None else DT(0)
        self.value = None

    def forward(self):
        # todo
        self.value = 1 / (1 + np.exp(-self.a.value))

    def backward(self):
        if self.a.grad is not None:
            # todo
            # raise NotImplementedError
            self.a.grad += self.grad * self.value * (1 - self.value)


'''
Class Name: RELU
Class Usage: compute the elementwise RELU activation. Input is vector or matrix. In case of vector, 
    [a_{0}, a_{1}, ..., a_{n}], output is vector [b_{0}, b_{1}, ..., b_{n}] where b_{i} = max(0, a_{i})
Class Functions:
    forward: compute activation b_{i} for all i.
    backward: compute the derivative w.r.t input vector/matrix a  
'''


class RELU:
    def __init__(self, a):
        self.a = a
        self.grad = None if a.grad is None else DT(0)
        self.value = None

    def forward(self):
        # todo
        # raise NotImplementedError
        self.value = np.maximum(0, self.a.value)

    def backward(self):
        if self.a.grad is not None:
            # todo
            self.a.grad += self.grad * (self.value > 0)


'''
Class Name: SoftMax
Class Usage: compute the softmax activation for each element in the matrix, normalization by each all elements 
    in each batch (row). Specifically, input is matrix [a_{00}, a_{01}, ..., a_{0n}, ..., a_{b0}, a_{b1}, ..., a_{bn}], 
    output is a matrix [p_{00}, p_{01}, ..., p_{0n},...,p_{b0},,,p_{bn} ] where p_{bi} = exp(a_{bi})/(exp(a_{b0}) + ... + exp(a_{bn}))
Class Functions:
    forward: compute probability p_{bi} for all b, i.
    backward: compute the derivative w.r.t input matrix a 
'''


class SoftMax:
    def __init__(self, a):
        self.a = a
        self.grad = None if a.grad is None else DT(0)
        self.value = None

    def forward(self):
        # Subtracting max to avoid numerical instability
        exp_a = np.exp(self.a.value - np.max(self.a.value))
        self.value = exp_a / np.sum(exp_a, axis=0)

    def backward(self):
        if self.a.grad is not None:
            # Compute gradient with respect to x
            self.a.grad += self.value * self.grad - \
                np.sum(self.value * self.grad, axis=0) * self.value


'''
Class Name: Log
Class Usage: compute the elementwise log(a) given a.
Class Functions:
    forward: compute log(a)
    backward: compute the derivative w.r.t input vector a
'''


class Log:  # Elementwise Log
    def __init__(self, a):
        self.a = a
        self.grad = None if a.grad is None else DT(0)
        self.value = None

    def forward(self):
        self.value = np.log(self.a.value)

    def backward(self):
        if self.a.grad is not None:
            self.a.grad += self.grad/self.a.value


'''
Class Name: Aref
Class Usage: get some specific entry in a matrix. a is the matrix with shape (batch_size, N) and idx is vector containing 
    the entry index and a is differentiable.
Class Functions:
    forward: compute a[batch_size, idx]
    backward: compute the derivative w.r.t input matrix a
'''


class Aref:
    def __init__(self, a, idx):
        self.a = a
        self.idx = idx
        self.grad = None if a.grad is None else DT(0)

    def forward(self):
        xflat = self.a.value.reshape(-1)
        iflat = self.idx.value.reshape(-1)
        outer_dim = len(iflat)
        inner_dim = len(xflat) / outer_dim
        self.pick = np.int32(np.array(range(outer_dim)) * inner_dim + iflat)
        self.value = xflat[self.pick].reshape(self.idx.value.shape)

    def backward(self):
        if self.a.grad is not None:
            grad = np.zeros_like(self.a.value)
            gflat = grad.reshape(-1)
            gflat[self.pick] = self.grad.reshape(-1)
            self.a.grad = self.a.grad + grad


'''
Class Name: Accuracy
Class Usage: check the predicted label is correct or not. a is the probability vector where each probability is 
            for each class. idx is ground truth label.
Class Functions:
    forward: find the label that has maximum probability and compare it with the ground truth label.
    backward: None 
'''


class Accuracy:
    def __init__(self, a, idx):
        self.a = a
        self.idx = idx
        self.grad = None
        self.value = None

    def forward(self):
        self.value = np.mean(
            np.argmax(self.a.value, axis=-1) == self.idx.value)

    def backward(self):
        pass


# Set of allowed/implemented activation functions
ACTIVATIONS = {'relu': RELU,
               'sigmoid': Sigmoid}


class NN:
    def __init__(self, nodes_array, activation):
        # assert nodes_array is a list of positive integers
        assert all(isinstance(item, int) and item > 0 for item in nodes_array)
        # assert activation is supported
        assert activation in ACTIVATIONS.keys()
        self.nodes_array = nodes_array
        self.activation = activation
        self.activation_func = ACTIVATIONS[self.activation]
        self.layer_number = len(nodes_array) - 1
        self.weights = []
        # dictionary of trainable parameters
        self.params = {}
        # list of computational graph
        self.components = []
        self.sample_placeholder = Value()
        self.label_placeholder = Value()
        self.pred_placeholder = None
        self.loss_placeholder = None
        self.accy_placeholder = None

    # helper function for creating a unary operation object and add it to the computational graph
    def nn_unary_op(self, op, a):
        unary_op = op(a)
        print(
            f"Append <{unary_op.__class__.__name__}> to the computational graph")
        self.components.append(unary_op)
        return unary_op

    # helper function for creating a binary operation object and add it to the computational graph
    def nn_binary_op(self, op, a, b):
        binary_op = op(a, b)
        print(
            f"Append <{binary_op.__class__.__name__}> to the computational graph")
        self.components.append(binary_op)
        return binary_op

    def set_params_by_dict(self, param_dict: dict):
        """
        :param param_dict: a dict of parameters with parameter names as keys
        """
        # reset params to an empty dict before setting new values
        self.params = {}
        # add Param objects to the dictionary of trainable parameters with names and values
        for name, value in param_dict.items():
            self.params[name] = Param(value)

    def set_weights(self, weights):
        """
        :param weights: a list of tuples (matrices and vectors)
        :return:
        """
        weights = np.array(weights)
        # assert weights have the right shapes
        if len(weights) != self.layer_number:
            raise ValueError(
                f"You should provide weights for {self.layer_number} layers instead of {len(weights)}")
        for i, item in enumerate(weights):
            weight, bias = item
            if weight.shape != (self.nodes_array[i], self.nodes_array[i + 1]):
                raise ValueError(
                    f"The weight for the layer {i} should have shape ({self.nodes_array[i]}, {self.nodes_array[i + 1]}) instead of {weight.shape}")
            if bias.shape != (self.nodes_array[i + 1],):
                raise ValueError(
                    f"The bias for the layer {i} should have shape ({self.nodes_array[i + 1]}, ) instead of {bias.shape}")

        # reset params to empty list before setting new values
        param_dict = {}
        for i, item in enumerate(weights):
            # add Param objects to the list of trainable paramters with specified values
            weight, bias = item
            param_dict[f"weight{i}"] = weight
            param_dict[f"bias{i}"] = bias
        self.set_params_by_dict(param_dict)

    def get_weights(self):
        weights = []
        for i in range(self.layer_number):
            w = self.params[f"weight{i}"].value
            b = self.params[f"bias{i}"].value
            weights.append((w, b))

        return weights

    # init_weights_with_xavier(), uses the popular Xavier initialization for the parameters
    def init_weights_with_xavier(self):
        weights = []
        for i in range(self.layer_number):
            weight = xavier((self.nodes_array[i], self.nodes_array[i+1]), "w")
            bias = xavier((self.nodes_array[i], self.nodes_array[i+1]), "b")
            weights.append((weight, bias))
        self.set_weights(weights)

    def build_computational_graph(self):
        if len(self.params) != self.layer_number*2:
            raise ValueError(
                "Trainable Parameters have not been initialized yet. Call init_weights_with_xavier first.")

        # Reset computational graph to empty list
        self.components = []
        prev_output = self.sample_placeholder

        for i in range(self.layer_number - 1):
            # Compute weighted sum
            weighted_sum = self.nn_binary_op(
                VDot, prev_output, self.params[f"weight{i}"])
            # Add bias
            weighted_sum_with_bias = self.nn_binary_op(
                Add, weighted_sum, self.params[f"bias{i}"])
            # Apply activation function
            activation = self.nn_unary_op(
                self.activation_func, weighted_sum_with_bias)
            prev_output = activation

        i = self.layer_number - 1
        # Apply Softmax activation for output layer
        prev_output = self.nn_binary_op(
            VDot, prev_output, self.params[f"weight{i}"])
        prev_output = self.nn_binary_op(
            Add, prev_output, self.params[f"bias{i}"])
        prev_output = self.nn_unary_op(SoftMax, prev_output)

        return prev_output

    def cross_entropy_loss(self):
        label_prob = self.nn_binary_op(
            Aref, self.pred_placeholder, self.label_placeholder)
        log_prob = self.nn_unary_op(Log, label_prob)
        loss = self.nn_binary_op(Mul, log_prob, Value(-1))
        return loss

    def eval(self, X, y):
        if len(self.components) == 0:
            raise ValueError(
                "Computational graph not built yet. Call build_computational_graph first.")
        accuracy = 0.
        objective = 0.
        for k in range(len(y)):
            self.sample_placeholder.set(X[k])
            self.label_placeholder.set(y[k])
            self.forward()
            accuracy += self.accy_placeholder.value
            objective += self.loss_placeholder.value
        accuracy /= len(y)
        objective /= len(y)
        return accuracy, objective

    def fit(self, X, y, alpha, t):
        """
            Use the cross entropy loss.  The stochastic
            gradient descent should go through the examples in order, so
            that your output is deterministic and can be verified.
        :param X: an (m, n)-shaped numpy input matrix
        :param y: an (m,1)-shaped numpy output
        :param alpha: the learning rate
        :param t: the number of iterations
        :return:
        """
        # create sample and input placeholder
        self.pred_placeholder = self.build_computational_graph()
        self.loss_placeholder = self.cross_entropy_loss()
        self.accy_placeholder = self.nn_binary_op(
            Accuracy, self.pred_placeholder, self.label_placeholder)

        train_loss = []
        train_acc = []
        since = time.time()
        for epoch in range(t):
            for i in range(X.shape[0]):
                for p in self.params.values():
                    p.grad = DT(0)
                self.sample_placeholder.set(X[i])
                self.label_placeholder.set(y[i])
                # Forward pass to compute predictions and loss
                self.forward()
                # Backward pass to compute gradients
                # Pass the loss as an argument
                self.backward(self.loss_placeholder)
                # Update weights using gradient descent
                self.sgd_update_parameter(alpha)

            # evaluate on train set
            avg_acc, avg_loss = self.eval(X, y)
            print("Epoch %d: train loss = %.4f, accy = %.4f, [%.3f secs]" % (
                epoch, avg_loss, avg_acc, time.time()-since))
            train_loss.append(avg_loss)
            train_acc.append(avg_acc)
            since = time.time()

    def forward(self):
        for c in self.components:
            c.forward()

    def backward(self, loss):
        for c in self.components:
            if c.grad is not None:
                c.grad = DT(0)
        loss.grad = np.ones_like(loss.value)
        for c in self.components[::-1]:
            c.backward()

    # Optimization functions
    def sgd_update_parameter(self, lr):
        # update the parameter values in self.params

        for i in range(self.layer_number):
            self.params[f"weight{i}"].value -= lr * \
                self.params[f"weight{i}"].grad
            self.params[f"bias{i}"].value -= lr * self.params[f"bias{i}"].grad

    def gradient_estimate(self, param, epsilon=eps):
        # optional, could be used for debugging
        pass


def test_set_and_get_weights():
    # we will change nodes_array in our test
    nodes_array = [4, 5, 5, 3]
    nn = NN(nodes_array, activation="sigmoid")
    # make sure to have the same datatype as DT
    weights = []
    for i in range(nn.layer_number):
        w = np.random.random((nodes_array[i], nodes_array[i+1])).astype(DT)
        b = np.random.random((nodes_array[i+1],)).astype(DT)
        weights.append((w, b))

    nn.set_weights(weights)
    nn_weights = nn.get_weights()

    for i in range(nn.layer_number):
        weight, bias = weights[i]
        nn_weight, nn_bias = nn_weights[i]
        if not np.array_equal(weight, nn_weight):
            raise AssertionError(
                f"The weight on layer {i} is not consistent.\n Set as {weight}, returned as {nn_weight}")
        if not np.array_equal(bias, nn_bias):
            raise AssertionError(
                f"The bias on layer {i} is not consistent.\n Set as {bias}, returned as {nn_bias}")
    print("Passed the test for set_weights and get_weights.")


def main():
    test_set_and_get_weights()


if __name__ == "__main__":
    main()
