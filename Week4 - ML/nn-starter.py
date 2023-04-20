import nn_utils as nn  # to uncomment
import pickle
import numpy as np
import pandas as pd
import importlib
import time
from uu import Error
from IPython.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))
importlib.reload(nn)


# ### Brief overview of your tasks in the accompanying Python script.  Please also see the outline in the homework instructions.
# - You need to complete the neural network by fill in codes where there is a #todo prompt.
# - The architecture of the neural network you are building is a vanilla version of that of Pytorch/Tensorflow consisting of nodes of *Operation Classes*. Each *Operation Class* has a forward method (to calculate the operation result stored in self.value) and a backward method (to calculate the gradient of the operation w.r.t. the final loss function).
# - You need to implement forward and backward methods for *Operation Classes* Mul, VDot, Sigmoid, Relu (not required), Softmax, Log. Add, Aref, and Accuracy has been implemented for you for reference
#
#
# - self.params is the list to store the trainable parameters (objects of Class Param).
# - set_weights(weights) has been implemented for you where the provided weights and biases are converted to Param objects and stored in self.params. You need to study the code and implement <code>get_weights()</code> and <code>init_weights_with_xavier()</code>
# - self.components is the list to natively mimic the function of the computational graph. Helper functions nn_unary_op(op, x) and nn_binary_op(op, x, y) are provided to facilitate creating an operation and adding it to the computational graph. For example, instead of $a=b+c$, you should use <code>a = self.nn_binary_op(Add, Value(b), Value(c))</code>. Only in this way you can create an operation object <code>a</code> with <code>a.value</code> and <code>a.grad</code> to support the forward and backward method of the neural network.
# - Placeholder including <code>self.sample_placeholder</code>, <code>self.label_placeholder</code>, <code>self.pred_placeholder</code>, <code>self.loss_placeholder</code>, <code>self.accy_placeholder</code> are all empty vectors that will be assigned values when executing <code>forward</code> or <code>backward</code>. They facilitate the construcation of the computational graph. <code>self.sample_placeholder</code> is the input to the NN. We feed different examples by calling <code>self.sample_placeholder.set(X[i])</code> and <code>self.label_placeholder.set(y[i])</code>. <code>self.pred_placeholder</code>, <code>self.loss_placeholder</code>, <code>self.accy_placeholder</code> changes values in each iteration in <code>fit</code>.
# - <code>self.forward()</code> is provided for you where each operation object in self.components are evalued from the begining to the end.
# - <code>self.backward()</code> is provided for you where derivative of each operation object in self.components are calculated from the end (loss function) to the beginning.
# - You need to implement sgd_update_parameter.
# - You could implement gradient_estimate to debug.
# - A test function test_set_and_get_weights() has been provided for you to test <code>self.get_weights</code>. Feel free to create more test functions.

# ### Below is the test code for get_weights

# You may get a decprecation warning --- ignore it.
nn.test_set_and_get_weights()


m1 = nn.Value(np.arange(12).reshape((3, 4)))
m2 = nn.Value(np.arange(12, 24).reshape((3, 4)))
m3 = nn.Value(np.arange(24, 36).reshape((3, 4)))
m4 = nn.Value(np.arange(36, 48).reshape((3, 4)))

v1 = nn.Value(np.arange(3).reshape((3,)))
v2 = nn.Value(np.arange(3, 6).reshape((3,)))
v3 = nn.Value(np.arange(6, 9).reshape((3,)))
v4 = nn.Value(np.arange(4).reshape((4,)))

# Test VDot
importlib.reload(nn)

x = nn.Mul(m1, m2)
y = nn.VDot(v1, x)
z = nn.Mul(y, v4)
x.forward()
y.forward()
z.forward()
z.grad = np.ones_like(z.value)
y.grad = 0
x.grad = 0
v1.grad = 0
z.backward()
y.backward()
x.backward()

yvalue = np.array([384., 463., 548., 639.])
ygrad = np.array([0., 1., 2., 3.])
yagrad = np.array([204.,  700., 1388.])
ybgrad = np.array([[0., 0., 0., 0.],
                   [0., 1., 2., 3.],
                   [0., 2., 4., 6.]])

if not np.array_equal(y.value, yvalue):
    raise Error("y.value not equal to matrix product of x.value and v1.value")
if not np.array_equal(y.grad, ygrad):
    raise Error("gradient of y is incorrect")
if not np.array_equal(y.a.grad, yagrad):
    raise Error("gradient of a in y is incorrect")
if not np.array_equal(y.b.grad, ybgrad):
    raise Error("gradient of b in y is incorrect")
print("Passed Test on VDot")


# Test Sigmoid
x = nn.Add(v1, v2)
y = nn.Sigmoid(x)
z = nn.Mul(y, v3)
x.forward()
y.forward()
z.forward()
z.grad = np.ones_like(z.value)
y.grad = 0
x.grad = 0
z.backward()
y.backward()
x.backward()

yvalue = np.array([0.95257413, 0.9933072, 0.999089], dtype=np.float32)
yagrad = np.array([0.2710599, 0.04653623, 0.00728134], dtype=np.float32)

if not np.array_equal(np.round(y.value, 5), np.round(yvalue, 5)):
    raise Error("y.value not equal to sigmoid of x")
if not np.array_equal(np.round(y.a.grad, 5), np.round(yagrad, 5)):
    raise Error("gradient of a in y is incorrect")
print("Passed Test on Sigmoid")


# Test RELU
# THIS IS OPTIONAL.
x = nn.Add(v1, v2)
y = nn.RELU(x)
z = nn.Mul(y, v3)
x.forward()
y.forward()
z.forward()
z.grad = np.ones_like(z.value)
y.grad = 0
x.grad = 0
z.backward()
y.backward()
x.backward()

yvalue = np.array([3., 5., 7.], dtype=np.float32)
yagrad = np.array([6., 7., 8.], dtype=np.float32)

# print("Passing this teset is optional.")

if not np.array_equal(np.round(y.value, 5), np.round(yvalue, 5)):
    raise Error("y.value not equal to Relu of x")
if not np.array_equal(np.round(y.a.grad, 5), np.round(yagrad, 5)):
    raise Error("gradient of a in y is incorrect")
print("Passed Test on RELU")


# Test SoftMax

x = nn.Add(v1, v2)
y = nn.SoftMax(x)
z = nn.Mul(y, v3)
x.forward()
y.forward()
z.forward()
z.grad = np.ones_like(z.value)
y.grad = 0
x.grad = 0
z.backward()
y.backward()
x.backward()

yvalue = np.array([0.01587624, 0.11731043, 0.86681336], dtype=np.float32)
yagrad = np.array([-0.02938593, -0.09982383,  0.12920949], dtype=np.float32)

if not np.array_equal(np.round(y.value, 5), np.round(yvalue, 5)):
    raise Error("y.value not equal to SoftMax of x")
if not np.array_equal(np.round(y.a.grad, 5), np.round(yagrad, 5)):
    raise Error("gradient of a in y is incorrect")
print("Passed Test on SoftMax")


# Test Log
x = nn.Add(v1, v2)
y = nn.Log(x)
z = nn.Mul(y, v3)
x.forward()
y.forward()
z.forward()
z.grad = np.ones_like(z.value)
y.grad = 0
x.grad = 0
z.backward()
y.backward()
x.backward()

yvalue = np.array([1.0986123, 1.609438, 1.9459102], dtype=np.float32)
yagrad = np.array([2., 1.4, 1.1428572], dtype=np.float32)

if not np.array_equal(np.round(y.value, 5), np.round(yvalue, 5)):
    raise Error("y.value not equal to log of x")
if not np.array_equal(np.round(y.a.grad, 5), np.round(yagrad, 5)):
    raise Error("gradient of a in y is incorrect")
print("Passed Test on Log")


# ## Applying to the MNIST dataset
# - You should use the MNIST dataset to test the neural network you build above.


# we will use train.csv for training and testing as it test.csv doesn't contain label
data = pd.read_csv("./mnist_data/train.csv")
train_data = data.iloc[:30000]  # 30000
test_data = data.iloc[30000:]  # 12000

pixel_columns = [f"pixel{i}" for i in range(784)]

# normalize by dividing by 255 as the pixel ranges from 0 to 255
train_x = train_data[pixel_columns].values.astype(nn.DT)/255
train_y = train_data["label"].values.astype(nn.DT)

test_x = test_data[pixel_columns].values.astype(nn.DT)/255
test_y = test_data["label"].values.astype(nn.DT)


# ### Debugging the fit function

# important line so that the changes you made on p2.py will be reflected without restarting the kernel
importlib.reload(nn)
# Here we use the 2-layer NN with 1 hidden layer, feel free to experiment on your own
nodes_array = [784, 128, 10]
model = nn.NN(nodes_array, "sigmoid")


class FailTestError(Exception):
    'Raised when a test fails'
    pass


# You can use the provided sample weights for initialization to help debug
with open("./mnist_data/sample_weights.pkl", 'rb') as f:
    weights = pickle.load(f)
model.set_weights(weights)

# You can use the first 2 samples to test if the gradients are correct
X = train_x[:2]
y = train_y[:2]

# when calling fit, a computational graph will be built first, you should expect the exact lines printed
model.fit(X, y, alpha=0.01, t=1)

# Load the sample gradient for debugging
with open("./mnist_data/sample_gradient.pkl", 'rb') as f:
    sample_grad = pickle.load(f)

# print(np.allclose(model.params["weight0"].grad, sample_grad["w1"]))
# print(sample_grad["w1"])
# print(model.params["weight0"].grad)

# print(np.allclose(model.params["bias0"].grad, sample_grad["b1"]))
# print(sample_grad["b1"])
# print(model.params["bias0"].grad)

# first layer's weight of shape (784, 128)
if not np.allclose(model.params["weight0"].grad, sample_grad["w1"]):
    raise FailTestError("gradient of the first layer's weight is incorrect")
# first layer's bias of shape (128, )
if not np.allclose(model.params["bias0"].grad, sample_grad["b1"]):
    raise FailTestError("gradient of the first layer's bias is incorrect")
# second layer's weight of shape (128, 10)
if not np.allclose(model.params["weight1"].grad, sample_grad["w2"]):
    raise FailTestError("gradient of the second layer's weight is incorrect")
# second layer's bias of shape (10, )
if not np.allclose(model.params["bias1"].grad, sample_grad["b2"]):
    raise FailTestError("gradient of the second layer's bias is incorrect")
print("Congrats! You have passed the test of your fit function, your NN model should be good to go!")


# important line so that the changes you made on p2.py will be reflected without restarting the kernel
importlib.reload(nn)
# Here we use the 2-layer NN with 1 hidden layer, feel free to experiment on your own
nodes_array = [784, 128, 10]
model = nn.NN(nodes_array, "sigmoid")
model.init_weights_with_xavier()
model.fit(train_x, train_y, 0.01, 10)

# After 10 epochs of training, you should expect an accuracy over 95% and loss around 0.1
accy, loss = model.eval(test_x, test_y)
print("Test accuracy = %.4f, Loss = %.4f" % (accy, loss))
