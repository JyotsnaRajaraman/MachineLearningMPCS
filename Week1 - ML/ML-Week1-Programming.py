import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

# Read the data into a pandas dataframe
df = pd.read_csv('./arrhythmia.csv', header=None, na_values="?")

# Replace each missing value with the mode
# The preferred pandas function for finding missing values is isnull()
for i in range(280):
    if df[i].isnull().sum() > 0:
        df.iloc[:, i].fillna(df[i].mode()[0], inplace=True)

# Create a small data set to use for debugging and testing
Xsmall = df.loc[0:9, [0, 1, 2]]
ysmall = df.loc[0:9, 279]


class Node(object):
    def __init__(self):
        self.name = None
        self.node_type = None
        self.predicted_class = None
        self.X = None
        self.test_attribute = None
        self.test_value = None
        self.children = []

    def __repr__(self):
        if self.node_type != 'leaf':
            s = (f"{self.name} Internal node with {self.X.shape[0]} examples, "
                 f"tests attribute {self.test_attribute} at {self.test_value}")

        else:
            s = (f"{self.name} Leaf with {self.X.shape[0]} examples, predicts"
                 f" {self.predicted_class}")
        return s


class DecisionTree(object):

    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        '''
        Fit a tree on data, in which X is a 2-d numpy array
        of inputs, and y is a 1-d numpy array of outputs.
        '''
        self.root = self.recursive_build_tree(
            X, y, curr_depth=0, name='0')

    def recursive_build_tree(self, X, y, curr_depth, name):

        # WRITE YOUR CODE HERE
        # pick best attribute and split
        # create two subtrees and recurse until all leaf/maxdepth reached

        # step1: pick best attribute
        # go through all and find best split -> cal entropy
        # the min entropy is best split and attribute
        while curr_depth < self.max_depth:
            newNode = Node()
            newNode.name = name
            newNode.X = X

            transposed_X = X.T
            transposed_y = y.T
            # check whether leaf
            # or if all attributes match, no further analysis
            if len(np.unique(transposed_y)) == 1 or (X == X[0]).all():
                newNode.node_type = "leaf"
                newNode.predicted_class = np.unique(transposed_y)[0]
                return newNode
            i = 0
            tattribute, minentropy, tsplit = 0, float('inf'), 0
            for attibute in transposed_X:
                a = np.stack((attibute, transposed_y), axis=1)
                a = a[a[:, 0].argsort()]
                prevpair = a[0]
                for pair in a:
                    val = pair[1]
                    prevval = prevpair[1]
                    if val != prevval:
                        split = (prevpair[0]+pair[0])/2
                        # find entropy
                        # [(number of examples in child 1)/(total number of examples)*H of child 1 +
                        # (number of examples in child 2)/(total number of examples)*H of child 2]
                        child1 = [row[1] for row in a if row[0] <= split]
                        len1 = len(child1)
                        child2 = [row[1] for row in a if row[0] > split]
                        len2 = len(child2)
                        entropy1 = self.entropy(child1)
                        entropy2 = self.entropy(child2)
                        overallentropy = (
                            (len1/(len1+len2)) * entropy1) + ((len2/(len1+len2)) * entropy2)
                        if overallentropy < minentropy:
                            minentropy = overallentropy
                            tattribute = i
                            tsplit = split
                    prevpair = pair
                i += 1
            newNode.test_attribute = tattribute
            newNode.test_value = tsplit
            # child1
            name1 = name+".0"
            newdepth = curr_depth + 1
            y = y.reshape(len(y), 1)
            vals = np.concatenate((X, y), axis=1)
            # split vals into 2 arrays based on tattribute and tsplit
            child1 = vals[vals[:, tattribute] <= tsplit]
            child2 = vals[vals[:, tattribute] > tsplit]
            newNode.children.append(self.recursive_build_tree(
                child1[:, :-1], child1[:, -1], newdepth, name1))
            # child2
            name2 = name+".1"
            newNode.children.append(self.recursive_build_tree(
                child2[:, :-1], child2[:, -1], newdepth, name2))
            return newNode

        if curr_depth == self.max_depth:
            newNode = Node()
            newNode.name = name
            newNode.X = X
            newNode.node_type = "leaf"
            newNode.predicted_class = scipy.stats.mode(y)[0][0]
            return newNode

        pass

    def predict(self, testset):

        # WRITE YOUR CODE HERE
        predictions = []
        for data in testset:
            root = self.root
            X = data[:-1]
            y = data[-1]
            transposed_X = X.T
            transposed_y = y.T
            predictions.append(self.traversetree(transposed_X, root))
        return predictions
        pass

    def traversetree(self, data, root):
        while root:
            if root.node_type == "leaf":
                return root.predicted_class
            else:
                attribute = root.test_attribute
                split = root.test_value
                if data[attribute] <= split:
                    root = root.children[0]
                else:
                    root = root.children[1]
                self.traversetree(data, root)

    def print(self):
        self.recursive_print(self.root)

    def recursive_print(self, node):
        print(node)
        for u in node.children:
            self.recursive_print(u)

    def entropy(self, y):
        'Return the information entropy in 1-d array y'

        _, counts = np.unique(y, return_counts=True)
        probs = counts/counts.sum()
        return -(np.log2(probs) * probs).sum()


tree = tree = DecisionTree(3)
tree.fit(Xsmall.values, ysmall.values)
tree.print()


# For above you should get the following
# 0 Internal node with 10 examples, tests attribute 0 at 55.5
# 0.0 Internal node with 7 examples, tests attribute 0 at 26.5
# 0.0.0 Leaf with 1 examples, predicts 14
# 0.0.1 Internal node with 6 examples, tests attribute 0 at 49.5
# 0.0.1.0 Leaf with 3 examples, predicts 1
# 0.0.1.1 Leaf with 3 examples, predicts 10
# 0.1 Internal node with 3 examples, tests attribute 0 at 65.5
# 0.1.0 Leaf with 1 examples, predicts 6
# 0.1.1 Leaf with 2 examples, predicts 7


# Limiting the data for quicker output
df = df[:100]
df


def validation_curve_accuracy():

    # step1 - shuffle the data and divide into three

    df1, df2, df3 = np.split(df.sample(frac=1, random_state=42), [
                             int(.34*len(df)), int(.67*len(df))])

    overalltrainingaccuracy = []
    overalltestaccuracy = []
    # df1 and df2 = training
    training = np.concatenate((df1, df2), axis=0)
    test = df3.values
    trainingaccuracy = []
    testaccuracy = []
    for i in range(2, 17, 2):
        tree = DecisionTree(i)
        X = training[:, :-1]
        y = training[:, 279]
        tree.fit(X, y)
        predictions = tree.predict(training)
        actual = y.T
        trainingaccuracy.append(accuracy(predictions, actual))
        X = test[:, :-1]
        y = test[:, 279]
        predictions = tree.predict(test)
        actual = y.T
        testaccuracy.append(accuracy(predictions, actual))
    overalltrainingaccuracy.append(trainingaccuracy)
    overalltestaccuracy.append(testaccuracy)
    # df1 and df3 = training
    training = np.concatenate((df1, df3), axis=0)
    test = df2.values
    trainingaccuracy = []
    testaccuracy = []
    for i in range(2, 17, 2):
        tree = DecisionTree(i)
        X = training[:, :-1]
        y = training[:, 279]
        tree.fit(X, y)
        predictions = tree.predict(training)
        actual = y.T
        trainingaccuracy.append(accuracy(predictions, actual))
        X = test[:, :-1]
        y = test[:, 279]
        predictions = tree.predict(test)
        actual = y.T
        testaccuracy.append(accuracy(predictions, actual))
    overalltrainingaccuracy.append(trainingaccuracy)
    overalltestaccuracy.append(testaccuracy)
    # df3 and df2 = training
    training = np.concatenate((df3, df2), axis=0)
    test = df1.values
    trainingaccuracy = []
    testaccuracy = []
    for i in range(2, 17, 2):
        tree = DecisionTree(i)
        X = training[:, :-1]
        y = training[:, 279]
        tree.fit(X, y)
        predictions = tree.predict(training)
        actual = y.T
        trainingaccuracy.append(accuracy(predictions, actual))
        X = test[:, :-1]
        y = test[:, 279]
        predictions = tree.predict(test)
        actual = y.T
        testaccuracy.append(accuracy(predictions, actual))
    overalltrainingaccuracy.append(trainingaccuracy)
    overalltestaccuracy.append(testaccuracy)
    return (np.array(overalltrainingaccuracy), np.array(overalltestaccuracy))


def accuracy(predictions, actual):
    a = np.isin(predictions, actual)
    hits = 0
    misses = 0
    unique, counts = np.unique(a, return_counts=True)
    check = dict(zip(unique, counts))
    if check.get(True) != None:
        hits = check[True]
    if check.get(False) != None:
        misses = check[False]
    return (hits/(hits+misses))


def validation_curve():
    training_accuracy, test_accuracy = validation_curve_accuracy()
    avgtraining = np.average(training_accuracy.reshape(-1, 8), axis=0)
    avgtest = np.average(test_accuracy.reshape(-1, 8), axis=0)
    plt.plot([2, 4, 6, 8, 10, 12, 14, 16], avgtraining, label="Training Data")
    plt.plot([2, 4, 6, 8, 10, 12, 14, 16], avgtest, label="Test Data")
    plt.legend()
    plt.title("Validation curve")
    plt.xlabel("Depth of Decision Tree")
    plt.ylabel("Accuracy")
    plt.savefig('validationcurve.png')
    plt.show()


validation_curve()
