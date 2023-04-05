from sklearn.metrics import cohen_kappa_score
import numpy as np
from collections import Counter
import random

# define array of ratings for both classifications
classifier1 = [0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0]
classifier2 = [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0]

# step 1: calculate value of k
k = max(len(set(classifier1)), len(set(classifier2)))

# Construct k*k matrix o, such that the entries oij of O are:
# the observed proportion of observations that are classified as i by the first classifier and j by the second.
O = np.array([[0]*k for _ in range(k)])

for ind in range(len(classifier1)):
    for i in range(k):
        for j in range(k):
            if (classifier1[ind] == i and classifier2[ind] == j):
                O[i][j] += 1
O = O/len(classifier1)


# Construct k*k matrix E, such that the entries eij of E are:
# the proportion of observations that are classified as i by the first classifier * j by the second.
E = np.array([[0]*k for _ in range(k)])

# let us first find out the number of times each classification is made
c1 = Counter(classifier1)
c2 = Counter(classifier2)

for i in range(k):
    for j in range(k):
        c1cout = c1.get(i)
        if c1cout == None:
            c1cout = 0
        c2cout = c2.get(j)
        if c2cout == None:
            c2cout = 0
        E[i][j] = c1cout * c2cout

E = E/(len(classifier1)**2)


# Construct k*k matrix W, such that the entries wij of W are:
# The entries wij of W are the weights, which we define here to be (i - j)^2.
W = np.array([[0]*k for _ in range(k)])

for i in range(k):
    for j in range(k):
        W[i][j] = (i-j)**2

# Sum(wij*oij)
WOSum = 0
WESum = 0

for i in range(k):
    for j in range(k):
        WOSum += W[i][j]*O[i][j]
        WESum += W[i][j]*E[i][j]

kappa = 1 - (WOSum/WESum)
print(kappa)

# calculate Cohen's Kappa
print(cohen_kappa_score(classifier1, classifier2))
