import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Read the data into a pandas dataframe
train = pd.read_csv(
    './train.csv', index_col='Id')


# ### The [data description section](https://www.kaggle.com/competitions/prudential-life-insurance-assessment/data) on Kaggle gives us the following info:
#
# ##### The following variables are all categorical (nominal):
#
# Product_Info_1, Product_Info_2, Product_Info_3, Product_Info_5, Product_Info_6, Product_Info_7, Employment_Info_2, Employment_Info_3, Employment_Info_5, InsuredInfo_1, InsuredInfo_2, InsuredInfo_3, InsuredInfo_4, InsuredInfo_5, InsuredInfo_6, InsuredInfo_7, Insurance_History_1, Insurance_History_2, Insurance_History_3, Insurance_History_4, Insurance_History_7, Insurance_History_8, Insurance_History_9, Family_Hist_1, Medical_History_2, Medical_History_3, Medical_History_4, Medical_History_5, Medical_History_6, Medical_History_7, Medical_History_8, Medical_History_9, Medical_History_11, Medical_History_12, Medical_History_13, Medical_History_14, Medical_History_16, Medical_History_17, Medical_History_18, Medical_History_19, Medical_History_20, Medical_History_21, Medical_History_22, Medical_History_23, Medical_History_25, Medical_History_26, Medical_History_27, Medical_History_28, Medical_History_29, Medical_History_30, Medical_History_31, Medical_History_33, Medical_History_34, Medical_History_35, Medical_History_36, Medical_History_37, Medical_History_38, Medical_History_39, Medical_History_40, Medical_History_41
#
# ##### The following variables are continuous:
#
# Product_Info_4, Ins_Age, Ht, Wt, BMI, Employment_Info_1, Employment_Info_4, Employment_Info_6, Insurance_History_5, Family_Hist_2, Family_Hist_3, Family_Hist_4, Family_Hist_5
#
# ##### The following variables are discrete:
#
# Medical_History_1, Medical_History_10, Medical_History_15, Medical_History_24, Medical_History_32
#
# Medical_Keyword_1-48 are dummy variables.

print(train.dtypes.unique())


# checking for null
print(train.isnull().sum()[train.isnull().sum() != 0])


# creating boolean series to figure out which rows have missing values for each of the above columns
print(train["Employment_Info_1"].isna())
print(train["Employment_Info_4"].isna())
print(train["Employment_Info_6"].isna())
print(train["Insurance_History_5"].isna())
print(train["Family_Hist_2"].isna())
print(train["Family_Hist_3"].isna())
print(train["Family_Hist_4"].isna())
print(train["Family_Hist_5"].isna())
print(train["Medical_History_1"].isna())
print(train["Medical_History_10"].isna())
print(train["Medical_History_15"].isna())
print(train["Medical_History_24"].isna())
print(train["Medical_History_32"].isna())

# ##### This could be a useful feature, especially for smaller datasets because it may allow us to eliminate rows of data with little to no useful  information. However in much larger datasets, it may not be worth the effort in the big picture
# print(train.keys())


# ###### Discussion
#
# For continous columns we opt to fill with the mean value because we can assume that the value that we do not know is the average of current values to avoid creating any bias for expected value within the data set
#
# For categorical/discrete data types we will choose the most frequent occurence because we do not want to pick the 'mean' as it may not be a category which would introduce error into ur dataset. Using the mode/most frequent data allows us to avoid introducing error/bias into the data

# according to the information we know, the following cols are continous and have values missing
# we will use the mean to fill the missing values
continous = ['Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6',
             'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5']

meanimputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
train.Employment_Info_1 = meanimputer.fit_transform(
    train['Employment_Info_1'].values.reshape(-1, 1))[:, 0]
train.Employment_Info_4 = meanimputer.fit_transform(
    train['Employment_Info_4'].values.reshape(-1, 1))[:, 0]
train.Employment_Info_6 = meanimputer.fit_transform(
    train['Employment_Info_6'].values.reshape(-1, 1))[:, 0]
train.Insurance_History_5 = meanimputer.fit_transform(
    train['Insurance_History_5'].values.reshape(-1, 1))[:, 0]
train.Family_Hist_2 = meanimputer.fit_transform(
    train['Family_Hist_2'].values.reshape(-1, 1))[:, 0]
train.Family_Hist_3 = meanimputer.fit_transform(
    train['Family_Hist_3'].values.reshape(-1, 1))[:, 0]
train.Family_Hist_4 = meanimputer.fit_transform(
    train['Family_Hist_4'].values.reshape(-1, 1))[:, 0]
train.Family_Hist_5 = meanimputer.fit_transform(
    train['Family_Hist_5'].values.reshape(-1, 1))[:, 0]


# checking for null
print(train.isnull().sum()[train.isnull().sum() != 0])

# according to the information we know, the following cols are categorical and have values missing
# we will use the most frequent value (mode) to fill the missing values
categorical = ['Medical_History_1', 'Medical_History_10',
               'Medical_History_15', 'Medical_History_24', 'Medical_History_32']

frequencyimputer = SimpleImputer(
    missing_values=np.NaN, strategy='most_frequent')
train.Medical_History_1 = frequencyimputer.fit_transform(
    train['Medical_History_1'].values.reshape(-1, 1))[:, 0]
train.Medical_History_10 = frequencyimputer.fit_transform(
    train['Medical_History_10'].values.reshape(-1, 1))[:, 0]
train.Medical_History_15 = frequencyimputer.fit_transform(
    train['Medical_History_15'].values.reshape(-1, 1))[:, 0]
train.Medical_History_24 = frequencyimputer.fit_transform(
    train['Medical_History_24'].values.reshape(-1, 1))[:, 0]
train.Medical_History_32 = frequencyimputer.fit_transform(
    train['Medical_History_32'].values.reshape(-1, 1))[:, 0]

# checking for null to make sure no columns has any more null values left
print(train.isnull().sum()[train.isnull().sum() != 0])

# print(train.head())


train = pd.get_dummies(train, columns=['Product_Info_1', 'Product_Info_2', 'Product_Info_3', 'Product_Info_5', 'Product_Info_6', 'Product_Info_7', 'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5', 'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3',
                       'InsuredInfo_4', 'InsuredInfo_5', 'InsuredInfo_6', 'InsuredInfo_7', 'Insurance_History_1', 'Insurance_History_2', 'Insurance_History_3', 'Insurance_History_4', 'Insurance_History_7', 'Insurance_History_8', 'Insurance_History_9', 'Family_Hist_1'], drop_first=True)


data = train.sample(n=10000, replace=True, random_state=5)
# print(data.head())


X = data[data.columns[:-1]]
X.values.dtype


y = data.Response
y.head()


maxdepthvals = np.arange(1, 10)


train_scores, test_scores = validation_curve(DecisionTreeClassifier(
), X.values, y.values, param_name='max_depth', param_range=maxdepthvals)


print(train_scores)


train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)


plt.subplots(1, figsize=(7, 7))
plt.plot(maxdepthvals, train_scores_mean-train_scores_std, color="red")
plt.title("Validation Curve with DecisionTreeClassifier")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.show()


plt.subplots(1, figsize=(7, 7))
# plt.plot(maxdepthvals, train_scores_mean-train_scores_std, train_scores_mean+train_scores_std,color="red")
plt.plot(maxdepthvals, test_scores_mean-test_scores_std, color="blue")
plt.title("Validation Curve with DecisionTreeClassifier")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.show()


tree = DecisionTreeClassifier(max_depth=7)
train_size_abs, train_scores, test_scores = learning_curve(
    tree, X.values, y.values, train_sizes=[0.3, 0.6, 0.9])


train_scores_mean7 = -train_scores.mean(axis=1)
test_scores_mean7 = -test_scores.mean(axis=1)


tree = DecisionTreeClassifier(max_depth=6)
train_size_abs, train_scores, test_scores = learning_curve(
    tree, X.values, y.values, train_sizes=[0.3, 0.6, 0.9])
train_scores_mean5 = -train_scores.mean(axis=1)
test_scores_mean5 = -test_scores.mean(axis=1)


# plt.style.use('seaborn')
plt.plot(train_size_abs, train_scores_mean5,
         label='Training error for depth 5')
plt.plot(train_size_abs, test_scores_mean5,
         label='Validation error for depth 5')
plt.ylabel('Error', fontsize=14)
plt.xlabel('Training set size', fontsize=14)
plt.title('Learning curves for Decision Tree Classifier')
plt.legend()

plt.show()

# plt.style.use('seaborn')
plt.plot(train_size_abs, train_scores_mean7,
         label='Training error for depth 7')
plt.plot(train_size_abs, test_scores_mean7,
         label='Validation error for depth 7')
plt.ylabel('Error', fontsize=14)
plt.xlabel('Training set size', fontsize=14)
plt.title('Learning curves for Decision Tree Classifier')
plt.legend()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.3, random_state=1,
                                                    stratify=data.iloc[:, -1])

pipeline = make_pipeline(StandardScaler(), LogisticRegression(
    solver='lbfgs', penalty='l2', max_iter=10000, random_state=1))

param_range = np.arange(1, 1000, 100)
train_scores, test_scores = validation_curve(estimator=pipeline,
                                             X=X_train, y=y_train,
                                             cv=10,
                                             param_name='logisticregression__C', param_range=param_range)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
#
# Plot the model scores (accuracy) against the paramater range
#
plt.plot(param_range, train_mean,
         marker='o', markersize=5,
         color='blue', label='Training Accuracy')
plt.plot(param_range, test_mean,
         marker='o', markersize=5,
         color='green', label='Validation Accuracy')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

train_size_abs, train_scores, test_scores = learning_curve(
    pipeline, X.values, y.values, train_sizes=[0.3, 0.6, 0.9])

train_scores_mean = -train_scores.mean(axis=1)
test_scores_mean = -test_scores.mean(axis=1)

plt.style.use('seaborn')
plt.plot(train_size_abs, train_scores_mean, label='Training error')
plt.plot(train_size_abs, test_scores_mean, label='Validation error')
plt.ylabel('Error', fontsize=14)
plt.xlabel('Training set size', fontsize=14)
plt.title('Learning curves for Logical Regression')
plt.legend()
