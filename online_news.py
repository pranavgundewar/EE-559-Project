# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 19:56:13 2018

@author: tchat
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as prep
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

#%%
# Import data and split input and target features
data = pd.read_csv('OnlineNewsPopularityReduced.csv', skipinitialspace = True)
data_Y = data['shares']
data_X = data.drop(['url', 'timedelta', 'shares'], axis=1)

#%%
# Binarizing Target Feature by Median
perc_95 = np.percentile(data_Y.values, 95)
perc_5 = np.percentile(data_Y.values, 5)

print('\nThe median of target feature is: %.2f' %data_Y.median())

# Feature Scaling and Outlier Counting
def outlierCounts(series):
    centered = np.abs(series - series.mean())
    mask     = centered >= (5 * series.std())
    return len(series[mask])

def uniqueValueCount(series):
    return len(series.unique())

input_feats = data_X.dtypes.reset_index()
input_feats.columns  = ['name', 'dtype']
input_feats['mean']  = data_X.mean().reset_index(drop=True)
input_feats['std']   = data_X.std().reset_index(drop=True)
input_feats['range'] = (data_X.max() - data_X.min()).reset_index(drop=True)
input_feats['unique_values_count'] = data_X.apply(uniqueValueCount, axis=0).reset_index(drop=True)
input_feats['outliers_count'] = data_X.apply(outlierCounts, axis=0).reset_index(drop=True)

# Merge Binary Features
def MergeFeatures(data, old_features, new_feature):

    counter = 0
    data[new_feature] = counter

    for old_feature in old_features:
        counter = counter + 1
        data.loc[data[old_feature] == 1, new_feature] = counter
        del data[old_feature]

    return data

data_channels = [
    'data_channel_is_lifestyle',
    'data_channel_is_entertainment',
    'data_channel_is_bus',
    'data_channel_is_socmed',
    'data_channel_is_tech',
    'data_channel_is_world'
]

weekdays = [
    'weekday_is_monday',
    'weekday_is_tuesday',
    'weekday_is_wednesday',
    'weekday_is_thursday',
    'weekday_is_friday',
    'weekday_is_saturday',
    'weekday_is_sunday'
]

data = MergeFeatures(data_X, data_channels, 'data_channel')
data = MergeFeatures(data_X, weekdays, 'pub_weekday')

# Remove Outliers
for col in data_X.columns:
    centered = np.abs(data_X[col]-data_X[col].mean())
    mask = centered <= (5 * data_X[col].std())
    data_X = data_X[mask]

data_Y = data_Y[data_X.index]

# Standardize the Data
def standarize(arr_X):
    arr_X = prep.MinMaxScaler().fit_transform(arr_X)
    return arr_X - arr_X.mean(axis=1).reshape(-1, 1)

arr_X = data_X.values
arr_X = standarize(arr_X)

# Binarize the Target Feature
arr_Y = prep.binarize(data_Y.values.reshape(-1, 1), threshold=1400)
data_Y  = pd.Series(arr_Y.ravel())

#unique_items, counts = np.unique(arr_Y, return_counts=True)
#print('size of class 0: {0}'.format(counts[0]))
#print('size of class 1: {0}'.format(counts[1]))

# Split data set
X_train, X_test, y_train, y_test = train_test_split(arr_X, arr_Y, test_size=0.10)

# Apply PCA
pca = PCA().fit(X_train)

X_train_pca = pca.transform(X_train)
X_train_pca = standarize(X_train_pca)
X_test_pca  = pca.transform(X_test)
X_test_pca  = standarize(X_test_pca)

'''
# SVM Classifier
skf = StratifiedKFold(n_splits = 5, shuffle = True)
Cs = np.logspace(-3, 3, 10)
gammas = np.logspace(-3, 3, 10)

ACC = np.zeros((10,10))
DEV = np.zeros((10,10))

for i, gamma in enumerate(gammas):
    for j, C in enumerate(Cs):
        acc = []
        for train_index, dev_index in skf.split(X_train_pca, y_train):
            X_cv_train, X_cv_dev = X_train_pca[train_index], X_train_pca[dev_index]
            y_cv_train, y_cv_dev = y_train[train_index], y_train[dev_index]
            clf = SVC(C = C, kernel = 'rbf', gamma = gamma, )
            clf.fit(X_cv_train, y_cv_train)
            acc.append(accuracy_score(y_cv_dev, clf.predict(X_cv_dev)))

        ACC[i,j] = np.mean(acc)
        DEV[i,j] = np.std(acc)

i, j = np.argwhere(ACC == np.max(ACC))[0]
print('The best pair is C = ' + str(Cs[j]) + ' and gamma = ' + str(gammas[i]))
print('The mean Cross-Validation Accuracy for the best pair = ', ACC[i,j])
print('The Standard deviation for the best pair = ', DEV[i,j])
print('')

clf1 = SVC(C = Cs[j], kernel = 'rbf', gamma = gammas[i], decision_function_shape = 'ovr')
clf1.fit(X_train_pca, y_train)
y_pred = clf1.predict(X_test_pca)
acc = accuracy_score(y_test, y_pred)
print('Test accuracy  = ', acc)
'''

# Naive Bayes
skf = StratifiedKFold(n_splits = 5, shuffle = True)

acc = []
for train_index, dev_index in skf.split(X_train_pca, y_train):
    X_cv_train, X_cv_dev = X_train_pca[train_index], X_train_pca[dev_index]
    y_cv_train, y_cv_dev = y_train[train_index], y_train[dev_index]
    clf = GaussianNB()
    clf.fit(X_cv_train, y_cv_train)
    acc.append(accuracy_score(y_cv_dev, clf.predict(X_cv_dev)))

print('The best Cross-Validation Accuracy (PCA) is = ', max(acc))

acc = []
for train_index, dev_index in skf.split(X_train, y_train):
    X_cv_train, X_cv_dev = X_train[train_index], X_train[dev_index]
    y_cv_train, y_cv_dev = y_train[train_index], y_train[dev_index]
    clf = GaussianNB()
    clf.fit(X_cv_train, y_cv_train)
    acc.append(accuracy_score(y_cv_dev, clf.predict(X_cv_dev)))

print('The best Cross-Validation Accuracy is = ', max(acc))

clf1 = GaussianNB()
clf1.fit(X_train_pca, y_train)
y_pred = clf1.predict(X_test_pca)
acc = accuracy_score(y_test, y_pred)
print('Test accuracy  = ', acc)

clf1 = GaussianNB()
clf1.fit(X_train, y_train)
y_pred = clf1.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('Test accuracy  = ', acc)


# kNN Classifier
'''
skf = StratifiedKFold(n_splits = 5, shuffle = True)

acc = []
for train_index, dev_index in skf.split(X_train_pca, y_train):
    X_cv_train, X_cv_dev = X_train_pca[train_index], X_train_pca[dev_index]
    y_cv_train, y_cv_dev = y_train[train_index], y_train[dev_index]
    clf = KNeighborsClassifier(n_neighbors = 2)
    clf.fit(X_cv_train, y_cv_train)
    acc.append(accuracy_score(y_cv_dev, clf.predict(X_cv_dev)))

print('The best Cross-Validation Accuracy (PCA) is = ', max(acc))

acc = []
for train_index, dev_index in skf.split(X_train, y_train):
    X_cv_train, X_cv_dev = X_train[train_index], X_train[dev_index]
    y_cv_train, y_cv_dev = y_train[train_index], y_train[dev_index]
    clf = KNeighborsClassifier(n_neighbors = 2)
    clf.fit(X_cv_train, y_cv_train)
    acc.append(accuracy_score(y_cv_dev, clf.predict(X_cv_dev)))

print('The best Cross-Validation Accuracy is = ', max(acc))
'''

clf1 = KNeighborsClassifier(n_neighbors = 5)
clf1.fit(X_train_pca, y_train)
y_pred = clf1.predict(X_test_pca)
acc = accuracy_score(y_test, y_pred)
print('Test accuracy  = ', acc)

clf1 = KNeighborsClassifier(n_neighbors = 3)
clf1.fit(X_train, y_train)
y_pred = clf1.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('Test accuracy  = ', acc)
