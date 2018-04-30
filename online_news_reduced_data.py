#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:38:44 2018

@author: adityakillekar
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Input, Dense
from keras.models import Model, Sequential
import matplotlib.pyplot as plt
import handle_missing_data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
#import seaborn as sns
from imblearn.combine import SMOTEENN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as m
from sklearn.model_selection import StratifiedShuffleSplit


##%%
## Import data and split input and target features
#data_original = pd.read_csv('OnlineNewsPopularityReduced.csv', skipinitialspace = True)
#
##Data for linear regression
#data_reg_Y = data_original['shares']
#data_reg_X = data_original.drop(['url', 'timedelta', 'kw_max_max', 'shares'], axis=1)
#
##Data for classification algorithms
#data = data_original.drop(['url', 'timedelta', 'kw_max_max'], axis=1)
#
#popular = data['shares'] > 1600
#not_popular = data['shares'] <= 1600
#
#popular_1 = data['shares'] <= 900
#popular_2 = (data['shares'] > 900) & (data['shares'] <= 1200)
#popular_3 = (data['shares'] > 1200) & (data['shares'] <= 1600)
#popular_4 = (data['shares'] > 1600) & (data['shares'] <= 3400)
#popular_5 = data['shares'] > 3400
#
#data.loc[popular,'two_class_popularity'] = 1
#data.loc[not_popular,'two_class_popularity'] = 0
#
#data.loc[popular_1,'multi_class_popularity'] = 1
#data.loc[popular_2,'multi_class_popularity'] = 2
#data.loc[popular_3,'multi_class_popularity'] = 3
#data.loc[popular_4,'multi_class_popularity'] = 4
#data.loc[popular_5,'multi_class_popularity'] = 5
#
#data = data.drop(['shares'], axis=1)
#
#data_multiclass_Y = data['multi_class_popularity']
#data_twoclass_Y = data['two_class_popularity']
#data_X = data.drop(['two_class_popularity','multi_class_popularity'],axis=1)
#
##data.to_csv('ModifiedOnlineNewsPopularityReduced.csv',index=False)

#%%
#Import modified data
data_original = pd.read_csv('ModifiedOnlineNewsPopularityReduced_input.csv', skipinitialspace = True)

#Data for linear regression
data_reg_Y = data_original['shares']
data_twoclass_Y = data_original['two_class_popularity']
data_multiclass_Y = data_original['multi_class_popularity']
data_X = data_original.drop(['shares','two_class_popularity','multi_class_popularity', 'Unnamed: 40'], axis=1)

#%%
def PCA_dim_reduction(X,dim):
    pca = PCA(n_components=dim)
    X_new = pca.fit_transform(X)
    return X_new

#%%
def scale(X):
    scaler = StandardScaler()
    X_new = scaler.fit_transform(X)
    return X_new

#%%
def Non_linear_SVM(X, Y, C=10, gamma=0.01, PCA=False, dim=10, norm=False):
    if norm:
        X = scale(X)
        
    if PCA:
        X = PCA_dim_reduction(X, dim)
    
    X_train, Y_train = shuffle(X, Y)

    k_fold = StratifiedKFold(n_splits=5,shuffle=True)
    all_acc = []
    count = 1
    
    for train_index, dev_index in k_fold.split(X_train,Y_train):
        X_cv_train, X_cv_dev = X_train[train_index], X_train[dev_index]
        Y_cv_train, Y_cv_dev = Y_train[train_index], Y_train[dev_index]
        svm_classifier = SVC(C=C,kernel='rbf',gamma=gamma)
        svm_classifier.fit(X_cv_train, Y_cv_train)
        acc = svm_classifier.score(X_cv_dev,Y_cv_dev)
        all_acc.append(acc)
        print("Accuracy with Split #{0}: {1}".format(count,acc))
        count += 1
    avg_acc = np.mean(all_acc)
    print("Average Accuracy: {0}".format(avg_acc))

#%%
def run_nn(X, Y, hidden_layers=[10,20,30,40,30,20,10]):
    enc = OneHotEncoder()
    print(Y)
    print(Y.reshape(-1, 1))
    Y_enc = enc.fit_transform(Y.reshape(-1, 1))
    print(Y_enc)
    Y_enc = np.reshape(Y_enc, (X.shape[0],Y_enc.shape[1]))
    nn_model = neural_model(input_shape=X.shape[1],output_shape=Y_enc.shape[1], hidden_layer_config=hidden_layers)
    result = run_network(nn_model, X, Y_enc, epochs=20)
    
    print(result.history.keys())

    from keras.utils import plot_model
    plot_model(nn_model, to_file='model.png', show_shapes=True)

    plt.plot(result.history['loss'])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(result.history['val_loss'])
    plt.show()
    
    
    
#%%
def neural_model(input_shape, output_shape, hidden_layer_config):
    input_layer = Input(shape=(input_shape,))
    network_layers = []
    network_layers.append(input_layer)
#    hidden_layers = len(hidden_layer_config)
    for hidden_layer_size in hidden_layer_config:
        new_layer = Dense(
            hidden_layer_size,
            activation='relu',
            kernel_initializer='truncated_normal',
            bias_initializer='zeros',
            kernel_regularizer='l1',
            bias_regularizer='l1'
        )(network_layers[-1])
        network_layers.append(new_layer)
    output_layer = Dense(output_shape, activation='softmax')(network_layers[-1])
    network_layers.append(output_layer)

    network = Model(network_layers[0], network_layers[-1])
    # L1 = Model(network_layers[0], network_layers[1])

    network.compile(
        optimizer='sgd',
        loss='mean_squared_error',
        metrics=['accuracy']
    )

    # return network, L1
    return network

#%%
def run_network(network, X, Y, epochs=50):
    # print(train_test_split(X, Y, test_size=0.3))
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
    for train_index, test_index in sss.split(X, Y):
        X_train, X_vald = X[train_index], X[test_index]
        Y_train, Y_vald = Y[train_index], Y[test_index]
    
#    x_train, x_vald, y_train, y_vald = train_test_split(X, Y, test_size=0.3)

    x_train = np.asarray(X_train)
    y_train = np.asarray(Y_train)
    x_vald = np.asarray(X_vald)
    y_vald = np.asarray(Y_vald)

    model_result = network.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=20,
        shuffle=True,
        validation_data=(x_vald, y_vald)
    )

    return model_result

#%%
Non_linear_SVM(data_X.values, data_twoclass_Y.values, C=100, gamma=0.0001, PCA=True, dim=25, norm =True)
    
#%%
run_nn(data_X.values, data_multiclass_Y.values, hidden_layers=[64,100,100,350,500,350,100,64,20]) 

#%%
classifiers = {'Gradient Boosting Classifier':GradientBoostingClassifier(),'Adaptive Boosting Classifier'
               :AdaBoostClassifier(),'Linear Discriminant Analysis':LinearDiscriminantAnalysis(),
               'Logistic Regression':LogisticRegression(),'Random Forest Classifier': RandomForestClassifier(),
               'K Nearest Neighbour':KNeighborsClassifier(7),'Decision Tree Classifier'
               :DecisionTreeClassifier(),'Gaussian Naive Bayes Classifier':GaussianNB(),
               'Support Vector Classifier':SVC(probability=True, C=1000, gamma=0.001)}

log_cols = ["Classifier", "Accuracy","F1-Score","roc-auc_Score"] #"Precision Score","Recall Score",]
#metrics_cols = []
log = pd.DataFrame(columns=log_cols)

#%%
X_train = data_X
y_train = data_twoclass_Y
rs = StratifiedShuffleSplit(n_splits=5, test_size=0.2,random_state=0)
for Name,classify in classifiers.items():
    accuracy=[]
    precision=[]
    recall=[]
    roc_auc=[]
    f1_score=[]
    print('Training: ', Name)
    for train_index, test_index in rs.split(X_train,y_train):
        #print("TRAIN:", train_index, "TEST:", test_index)
        y,y_test = y_train.iloc[train_index], y_train.iloc[test_index]
        X,X_test = X_train.iloc[train_index], X_train.iloc[test_index]        
        # Scaling of Features 
        sc_X = StandardScaler()
        X = sc_X.fit_transform(X)
        X_test = sc_X.transform(X_test)
        cls = classify
        cls =cls.fit(X,y)
        y_out = cls.predict(X_test)
        accuracy.append(m.accuracy_score(y_test,y_out))
        precision.append(m.precision_score(y_test,y_out,average='weighted'))
        recall.append(m.recall_score(y_test,y_out,average='weighted'))
        roc_auc.append(roc_auc_score(y_out,y_test,average='weighted'))
        f1_score.append(m.f1_score(y_test,y_out,average='weighted'))
    acc = np.max(accuracy)
    p = np.mean(precision)
    re= np.mean(recall)
    roc = np.max(roc_auc)
    f1 = np.max(f1_score)
    log_entry = pd.DataFrame([[Name,acc,f1,roc]], columns=log_cols)
    log = log.append(log_entry)
    
log = log.drop(['index'])
print(log)