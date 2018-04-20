"""
Final Project for EE 559 Spring 2018
Program by :
@authors   : Pranav Gundewar    Aditya Killekar  
USC ID     : 4463612994         2051450417
Email      : gundewar@usc.edu   killekar@usc.edu
Dataset    : bank-additional.csv
Instructor : Professor Keith Jenkins
"""
# Importing Libraries
import numpy as np 
import pandas as pd
import handle_missing_data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA

def one_hot(df):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode 
    @return a DataFrame with one-hot encoding
    """
    # One-hot encode into 
    cols = ['job', 'marital', 'education', 'month', 'day_of_week', 'poutcome']
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    df = df.drop(cols,axis=1)
    return df

#def main():

#if __name__ == '__main__':
#    main()

print('Pre-Procesing the input data!\n')
df = handle_missing_data.main('RF')
df = one_hot(df)
print('\nPre-Processing Done!\n')

y = df['y'] 
df = df.drop(columns=['y'], axis=1)
X = df.drop(columns=['default'], axis=1)

"""
PCA
"""
pca = PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0, stratify = y)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_n=scaler.transform(X_train)
X_test_n=scaler.transform(X_test)

clsr_names=["Nearest Neighbors", "Linear SVM", "RBF SVM",
 "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
 "Naive Bayes"]
classifiers = [KNeighborsClassifier(5),
 SVC(kernel="linear", C=0.025), SVC(gamma=1, C=0.5),
 DecisionTreeClassifier(max_depth=5),
 RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
 MLPClassifier(alpha=1),
 AdaBoostClassifier(),
 GaussianNB()] 

print("Comparing different models:\n")
for name, clf in zip(clsr_names, classifiers):
    model=clf.fit(X_train_n,y_train)
    y_pred=model.predict(X_test_n)
    print(name+" Accuracy: {0:.4f}%".format(100*float((y_pred==y_test).sum())/float(len(y_test)))) 