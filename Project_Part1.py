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
from pandas import Series
from sklearn.preprocessing import label_binarize
import random

# Binarize the columns
# Convert the columns that contain a Yes or No. (Binary Columns)
def convert_to_int(df, new_column, target_column):
    df[new_column] = df[target_column].apply(lambda x: 0 if x == 'no' else 1)
    return df[new_column].value_counts()

def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode 
    @return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    return df

def handle_missing_data(df, method):
    """
    :param 
    :param method: {mode, median, remove, backward filling, forward filling, KNN, SVM}
    :return: data with filled missing values
    :missing_handles: fillna with parameters of training set, since it will be unknown for test set
    """

    if method == 'mode':
        cols = df.columns[df.isna().any()].tolist()
        print('Columns that has missing values:',cols)
        df[cols] = df[cols].fillna(df.mode().iloc[0])
        
    elif method == 'ffill':
        # forward-fill
        df = df.fillna(method='ffill')
        
    elif method == 'bfill':
        # back-fill
        df = df.fillna(method='bfill')
        
    elif method == 'drop':
        df=df.dropna(axis=0, how='any') 

    return df

#def main():
#    cols = ['job', 'marital', 'education']
#    print('Columns to be one-hot encoded:',cols)
#    # Data loading and preprocessing
#    df = pd.read_csv('bank-additional.csv', delimiter = ',')
#    # No missing values ?
#    df = df.replace(to_replace=['unknown'], value = np.NaN , regex = True)
#    df = handle_missing_data(df, 'mode')
#    convert_to_int(df, "y", "y") #Create a deposit int
#    convert_to_int(df, "housing", "housing") # Create housingint column
#    convert_to_int(df, "loan", "loan") #Create a loan_int column
#    convert_to_int(df, "default", "default") #Create a default_int column
#    c = Series.unique(df['contact'])
#    df['contact'] = label_binarize(df['contact'], classes=c)
#    print(df['default'].value_counts())
#    df1 = one_hot(df, cols)
#    
#    
#if __name__ == '__main__':
#    main()

cols = ['job', 'marital', 'education']
#print('Columns to be one-hot encoded:',cols)
# Data loading and preprocessing
df = pd.read_csv('bank-additional.csv', delimiter = ',')
print(df['housing'].value_counts())
# No missing values ?
df = df.replace(to_replace=['unknown'], value = np.NaN , regex = True)
df = handle_missing_data(df, 'mode')
convert_to_int(df, "y", "y") #Create a deposit int
convert_to_int(df, "housing", "housing") # Create housingint column
convert_to_int(df, "loan", "loan") #Create a loan_int column
convert_to_int(df, "default", "default") #Create a default_int column
c = Series.unique(df['contact'])
df['contact'] = label_binarize(df['contact'], classes=c)
print(df['housing'].value_counts())
df1 = one_hot(df, cols)
