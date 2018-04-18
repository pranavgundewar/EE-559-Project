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
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

# Data loading and preprocessing
df = pd.read_csv('bank-additional.csv', delimiter = ',')
#term_deposits = df.copy()
# Have a grasp of how our data looks.
#print(df.dtypes)


# Pre-procesing the data
"""
If there were missing values we will have to fill them with the median, mean or mode.
I prefer to use the median as it works better with large datasets
"""
# No missing values ?
df = df.replace(to_replace=['unknown'], value = np.NaN , regex = True)
#print(df.info())
#print(df.describe())

# Replace missing data marked as unknown to NaN to use imputation to fill the missing data

#Imputer
#df['housing'] = df['housing'].fillna(df['housing'].mode()[0])

## forward-fill
#df = df.fillna(method='ffill')
#
## back-fill
#df = df.fillna(method='bfill')

cols = df.columns[df.isna().any()].tolist()
print('Columns that has missing values:',cols)
df[cols] = df[cols].fillna(df.mode().iloc[0])

# Let's see how the numeric data is distributed.
#df.hist(bins=10, figsize=(12,12), color='#E14906')
#plt.show()

# Binarize the columns
# Convert the columns that contain a Yes or No. (Binary Columns)
def convert_to_int(df, new_column, target_column):
    df[new_column] = df[target_column].apply(lambda x: 0 if x == 'no' else 1)
    return df[new_column].value_counts()

convert_to_int(df, "y", "y") #Create a deposit int
convert_to_int(df, "housing", "housing") # Create housingint column
convert_to_int(df, "loan", "loan") #Create a loan_int column
convert_to_int(df, "default", "default") #Create a default_int column
c = Series.unique(df['contact'])
df['contact'] = label_binarize(df['contact'], classes=c)


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
cols = ['job', 'marital', 'education']
print('Columns to be one-hot encoded:',cols)
df1 = one_hot(df, cols)
df1.to_csv('onehot.csv')
