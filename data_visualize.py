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
import seaborn as sns
from imblearn.combine import SMOTEENN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn import metrics as m
from sklearn.model_selection import StratifiedShuffleSplit
#%%
#df = pd.read_csv("bank-additional-preprocessed-mf.csv")
df = pd.read_csv("bank-additional.csv")
subscribed = df.y
#f,ax=plt.subplots(1,2,figsize=(18,8))
#colors=["#F08080", "#00FA9A"]
#labels = 'Refused a T.D. Suscription', 'Accepted a T.D. Suscription'
#df['y'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True, colors=colors, labels=labels,fontsize=14)
#ax[0].set_title('Term Deposits', fontsize=20)
#ax[0].set_xlabel('% of Total Potential Clients', fontsize=14)
#sns.countplot('y',data=df,ax=ax[1], palette=colors)
#ax[1].set_title('Term Deposits', fontsize=20)
#ax[1].set_xticklabels(['Refused', 'Accepted'], fontsize=14)
#plt.show()

#df = df.drop("y", axis=1)

subscribed = subscribed == "yes"
#print(df.head())
#print(df.shape)
#print(df.describe())
#
#print(df.describe(include=["object"]))
df = df.drop("default", axis=1)
categorical_vars = df.describe(include=["object"]).columns
continuous_vars = df.describe().columns
df = df.replace(to_replace=['unknown'], value = np.NaN , regex = True)
#df.info()
#print(continuous_vars)
#print(categorical_vars)

#_ = df.hist(column=continuous_vars, figsize = (15,15))

# Count plots of categorical variables

#fig, axes = plt.subplots(3, 3, figsize=(16, 16))
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.7, hspace=0.3)
#
#for i, ax in enumerate(axes.ravel()):
#    if i > 9:
#        ax.set_visible(False)
#        continue
#    sns.countplot(y = categorical_vars[i], data=df, ax=ax,order=df[categorical_vars[i]].value_counts().index)

#pd.plotting.scatter_matrix(df[continuous_vars], c=subscribed, alpha=.2, figsize=(15, 15), cmap="viridis");

def convert_to_int(df, new_column, target_column):
    df[new_column] = df[target_column].apply(lambda x: 0 if x == 'no' else 1)
    return df[new_column].value_counts()

lst = [df]

# Create a column with the numeric values of the months.
for column in lst:
    column.loc[column["month"] == "jan", "month_int"] = 1
    column.loc[column["month"] == "feb", "month_int"] = 2
    column.loc[column["month"] == "mar", "month_int"] = 3
    column.loc[column["month"] == "apr", "month_int"] = 4
    column.loc[column["month"] == "may", "month_int"] = 5
    column.loc[column["month"] == "jun", "month_int"] = 6
    column.loc[column["month"] == "jul", "month_int"] = 7
    column.loc[column["month"] == "aug", "month_int"] = 8
    column.loc[column["month"] == "sep", "month_int"] = 9
    column.loc[column["month"] == "oct", "month_int"] = 10
    column.loc[column["month"] == "nov", "month_int"] = 11
    column.loc[column["month"] == "dec", "month_int"] = 12

# Change datatype from int32 to int64
df["month_int"] = df["month_int"].astype(np.int64)

f, axes = plt.subplots(ncols=3, figsize=(15, 6))

# Graph Employee Satisfaction
sns.distplot(df['month_int'], kde=False, color="#ff3300", ax=axes[0]).set_title('Months of Marketing Activity Distribution')
axes[0].set_ylabel('Potential Clients Count')
axes[0].set_xlabel('Months')

# Graph Employee Evaluation
sns.distplot(df['age'], kde=False, color="#3366ff", ax=axes[1]).set_title('Age of Potentical Clients Distribution')
axes[1].set_ylabel('Potential Clients Count')

# Campaigns
sns.distplot(df['campaign'], kde=False, color="#546E7A", ax=axes[2]).set_title('Calls Received in the Marketing Campaign')
axes[2].set_ylabel('Potential Clients Count')

plt.show()