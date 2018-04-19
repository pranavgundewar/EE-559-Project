import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np


unknown_feature_headers = ['job','marital','education','default','housing','loan']

def split_files(df):
    df_filter_test = [(df['job'] == 'unknown'), (df['marital'] == 'unknown'), (df['education'] == 'unknown'), \
                      (df['default'] == 'unknown'), (df['housing'] == 'unknown'), (df['loan'] == 'unknown')]
    df_filter_train = [(df['job'] != 'unknown'), (df['marital'] != 'unknown'), (df['education'] != 'unknown'), \
                       (df['default'] != 'unknown'), (df['housing'] != 'unknown'), (df['loan'] != 'unknown')]

    for i,j in zip(df_filter_train,unknown_feature_headers):
        job_unknown_df = df[i]
        job_unknown_df = job_unknown_df[['age','campaign','pdays','previous','poutcome','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','y',j]]
        # job_unknown_df.to_csv('train_'+j+'.csv')
        job_unknown_df = one_hot(job_unknown_df,['poutcome'])
        convert_to_int(job_unknown_df, "y", "y")
        job_unknown_df = labelizer(job_unknown_df,j)
        job_unknown_df.to_csv('train_one_hot_'+j+'.csv')

    for i,j in zip(df_filter_test,unknown_feature_headers):
        job_unknown_df = df[i]
        job_unknown_df = job_unknown_df[['age','campaign','pdays','previous','poutcome','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','y']]
        # job_unknown_df.to_csv('test_'+j+'.csv')
        job_unknown_df = one_hot(job_unknown_df,['poutcome'])
        convert_to_int(job_unknown_df, "y", "y")
        job_unknown_df.to_csv('test_one_hot_'+j+'.csv')

def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    df = df.drop(cols,axis=1)
    return df

def convert_to_int(df, new_column, target_column):
    df[new_column] = df[target_column].apply(lambda x: 0 if x == 'no' else 1)
    return df[new_column].value_counts()

def labelizer(data, col):
    label_enc = LabelEncoder()
    data[col+'_new'] = label_enc.fit_transform(data[col])
    np.save(col+'_decode.npy',label_enc.classes_)
    data = data.drop([col],axis=1)
    return data

def main():
    df = pd.read_csv('bank-additional.csv')
    split_files(df)

if __name__ == '__main__':
    main()
