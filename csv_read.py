import pandas as pd
import re
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

unknown_feature_headers = ['job','marital','education','default','housing','loan']

def split_files(df):
    df_filter_test = [(df['job'] == 'unknown'), (df['marital'] == 'unknown'), (df['education'] == 'unknown'), \
                      (df['default'] == 'unknown'), (df['housing'] == 'unknown'), (df['loan'] == 'unknown')]
    df_filter_train = [(df['job'] != 'unknown'), (df['marital'] != 'unknown'), (df['education'] != 'unknown'), \
                       (df['default'] != 'unknown'), (df['housing'] != 'unknown'), (df['loan'] != 'unknown')]

    # job_unknown_df = df[(df['job'] == 'unknown')]
    # job_unknown_df = job_unknown_df[['age','campaign','pdays','previous','poutcome','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','y','job']]
    for i,j in zip(df_filter_train,unknown_feature_headers):
        job_unknown_df = df[i]
        job_unknown_df = job_unknown_df[['age','campaign','pdays','previous','poutcome','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','y',j]]
        job_unknown_df = one_hot(job_unknown_df,['poutcome',j])
        job_unknown_df.to_csv('train_'+j+'.csv')

    for i,j in zip(df_filter_test,unknown_feature_headers):
        job_unknown_df = df[i]
        job_unknown_df = job_unknown_df[['age','campaign','pdays','previous','poutcome','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','y']]
        job_unknown_df = one_hot(job_unknown_df,['poutcome'])
        job_unknown_df.to_csv('test_'+j+'.csv')

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

def main():
    df = pd.read_csv('bank-additional.csv')
    split_files(df)

if __name__ == '__main__':
    main()
