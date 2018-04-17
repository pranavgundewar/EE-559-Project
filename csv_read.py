import pandas as pd
import re
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

unknown_feature_headers = ['job','marital','default','housing','loan']

def split_files():
    df = pd.read_csv('bank-additional.csv')
    df_filter_test = [(df['job'] == 'unknown'), (df['marital'] == 'unknown'), (df['default'] == 'unknown'),\
                 (df['housing'] == 'unknown'), (df['loan'] == 'unknown')]
    df_filter_train = [(df['job'] != 'unknown'), (df['marital'] != 'unknown'), (df['default'] != 'unknown'),\
                 (df['housing'] != 'unknown'), (df['loan'] != 'unknown')]

    job_unknown_df = df[(df['job'] == 'unknown')]
    job_unknown_df = job_unknown_df[['age','campaign','pdays','previous','poutcome','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','y','job']]
    # for i,j in zip(df_filter_train,unknown_headers):
    #     job_unknown_df = df[i]
    #     job_unknown_df[['age','campaign','pdays','previous','poutcome','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','y',j]].to_csv('train_'+j+'.csv')
    #
    # for i,j in zip(df_filter_test,unknown_headers):
    #     job_unknown_df = df[i]
    #     job_unknown_df[['age','campaign','pdays','previous','poutcome','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','y']].to_csv('test_'+j+'.csv')
    print(job_unknown_df)
    return job_unknown_df

def one_hot_encode_features(data):
    poutcome_features = data.pop('poutcome')
    label_encoder = LabelEncoder()
    one_hot_encoder = OneHotEncoder(sparse=False)
    main_poutcome_features = label_encoder.fit_transform(poutcome_features)
    main_poutcome_features = main_poutcome_features.reshape(
        len(main_poutcome_features), 1)
    main_poutcome_features = pd.DataFrame(
        one_hot_encoder.fit_transform(main_poutcome_features)
    )
    print(main_poutcome_features)


def main():
    data = split_files()
    one_hot_encode_features(data)

if __name__ == '__main__':
    main()
