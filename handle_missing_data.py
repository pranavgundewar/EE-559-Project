from keras.layers import Input, Dense
from keras.models import Model, Sequential
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from pandas import Series
from sklearn.preprocessing import label_binarize


def handle_missing_data_using_NN(data, test_data):
    hidden_layer_config = [20, 60, 110, 210, 750, 1850, 660, 450, 210, 20]
    train_data = data[['age','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','y','poutcome_failure','poutcome_nonexistent','poutcome_success']].values
    train_labels = data[data.columns.difference(['age','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','y','poutcome_failure','poutcome_nonexistent','poutcome_success'])].values
    train_labels = np.delete(train_labels, 0, 1)

    test_data = test_data[['age','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','y','poutcome_failure','poutcome_nonexistent','poutcome_success']].values

    regressor = neural_model(input_shape=train_data.shape[1],output_shape=train_labels.shape[1],hidden_layer_config=hidden_layer_config)

    result, predictions = run_network(regressor, train_data, train_labels, test_data, epochs=20)
    predictions = (predictions == predictions.max(axis=1)[:,None]).astype(int)

    print(result.history.keys())
    print(predictions)

    from keras.utils import plot_model
    plot_model(regressor, to_file='model.png', show_shapes=True)

    plt.plot(result.history['loss'])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(result.history['val_loss'])
    plt.show()

def handle_missing_data_using_SVM(data, test_data):
    train_data = data[['age','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']].values
    train_labels = data[data.columns.difference(['age','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','y','poutcome_failure','poutcome_nonexistent','poutcome_success'])].values
    train_labels = np.delete(train_labels, 0, 1).reshape((1,-1))[0]

    test_data = test_data[['age','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']].values

    train_data, train_labels = shuffle(train_data,train_labels)

    scaler = preprocessing.StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    print(train_data)
    print(test_data)

    k_fold = StratifiedKFold(n_splits=5,shuffle=True)
    all_acc = []
    count = 1

    for train_index, dev_index in k_fold.split(train_data,train_labels):
        X_cv_train, X_cv_dev = train_data[train_index], train_data[dev_index]
        Y_cv_train, Y_cv_dev = train_labels[train_index], train_labels[dev_index]
        svm_classifier = SVC(C=1000,kernel='rbf',gamma=1)
        svm_classifier.fit(X_cv_train, Y_cv_train)
        acc = svm_classifier.score(X_cv_dev,Y_cv_dev)
        all_acc.append(acc)
        print("Accuracy with Split #{0}: {1}".format(count,acc))
        count += 1
        predictions = svm_classifier.predict(test_data)
        print(predictions)
    avg_acc = np.mean(all_acc)
    print("Average Accuracy: {0}".format(avg_acc))

def handle_missing_data_using_RF(data, test, feature):
    train_data = data[['age','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','y','poutcome_failure','poutcome_nonexistent','poutcome_success']].values
    train_labels = data[data.columns.difference(['age','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','y','poutcome_failure','poutcome_nonexistent','poutcome_success'])].values
    train_labels = np.delete(train_labels, 0, 1).reshape((1,-1))[0]

    test_data = test[['age','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','y','poutcome_failure','poutcome_nonexistent','poutcome_success']].values

    rf_clf = RandomForestClassifier(random_state=0)
    rf_clf.fit(train_data, train_labels)
    predictions = rf_clf.predict(test_data)

    if os.path.isfile('bank-additional-preprocessed.csv'):
        filled_df = pd.read_csv('bank-additional-preprocessed.csv')
    else:
        filled_df = pd.read_csv('bank-additional.csv')

    label_decode = np.load(feature+'_decode.npy')
    indices = test.ix[:,0]

    for pred, i in zip(predictions, indices):
        filled_df.loc[i, feature] = label_decode[pred]
    
    filled_df.to_csv('bank-additional-preprocessed.csv',index=False)
    return filled_df
    

def neural_model(input_shape, output_shape, hidden_layer_config=[10, 15]):
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

def run_network(network, X, Y, X_test, epochs=50):
    # print(train_test_split(X, Y, test_size=0.3))
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
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

    predictions = network.predict(X_test, batch_size=20)

    return model_result, predictions

# Convert the columns that contain a Yes or No. (Binary Columns)
def convert_to_int(df, new_column, target_column):
    df[new_column] = df[target_column].apply(lambda x: 0 if x == 'no' else 1)
    return df[new_column].value_counts()

def main(method):
    if os.path.isfile('bank-additional-preprocessed.csv'):
        os.remove('bank-additional-preprocessed.csv')
    # df_train = pd.read_csv('train_one_hot_job.csv')
    # df_test = pd.read_csv('test_one_hot_job.csv')
    df = pd.read_csv('bank-additional.csv', delimiter = ',')
    df = df.replace(to_replace=['unknown'], value = np.NaN , regex = True)
    globbed_files = glob.glob('train_one_hot_*'+'.csv')
    for csv in globbed_files:
        df_train = pd.read_csv(csv)
        test_csv = csv.split('_')[-1].split('.')[0]
        print('Handling missing data for '+test_csv)
        df_test = pd.read_csv('test_one_hot_'+test_csv+'.csv')
        if method == 'RF':
            df = handle_missing_data_using_RF(df_train, df_test, test_csv)
        elif method == 'NN':
            df = handle_missing_data_using_NN(df_train, df_test)
        elif method == 'SVM':
            df = handle_missing_data_using_SVM(df_train,df_test)
            
        elif method == 'mode':
            df = pd.read_csv('bank-additional.csv', delimiter = ',')
            df = df.replace(to_replace=['unknown'], value = np.NaN , regex = True)
            cols = df.columns[df.isna().any()].tolist()
            df[cols] = df[cols].fillna(df.mode().iloc[0])
        
        elif method == 'ffill':
            # forward-fill
            df = df.fillna(method='ffill')
        
        elif method == 'bfill':
            # back-fill
            df = df.fillna(method='bfill')
        
        elif method == 'drop':
            df = pd.read_csv('bank-additional.csv', delimiter = ',')
            df = df.replace(to_replace=['unknown'], value = np.NaN , regex = True)
            df=df.dropna(axis=0, how='any',) 
        convert_to_int(df, "y", "y") #Create a deposit int
        convert_to_int(df, "housing", "housing") # Create housingint column
        convert_to_int(df, "loan", "loan") #Create a loan_int column
        convert_to_int(df, "default", "default") #Create a default_int column
        c = Series.unique(df['contact'])
        df['contact'] = label_binarize(df['contact'], classes=c)    

    return df



#if __name__ == '__main__':
#    main()
