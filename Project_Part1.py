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
import itertools
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

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
#sm = SMOTEENN()
#X, y = sm.fit_sample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0) #, stratify = y)

#%%
#scaler = MinMaxScaler()
##scaler = RobustScaler()
##scaler = StandardScaler()
#scaler.fit(X_train)
#X_train_n=scaler.transform(X_train)
#X_test_n=scaler.transform(X_test)
"""
PCA
"""
#pca = PCA(n_components=3)
#pca.fit(X_train_n)
#X_train_n = pca.transform(X_train_n)
#X_test_n = pca.transform(X_test_n)

#clsr_names=["Nearest Neighbors", "Linear SVM",
# "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
# "Naive Bayes", "RBF SVM"]
#classifiers = [KNeighborsClassifier(9),
# SVC(kernel="linear", C=0.025),
# DecisionTreeClassifier(max_depth=5),
# RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
# MLPClassifier(alpha=1),
# AdaBoostClassifier(),
# GaussianNB(), SVC(gamma=2, C=1)] 
#
import warnings
warnings.filterwarnings('ignore')
#print("Comparing different models:\n")
#for name, clf in zip(clsr_names, classifiers):
#    model=clf.fit(X_train,y_train)
#    y_pred=model.predict(X_test)
#    print(name+" Accuracy: {0:.4f}%".format(100*float((y_pred==y_test).sum())/float(len(y_test))))
#    print(name+" F1: %1.3f" % f1_score(y_test, y_pred, average='macro'))
#    print(name+" ROC Score: {:.4f}%".format(roc_auc_score(y_pred,y_test)))
    

classifiers = {'Gradient Boosting Classifier':GradientBoostingClassifier(),'Adaptive Boosting Classifier'
               :AdaBoostClassifier(),'Linear Discriminant Analysis':LinearDiscriminantAnalysis(),
               'Logistic Regression':LogisticRegression(),'Random Forest Classifier': RandomForestClassifier(),
               'K Nearest Neighbour':KNeighborsClassifier(7),'Decision Tree Classifier'
               :DecisionTreeClassifier(),'Gaussian Naive Bayes Classifier':GaussianNB(),
               'Support Vector Classifier':SVC(probability=True), 'Support Vector Classifier Linear':SVC(probability=True, kernel='linear'),
               'Perceptron':Perceptron(penalty='l2', max_iter = 1000),  
               'MLP': MLPClassifier(alpha=1, hidden_layer_sizes=(160,))}

log_cols = ["Classifier", "Accuracy","F1-Score","roc-auc_Score"] #"Precision Score","Recall Score",]
#metrics_cols = []
log = pd.DataFrame(columns=log_cols)


#rs = StratifiedShuffleSplit(n_splits=2, test_size=0.2,random_state=0)
#rs.get_n_splits(X_train,y_train)
#for Name,classify in classifiers.items():
#    for train_index, test_index in rs.split(X_train,y_train):
#        #print("TRAIN:", train_index, "TEST:", test_index)
#        X,X_test = X_train.iloc[train_index], X_train.iloc[test_index]
#        y,y_test = y_train.iloc[train_index], y_train.iloc[test_index]
#        # Scaling of Features 
#        sc_X = StandardScaler()
#        X = sc_X.fit_transform(X)
#        X_test = sc_X.transform(X_test)
#        cls = classify
#        cls =cls.fit(X,y)
#        y_out = cls.predict(X_test)
#        accuracy = m.accuracy_score(y_test,y_out)
#        precision = m.precision_score(y_test,y_out,average='weighted')
#        recall = m.recall_score(y_test,y_out,average='weighted')
#        roc_auc = roc_auc_score(y_out,y_test)
#        f1_score = m.f1_score(y_test,y_out,average='weighted')
#        log_entry = pd.DataFrame([[Name,accuracy,precision,recall,f1_score,roc_auc]], columns=log_cols)
#        #metric_entry = pd.DataFrame([[precision,recall,f1_score,roc_auc]], columns=metrics_cols)
#        log = log.append(log_entry)
#        #metric = metric.append(metric_entry)
        
# Resampling the data to tackle the imbalance

rs = StratifiedShuffleSplit(n_splits=5, test_size=0.2,random_state=0)
for Name,classify in classifiers.items():
    accuracy=[]
    precision=[]
    recall=[]
    roc_auc=[]
    f1_score=[]
    for train_index, test_index in rs.split(X_train,y_train):
        #print("TRAIN:", train_index, "TEST:", test_index)
        y,y_test = y_train.iloc[train_index], y_train.iloc[test_index]
        X,X_test = X_train.iloc[train_index], X_train.iloc[test_index]        
        # Scaling of Features 
        sc_X = StandardScaler()
        X = sc_X.fit_transform(X)
        X_test = sc_X.transform(X_test)
#        pca = PCA(n_components=10)
#        pca.fit(X)
#        X = pca.transform(X)
#        X_test = pca.transform(X_test)
        
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
#plt.xlabel('Accuracy')
#plt.title('Classifier Accuracy')
#sns.set_color_codes("muted")
#sns.barplot(x='Accuracy', y='Classifier', data=log, color="g")  
#plt.show()

#%%
def plot_confusion_matrix(cm,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#%%
    
X_train, x_test, y_train, y_tst = train_test_split(X, y, test_size=0.2, stratify = y)
skf = StratifiedShuffleSplit(n_splits=5, test_size=0.2,random_state=0)
#skf = StratifiedKFold(n_splits=5, shuffle=True)   #Stratified K fold for 5 folds preserving percentage of each class 
all_acc = []
cnt=1
for train_index, test_index in skf.split(X_train, y_train): #Cross Vaidating Data used for Training SVM classifier for 5 folds
   y,y_test = y_train.iloc[train_index], y_train.iloc[test_index]
   X,X_test = X_train.iloc[train_index], X_train.iloc[test_index] 
   clf = SVC(C=1000,gamma=0.0001,kernel='rbf',probability=True)  # Create an object of SVM classifier
   clf.fit(X_train, y_train)   # Train the classifier
   y_pred = clf.predict(X_test)     # Predicting labels on Test Validation Data
   print("Accuracy in fold {:d} =".format(cnt),"{:.4f}".format(m.accuracy_score(y_test, y_pred)*100))
   all_acc.append(m.accuracy_score(y_test, y_pred))  #Calculate testing accuracy
   cnt +=1
#Calculating Mean accuracy over n folds
print("Average of the 5 fold cross validation accuracy = {:.4f}".format(np.mean(all_acc)*100))
y_prob = clf.predict_proba(x_test)
y_pred = clf.predict(x_test)
print("F1: %1.3f" % f1_score(y_tst, y_pred, average='weighted'))
cm = confusion_matrix(y_tst, y_pred)
print("Roc: ",roc_auc_score(y_tst,y_pred, average='weighted'))
plot_confusion_matrix(cm)