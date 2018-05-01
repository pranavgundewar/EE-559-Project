#%%
# Importing multiple libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import f_regression, f_classif, SelectKBest
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.linear_model import SGDClassifier, LassoCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, MaxAbsScaler
from sklearn.feature_selection import VarianceThreshold, SelectPercentile
from sklearn.kernel_approximation import Nystroem
from sklearn.neural_network import MLPClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import EditedNearestNeighbours, CondensedNearestNeighbour
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_union
import itertools
import handle_missing_data
from sklearn.metrics import confusion_matrix
# from xgboost import XGBClassifier
import matplotlib.pyplot as plt
#import seaborn as sns
plt.rcParams["figure.dpi"] = 100

#%%

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

#%%

df = pd.read_csv("bank-additional-preprocessed-mode.csv")
print('Pre-Procesing the input data!\n')
#df = handle_missing_data.main('mode')
#df.to_csv('bank-additional-preprocessed-mode.csv')
#df = one_hot(df)
print('\nPre-Processing Done!\n')

#df = pd.read_csv("bank-additional.csv")
y = df['y'] 
df = df.drop('y', axis=1)
X = df.drop('default', axis=1)
#subscribed = subscribed == "yes"
#print(df.head())
#print(df.shape)
#print(df.describe())
#
#print(df.describe(include=["object"]))
df = df.drop("default", axis=1)
categorical_vars = df.describe(include=["object"]).columns
continuous_vars = df.describe().columns
#df = df.replace(to_replace=['unknown'], value = np.NaN , regex = True)
#df.info()
#print(continuous_vars)
#print(categorical_vars)

#%%
# Creating dummy variables and finding different categorical and continous variables
data_df, holdout_df, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = y)
# Creating dummy variables
data_dummies_df = pd.get_dummies(data_df, columns=categorical_vars, drop_first=False)
holdout_dummies_df = pd.get_dummies(holdout_df, columns=categorical_vars, drop_first=False)

fpr_vals = {}
tpr_vals = {}
conf_mat_vals = {}
#holdout_dummies_df['default_yes'] = 0
#holdout_dummies_df = holdout_dummies_df[data_dummies_df.columns]
categorical_dummies = pd.get_dummies(data_df[categorical_vars], drop_first=False).columns

select_categorical = FunctionTransformer(lambda X: X[categorical_dummies],validate = False)
select_continuous = FunctionTransformer(lambda X: X[continuous_vars],validate = False)

#%%
def plot_confusion_matrix(cm,
                          name,
                          normalize=False,
                          cmap=plt.cm.Blues,
                          title='Confusion matrix '):
                          
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    The count of true negatives is C_{0,0}, 
                 false negatives is C_{1,0},
                 true positives is C_{1,1},
                 false positives is C_{0,1}.
    '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.grid(False)
    plt.title(title+name)
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
    plt.show()

#%%
def plot_roc_curve(y_test, y_prob, name):
    fpr, tpr,t = roc_curve(y_test.ravel(), y_prob[:,1].ravel())
    fpr_vals[name] = fpr
    tpr_vals[name] = tpr
#    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % auc(fpr,tpr))
#    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
#    plt.xlim([0.0, 1.0])
#    plt.ylim([0.0, 1.05])
#    plt.xlabel('False Positive Rate')
#    plt.ylabel('True Positive Rate')
#    plt.title('Receiver operating characteristic')
#    plt.legend(loc="lower right")

#%%
def show_roc_plot(k,fpr_vals,tpr_vals):
    for i in k:
        plt.plot(fpr_vals[i], tpr_vals[i], lw=1, label='%s ROC curve (area = %0.2f)' % (i,auc(fpr_vals[i],tpr_vals[i])))
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

#%%
def plot_conf_mat(y_test,y_pred, name):
    conf_mat = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)
    conf_mat_vals[name] = conf_mat

#%%
def show_conf_mat(k, conf_mat_vals):
    for i in k:
        plot_confusion_matrix(conf_mat_vals[i],name=i)

#%%
print('Logistic Regression CV with StandardScaler: ')
fu = make_union(select_categorical, make_pipeline(select_continuous, StandardScaler()))
pipe = make_pipeline(fu, LogisticRegressionCV())
pipe.fit(data_dummies_df, y_train)
print('ROC Score: ',np.mean(cross_val_score(pipe, data_dummies_df, y_train, cv=5, scoring="roc_auc")))
y_pred = pipe.predict(holdout_dummies_df)
y_prob = pipe.predict_proba(holdout_dummies_df)
print("F1: %1.3f" % f1_score(y_test, y_pred, average='weighted'))
plot_conf_mat(y_test, y_pred, 'LRCVSS')

#%%
#plt.figure(figsize=(5, 15))
#coef = pd.Series(pipe.named_steps['logisticregressioncv'].coef_.ravel(), index=data_dummies_df.columns)
#coef.sort_values().plot(kind="barh")
#plt.figure()
plot_roc_curve(y_test, y_prob, 'LRCVSS')



#%%
print('Logistic Regression with Feature Selection:')
pipe_fs = make_pipeline(StandardScaler(), SelectPercentile(score_func=f_classif, percentile=50),
                     PolynomialFeatures(interaction_only=True),
                     VarianceThreshold(),
                     LogisticRegressionCV())
pipe_fs.fit(data_dummies_df, y_train)
print('ROC Score: ',np.mean(cross_val_score(pipe_fs, data_dummies_df, y_train, cv=5, scoring="roc_auc")))

y_pred = pipe_fs.predict(holdout_dummies_df)
y_prob = pipe_fs.predict_proba(holdout_dummies_df)
print("F1: %1.3f" % f1_score(y_test, y_pred, average='weighted'))
plot_conf_mat(y_test, y_pred, 'LRFS')

#%%
plot_roc_curve(y_test, y_prob, 'LRFS')

#%%
print('MLP:')
pipe_mlp = make_pipeline(StandardScaler(), MLPClassifier(alpha=1, hidden_layer_sizes=(160,)))
pipe_mlp.fit(data_dummies_df, y_train)
print('ROC Score: ',np.max(cross_val_score(pipe_mlp, data_dummies_df, y_train, cv=5, scoring="roc_auc")))
y_pred = pipe_mlp.predict(holdout_dummies_df)
y_prob = pipe_mlp.predict_proba(holdout_dummies_df)
print("F1: %1.3f" % f1_score(y_test, y_pred, average='weighted'))
plot_conf_mat(y_test, y_pred, 'MLP')

#%%
plot_roc_curve(y_test, y_prob, 'MLP')

#%%
print('KNN:')
k_range = list(range(1, 15))
knn = KNeighborsClassifier()
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid, cv=5, scoring='roc_auc')
grid.fit(data_dummies_df, y_train)
print('ROC Score: ',grid.best_score_)
y_pred = grid.predict(holdout_dummies_df)
y_prob = grid.predict_proba(holdout_dummies_df)
print("F1: %1.3f" % f1_score(y_test, y_pred, average='weighted'))
plot_conf_mat(y_test, y_pred, 'KNN')

#%%
plot_roc_curve(y_test, y_prob, 'KNN')


#%%
print('Logistic Regression with Lasso:')
select_lassocv = SelectFromModel(LassoCV(), threshold="median")
pipe_lassocv = make_pipeline(StandardScaler(), select_lassocv, LogisticRegressionCV())
print('ROC Score: ', np.mean(cross_val_score(pipe_lassocv, data_dummies_df, y_train, cv=5, scoring="roc_auc")))
pipe_lassocv.fit(data_dummies_df, y_train)
y_pred = pipe_lassocv.predict(holdout_dummies_df)
y_prob = pipe_lassocv.predict_proba(holdout_dummies_df)
print("F1: %1.3f" % f1_score(y_test, y_pred, average='weighted'))
plot_conf_mat(y_test, y_pred, 'LRL')

#%%
plot_roc_curve(y_test, y_prob, 'LRL')

#%%
print('Logistic Regression Parameter Tuning: ')
pipe_lrpt = make_pipeline(SelectPercentile(score_func=f_classif, percentile=50),
                     PolynomialFeatures(interaction_only=True),
                     VarianceThreshold(),
                     LogisticRegressionCV())
param_grid = {'selectpercentile__percentile': [50],
              'polynomialfeatures__degree': [2]}
grid = GridSearchCV(pipe_lrpt, param_grid, cv=5, scoring="roc_auc")
grid.fit(data_dummies_df, y_train)
print('ROC Score: ',grid.best_score_)
y_pred = grid.predict(holdout_dummies_df)
y_prob = grid.predict_proba(holdout_dummies_df)
print("F1: %1.3f" % f1_score(y_test, y_pred, average='weighted'))
plot_conf_mat(y_test, y_pred, 'LRPT')

#%%
plot_roc_curve(y_test, y_prob, 'LRPT')

#%%
print('Decision Trees: ')
pipe_dt = make_pipeline(DecisionTreeClassifier(max_depth = 6))
param_grid = {'decisiontreeclassifier__max_depth':range(1, 7)}
#param_grid = {}
grid = GridSearchCV(pipe_dt, param_grid=param_grid, scoring='roc_auc', cv=StratifiedShuffleSplit(100))
grid.fit(data_dummies_df, y_train)
print('ROC Score: ',grid.best_score_)
y_pred = grid.predict(holdout_dummies_df)
y_prob = grid.predict_proba(holdout_dummies_df) 
print("F1: %1.3f" % f1_score(y_test, y_pred, average='weighted'))
plot_conf_mat(y_test, y_pred, 'DT')

#%%
plot_roc_curve(y_test, y_prob, 'DT')

#%%
print('Support Vector Machines: ')
svm = LinearSVC()
params={}
#pipe_rf = make_pipeline(StandardScaler(),svm)
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
grid = GridSearchCV(SVC(probability=True), tuned_parameters, cv=5,scoring='roc_auc')
#grid = GridSearchCV(pipe_rf, param_grid = params, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
grid.fit(data_dummies_df, y_train)
print('ROC Score: ',grid.best_score_)
y_pred = grid.predict(holdout_dummies_df)
y_prob = grid.predict_proba(holdout_dummies_df) 
print("F1: %1.3f" % f1_score(y_test, y_pred, average='weighted'))
plot_conf_mat(y_test, y_pred, 'SVM')

#%%
plot_roc_curve(y_test, y_prob, 'SVM')

#%%
print('Random Forest Model 1: ')
rf = RandomForestClassifier(warm_start=True)
params = {'randomforestclassifier__n_estimators' : range(10,100,10),
          'randomforestclassifier__max_depth' : range(3,15,2)}
pipe_rf = make_pipeline(StandardScaler(),rf)
grid = GridSearchCV(pipe_rf, param_grid = params, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
grid.fit(data_dummies_df, y_train)
print('ROC Score: ',grid.best_score_)
y_pred = grid.predict(holdout_dummies_df)
y_prob = grid.predict_proba(holdout_dummies_df) 
print("F1: %1.3f" % f1_score(y_test, y_pred, average='weighted'))
plot_conf_mat(y_test, y_pred, 'RF1')

#%%
plot_roc_curve(y_test, y_prob, 'RF1')

#%%
print('Random Forest Model 2: ')
rf = RandomForestClassifier(warm_start=True, n_estimators=100, max_depth=9)
params = {'randomforestclassifier__max_features' : range(4, 44, 4),
          'randomforestclassifier__criterion' : ['gini', 'entropy']}
pipe = make_pipeline(StandardScaler(),rf)
grid = GridSearchCV(pipe,param_grid = params, scoring='roc_auc', n_jobs=1, iid=False, cv=5)
grid.fit(data_dummies_df, y_train)
print('ROC Score: ',grid.best_score_)
y_pred = grid.predict(holdout_dummies_df)
y_prob = grid.predict_proba(holdout_dummies_df) 
print("F1: %1.3f" % f1_score(y_test, y_pred, average='weighted'))
plot_conf_mat(y_test, y_pred, 'RF2')

#%%
plot_roc_curve(y_test, y_prob, 'RF2')

#%%
print('Random Forest Final Model: ')
rf = RandomForestClassifier(warm_start=True, n_estimators=80,
                            max_depth=9,
                            max_features = 16,
                            criterion = 'entropy')
params = {}
pipe = make_pipeline(rf)
grid = GridSearchCV(pipe, param_grid = params, scoring='roc_auc', n_jobs=1,iid=False, cv=5)
grid.fit(data_dummies_df, y_train)
print('ROC Score: ',grid.best_score_)
y_pred = grid.predict(holdout_dummies_df)
y_prob = grid.predict_proba(holdout_dummies_df)
print("F1: %1.3f" % f1_score(y_test, y_pred, average='weighted'))
plot_conf_mat(y_test, y_pred, 'RF_final')

#%%
plot_roc_curve(y_test, y_prob, 'RF_final')

#%%
print('Undersampling with Logistic Regression:')
undersample_pipe = make_imb_pipeline(RandomUnderSampler(), LogisticRegressionCV())
scores = cross_val_score(undersample_pipe, data_dummies_df, y_train, cv=10, scoring='roc_auc')
print('ROC Score: ',np.mean(scores))
undersample_pipe.fit(data_dummies_df,y_train)
y_pred = undersample_pipe.predict(holdout_dummies_df)
y_prob = undersample_pipe.predict_proba(holdout_dummies_df)
print("F1: %1.3f" % f1_score(y_test, y_pred, average='weighted'))
plot_conf_mat(y_test, y_pred, 'Undersample_LogR')

#%%
plot_roc_curve(y_test, y_prob, 'Undersample_LogR')

#%%
print('Oversampling with Logistic Regression:')
oversample_pipe = make_imb_pipeline(RandomOverSampler(), LogisticRegressionCV())
scores = cross_val_score(oversample_pipe, data_dummies_df, y_train, cv=10, scoring='roc_auc')
print('ROC Score: ',np.mean(scores))
oversample_pipe.fit(data_dummies_df,y_train)
y_pred = oversample_pipe.predict(holdout_dummies_df)
y_prob = oversample_pipe.predict_proba(holdout_dummies_df)
print("F1: %1.3f" % f1_score(y_test, y_pred, average='weighted'))
plot_conf_mat(y_test, y_pred, 'oversample_LogR')

#%%
plot_roc_curve(y_test, y_prob, 'oversample_LogR')

#%%
print('Undersampling with Random Forest:')
undersample_pipe_rf = make_imb_pipeline(RandomUnderSampler(),
                                        RandomForestClassifier(warm_start=True,
                                                               n_estimators=80,
                                                               max_depth=9,
                                                               max_features = 16,
                                                               criterion = 'entropy'))
scores = cross_val_score(undersample_pipe_rf, data_dummies_df, y_train, cv=10, scoring='roc_auc')
print('ROC Score: ',np.mean(scores))
undersample_pipe_rf.fit(data_dummies_df, y_train)
y_pred = undersample_pipe.predict(holdout_dummies_df)
y_prob = undersample_pipe_rf.predict_proba(holdout_dummies_df)
print("F1: %1.3f" % f1_score(y_test, y_pred, average='weighted'))
plot_conf_mat(y_test, y_pred, 'Undersample_RF')

#%%
plot_roc_curve(y_test, y_prob, 'Undersample_RF')

#%%
print('Oversampling with Random Forest:')
oversample_pipe_rf = make_imb_pipeline(RandomOverSampler(), RandomForestClassifier(warm_start=True,
                                                                                   n_estimators=80,
                                                                                   max_depth=9,
                                                                                   max_features = 16,
                                                                                   criterion = 'entropy'))
scores = cross_val_score(oversample_pipe_rf, data_dummies_df, y_train, cv=10, scoring='roc_auc')
print('ROC Score: ',np.mean(scores))
oversample_pipe_rf.fit(data_dummies_df, y_train)
y_pred = oversample_pipe.predict(holdout_dummies_df)
y_prob = oversample_pipe.predict_proba(holdout_dummies_df)
print("F1: %1.3f" % f1_score(y_test, y_pred, average='weighted'))
plot_conf_mat(y_test, y_pred, 'oversample_RF')

#%%
plot_roc_curve(y_test, y_prob, 'oversample_RF')

#%%
k = ['LRCVSS', 'LRFS', 'LRL', 'LRPT', 'DT','RF1','RF2','RF_final','Undersample_LogR',\
     'oversample_LogR','Undersample_RF','oversample_RF','MLP','KNN','SVM']
show_roc_plot(k,fpr_vals,tpr_vals)
show_conf_mat(k,conf_mat_vals)