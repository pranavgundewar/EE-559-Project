#%%
# Importing multiple libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.svm import LinearSVC
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
from sklearn.metrics import confusion_matrix
# from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.dpi"] = 100

#%%
df = pd.read_csv("bank-additional-preprocessed-mf.csv")
#df = pd.read_csv("bank-additional.csv")
subscribed = df.y

df = df.drop("y", axis=1)

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

#%%
# Creating dummy variables and finding different categorical and continous variables
data_df, holdout_df, y_train, y_test = train_test_split(df, subscribed, test_size=0.2, stratify = subscribed)
# Creating dummy variables
data_dummies_df = pd.get_dummies(data_df, columns=categorical_vars, drop_first=False)
holdout_dummies_df = pd.get_dummies(holdout_df, columns=categorical_vars, drop_first=False)

fpr_vals = {}
tpr_vals = {}
#holdout_dummies_df['default_yes'] = 0
#holdout_dummies_df = holdout_dummies_df[data_dummies_df.columns]
categorical_dummies = pd.get_dummies(data_df[categorical_vars], drop_first=False).columns

select_categorical = FunctionTransformer(lambda X: X[categorical_dummies],validate = False)
select_continuous = FunctionTransformer(lambda X: X[continuous_vars],validate = False)

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
def show_plot(k,fpr_vals,tpr_vals):
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
print('Logistic Regression CV with StandardScalar: ')
fu = make_union(select_categorical, make_pipeline(select_continuous, StandardScaler()))
pipe = make_pipeline(fu, LogisticRegressionCV())
pipe.fit(data_dummies_df, y_train)
print('ROC Score: ',np.mean(cross_val_score(pipe, data_dummies_df, y_train, cv=5, scoring="roc_auc")))
y_pred = pipe.predict(holdout_dummies_df)
y_prob = pipe.predict_proba(holdout_dummies_df)
print("F1: %1.3f" % f1_score(y_test, y_pred, average='weighted'))

#%%
plt.figure(figsize=(5, 15))
coef = pd.Series(pipe.named_steps['logisticregressioncv'].coef_.ravel(), index=data_dummies_df.columns)
coef.sort_values().plot(kind="barh")
plt.figure()
plot_roc_curve(y_test, y_prob, 'LRCVSS')


#%%
print('Logistic Regression with Kernel: ')
approx = Nystroem(gamma=1./data_dummies_df.shape[1], n_components=300)
pipe_lrk = make_pipeline(StandardScaler(), approx, LogisticRegressionCV())
pipe_lrk.fit(data_dummies_df, y_train)
print('ROC Score: ',np.mean(cross_val_score(pipe_lrk, data_dummies_df, y_train, cv=5, scoring="roc_auc")))
y_pred = pipe_lrk.predict(holdout_dummies_df)
y_prob = pipe_lrk.predict_proba(holdout_dummies_df)
print('ROC Score: ',roc_auc_score(y_test, y_pred))
print("F1: %1.3f" % f1_score(y_test, y_pred, average='weighted'))

#%%
plot_roc_curve(y_test, y_prob, 'LRK')

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
print('ROC Score: ',roc_auc_score(y_test, y_pred))
print("F1: %1.3f" % f1_score(y_test, y_pred, average='weighted'))

#%%
plot_roc_curve(y_test, y_prob, 'LRFS')

#%%
print('Logistic Regression with Lasso:')
select_lassocv = SelectFromModel(LassoCV(), threshold="median")
pipe_lassocv = make_pipeline(StandardScaler(), select_lassocv, LogisticRegressionCV())
print('ROC Score: ', np.mean(cross_val_score(pipe_lassocv, data_dummies_df, y_train, cv=10, scoring="roc_auc")))
pipe_lassocv.fit(data_dummies_df, y_train)
y_pred = pipe_lassocv.predict(holdout_dummies_df)
y_prob = pipe_lassocv.predict_proba(holdout_dummies_df)
print('ROC Score: ',roc_auc_score(y_test, y_pred))
print("F1: %1.3f" % f1_score(y_test, y_pred, average='weighted'))

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
print('ROC Score: ',roc_auc_score(y_test, y_pred))
print("F1: %1.3f" % f1_score(y_test, y_pred, average='weighted'))

#%%
plot_roc_curve(y_test, y_prob, 'LRPT')
#%%
k = ['LRCVSS', 'LRK', 'LRFS', 'LRL', 'LRPT']
show_plot(k,fpr_vals,tpr_vals)
#%%
print('Decision Trees: ')
pipe_dt = make_pipeline(DecisionTreeClassifier(max_depth = 6))
param_grid = {'decisiontreeclassifier__max_depth':range(1, 7)}
#param_grid = {}
grid = GridSearchCV(pipe_dt, param_grid=param_grid, scoring='roc_auc', cv=StratifiedShuffleSplit(100))
grid.fit(data_dummies_df, y_train)
print(grid.best_score_)
y_pred = grid.predict(holdout_dummies_df)
print('ROC Score: ',roc_auc_score(y_test, y_pred))
print("F1: %1.3f" % f1_score(y_test, y_pred, average='weighted'))

#%%
print('Random Forest Model 1: ')
rf = RandomForestClassifier(warm_start=True)
params = {'randomforestclassifier__n_estimators' : range(10,100,10),
          'randomforestclassifier__max_depth' : range(3,15,2)}
pipe_rf = make_pipeline(RobustScaler(),rf)
grid = GridSearchCV(pipe_rf, param_grid = params, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
grid.fit(data_dummies_df, y_train)
print(grid.best_score_)
y_pred = grid.predict(holdout_dummies_df)
print('ROC Score: ',roc_auc_score(y_test, y_pred))
print("F1: %1.3f" % f1_score(y_test, y_pred, average='weighted'))

#%%
print('Random Forest Model 2: ')
rf = RandomForestClassifier(warm_start=True, n_estimators=100, max_depth=9)
params = {'randomforestclassifier__max_features' : range(4, 44, 4),
          'randomforestclassifier__criterion' : ['gini', 'entropy']}
pipe = make_pipeline(RobustScaler(),rf)
grid = GridSearchCV(pipe,param_grid = params, scoring='roc_auc', n_jobs=1, iid=False, cv=5)
grid.fit(data_dummies_df, y_train)
print(grid.best_score_)
y_pred = grid.predict(holdout_dummies_df)
print('ROC Score: ',roc_auc_score(y_test, y_pred))
print("F1: %1.3f" % f1_score(y_test, y_pred, average='weighted'))

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
print(grid.best_score_)
y_pred = grid.predict(holdout_dummies_df)
print('ROC Score: ',roc_auc_score(y_test, y_pred))
print("F1: %1.3f" % f1_score(y_test, y_pred, average='weighted'))

#%%
#print('Gradient Boosting Classifier: ')
#param_test = {}
#pipe = make_pipeline(GradientBoostingClassifier(learning_rate=0.1, max_depth=5, n_estimators=80,
#                                                 random_state=10, min_samples_split = 1000))
#grid = GridSearchCV(pipe, param_grid = param_test, scoring='roc_auc', n_jobs=1, iid=True, cv=5)
#grid.fit(data_dummies_df, y_train)
#y_pred = grid.predict(holdout_dummies_df)
#print(grid.best_score_)
#print('ROC Score: ',roc_auc_score(y_test, y_pred))
#print("F1: %1.3f" % f1_score(y_test, y_pred, average='weighted'))

#%%
print('Ensembling:')
print('Voting Classifier with Poor Mans stacking: ')
clf1 = make_pipeline(StandardScaler(),LogisticRegressionCV(random_state=10))
clf2 = RandomForestClassifier(warm_start=True,n_estimators=80,max_depth=9,max_features = 16, criterion = 'entropy', random_state=10)
clf3 = GradientBoostingClassifier(learning_rate=0.1,max_depth=5, n_estimators=200, random_state=10,min_samples_split = 1000)

voting = VotingClassifier([('logreg', clf1),('RandomForest', clf2),('GradientBoost',clf3)], voting='soft')

print(np.mean(cross_val_score(voting, data_dummies_df, y_train,scoring = 'roc_auc', cv=5)))
voting.fit(data_dummies_df,y_train)
y_pred = voting.predict(holdout_dummies_df)
print('ROC Score: ',roc_auc_score(y_test, y_pred))
print("F1: %1.3f" % f1_score(y_test, y_pred, average='weighted'))

#%%
print('Voting Classifier with LRCV: ')
reshaper = FunctionTransformer(lambda X_: np.rollaxis(X_, 1).reshape(-1, 6)[:, 1::2], validate=False)
stacking = make_pipeline(voting, reshaper,
                         LogisticRegressionCV())
print(np.mean(cross_val_score(stacking,data_dummies_df, y_train, scoring = 'roc_auc',cv=5)))
stacking.fit(data_dummies_df, y_train)
y_pred = stacking.predict(holdout_dummies_df)
print('ROC Score: ',roc_auc_score(y_test, y_pred))
print("F1: %1.3f" % f1_score(y_test, y_pred, average='weighted'))

#%%
print('Undersampling with Logistic Regression:')
undersample_pipe = make_imb_pipeline(RandomUnderSampler(), LogisticRegressionCV())
scores = cross_val_score(undersample_pipe, data_dummies_df, y_train, cv=10, scoring='roc_auc')
print(np.mean(scores))
undersample_pipe.fit(data_dummies_df,y_train)
y_pred = undersample_pipe.predict(holdout_dummies_df)
print('ROC Score: ',roc_auc_score(y_test, y_pred))
print("F1: %1.3f" % f1_score(y_test, y_pred, average='weighted'))

#%%
print('Oversampling with Logistic Regression:')
oversample_pipe = make_imb_pipeline(RandomOverSampler(), LogisticRegressionCV())
scores = cross_val_score(oversample_pipe, data_dummies_df, y_train, cv=10, scoring='roc_auc')
print(np.mean(scores))
oversample_pipe.fit(data_dummies_df,y_train)
y_pred = oversample_pipe.predict(holdout_dummies_df)
print('ROC Score: ',roc_auc_score(y_test, y_pred))
print("F1: %1.3f" % f1_score(y_test, y_pred, average='weighted'))

#%%
print('Undersampling with Random Forest:')
undersample_pipe_rf = make_imb_pipeline(RandomUnderSampler(),
                                        RandomForestClassifier(warm_start=True,
                                                               n_estimators=80,
                                                               max_depth=9,
                                                               max_features = 16,
                                                               criterion = 'entropy'))
scores = cross_val_score(undersample_pipe_rf, data_dummies_df, y_train, cv=10, scoring='roc_auc')
print(np.mean(scores))
undersample_pipe_rf.fit(data_dummies_df, y_train)
y_pred = undersample_pipe.predict(holdout_dummies_df)
print('ROC Score: ',roc_auc_score(y_test, y_pred))
print("F1: %1.3f" % f1_score(y_test, y_pred, average='weighted'))

#%%
print('Oversampling with Random Forest:')
oversample_pipe_rf = make_imb_pipeline(RandomOverSampler(), RandomForestClassifier(warm_start=True,
                                                                                   n_estimators=80,
                                                                                   max_depth=9,
                                                                                   max_features = 16,
                                                                                   criterion = 'entropy'))
scores = cross_val_score(oversample_pipe_rf, data_dummies_df, y_train, cv=10, scoring='roc_auc')
print(np.mean(scores))
oversample_pipe_rf.fit(data_dummies_df, y_train)
y_pred = oversample_pipe.predict(holdout_dummies_df)
print('ROC Score: ',roc_auc_score(y_test, y_pred))
print("F1: %1.3f" % f1_score(y_test, y_pred, average='weighted'))
