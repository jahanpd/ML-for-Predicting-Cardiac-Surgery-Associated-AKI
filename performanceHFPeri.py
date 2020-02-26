import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, cross_validate,RepeatedKFold, GridSearchCV
from sklearn.metrics import  roc_auc_score, brier_score_loss, roc_curve, recall_score
from sklearn.preprocessing import MinMaxScaler as scale
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics.scorer import make_scorer
import os

# algorithms
from algorithms.mlp import MLP
from algorithms.autoencoder import autoencoder
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier

ros = RandomOverSampler(random_state=1)

class oversampled_Kfold():
    def __init__(self, n_splits, n_repeats=1):
        self.n_splits = n_splits
        self.n_repeats = n_repeats

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits*self.n_repeats

    def split(self, X, y, groups=None):
        splits = np.array_split(np.random.choice(len(X), len(X),replace=False), self.n_splits)
        train, test = [], []
        for repeat in range(self.n_repeats):
            for idx in range(len(splits)):
                test_array = splits[idx]
                trainingIdx = np.concatenate(splits[:idx] + splits[idx+1:])
                Xidx_r, y_r = ros.fit_resample(trainingIdx.reshape((-1,1)), y[trainingIdx.astype(int)])
                train.append(Xidx_r.flatten())
                test.append(splits[idx])
        return list(zip(train, test))

rkf_search = oversampled_Kfold(n_splits=5, n_repeats=1)
rkf = oversampled_Kfold(n_splits=5, n_repeats=4)

metrics = {'auc': make_scorer(roc_auc_score, needs_threshold=True),
           'brier':make_scorer(brier_score_loss),
           'sens': make_scorer(recall_score, pos_label=1),
           'spec': make_scorer(recall_score, pos_label=0)} # what performance metrics to measure

#### FULL DATASET
# create full training and test data
data = pd.read_csv("data/tempHF.csv")
data.pop("Unnamed: 0") # delete useless column
print(data.head())

if "NRF" in list(data):
    data.pop("NRF")
if "POSTCR" in list(data):
    data.pop("POSTCR")
if "OpID" in list(data):
    data.pop("OpID")
if "PatID" in list(data):
    data.pop("PatID")
if "DOA" in list(data):
    data.pop("DOA")

y = data.pop("HAEMOFIL")

ros = RandomOverSampler(random_state=1)

scaler = scale()
scaler.fit(data)

# print("SVM")
# param = {'C':[0.01,0.1,1,10,100]}
# gsearch = GridSearchCV(estimator = LinearSVC(),
#                       param_grid = param, scoring='roc_auc',iid=False, cv=rkf_search, verbose=2)
#
# gsearch.fit(scaler.transform(data.values), y.values)
# clf = gsearch.best_estimator_
# pd.DataFrame(gsearch.cv_results_).to_csv("output/HF/SVMfull.csv")
#
# output = cross_validate(clf, scaler.transform(data.values), y.values, scoring=metrics,cv=rkf,  verbose=2,return_train_score=True)
# pd.DataFrame(output).to_csv('output/HF/performanceSVMfull.csv')
#
#
# print("Logistic Regression")
# param = {'C':[0.01,0.1,1,10,100]}
# gsearch = GridSearchCV(estimator = LogisticRegression(),
#                       param_grid = param, scoring='roc_auc',iid=False, cv=rkf_search, verbose=2)
#
# gsearch.fit(scaler.transform(data.values), y.values)
# clf = gsearch.best_estimator_
# pd.DataFrame(gsearch.cv_results_).to_csv("output/HF/LRfull.csv")
#
# output = cross_validate(clf, scaler.transform(data.values), y.values, scoring=metrics,cv=rkf,  verbose=2,return_train_score=True)
# pd.DataFrame(output).to_csv('output/HF/performanceLRfull.csv')
#


print("Random Forest")
param = {'n_estimators':[5,10,50],
         'min_impurity_decrease':[0.1,0.01,0.001]
         }
gsearch = GridSearchCV(estimator = RandomForestClassifier(),
                      param_grid = param, scoring='roc_auc',iid=False, cv=rkf_search, verbose=2)

gsearch.fit(data.values, y.values)
clf = gsearch.best_estimator_
pd.DataFrame(gsearch.cv_results_).to_csv("output/HF/RFfull.csv")

output = cross_validate(clf, data.values, y.values, scoring=metrics,cv=rkf,  verbose=2,return_train_score=True)
pd.DataFrame(output).to_csv('output/HF/performanceRFfull.csv')

print("Decision Tree")
param = {
         'max_depth':[5,10,50],
         'min_impurity_decrease':[0.1,0.01,0.001]
         }

gsearch = GridSearchCV(estimator = DecisionTreeClassifier(),
                      param_grid = param, scoring='roc_auc',iid=False, cv=rkf_search, verbose=2)

gsearch.fit(data.values, y.values)
clf = gsearch.best_estimator_
pd.DataFrame(gsearch.cv_results_).to_csv("output/HF/DTfull.csv")

output = cross_validate(clf, data.values, y.values, scoring=metrics,cv=rkf,  verbose=2,return_train_score=True)
pd.DataFrame(output).to_csv('output/HF/performanceDTfull.csv')


print("Boosting")
param = {'n_estimators':[5,10,50],
         'gamma':[0.01,0.1,1]
         }


gsearch = GridSearchCV(estimator = XGBClassifier(),
                      param_grid = param, scoring='roc_auc',iid=False, cv=rkf_search, verbose=2)
gsearch.fit(data.values, y.values)
clf = gsearch.best_estimator_
pd.DataFrame(gsearch.cv_results_).to_csv("output/HF/GBfull.csv")

output = cross_validate(clf, data.values, y.values, scoring=metrics,cv=rkf,  verbose=2,return_train_score=True)
pd.DataFrame(output).to_csv('output/HF/performanceXGBfull.csv')


print("KNN")
param = {'n_neighbors':[2,5],
         'algorithm':['ball_tree'],
         'leaf_size':[5,10,20]
         }

gsearch = GridSearchCV(estimator = KNeighborsClassifier(),
                      param_grid = param, scoring='roc_auc',iid=False, cv=rkf_search, verbose=2)
gsearch.fit(data.values, y.values)
clf = gsearch.best_estimator_
pd.DataFrame(gsearch.cv_results_).to_csv("output/HF/KNNfull.csv")

output = cross_validate(clf, data.values, y.values, scoring=metrics, cv=rkf,  verbose=2,return_train_score=True)
pd.DataFrame(output).to_csv('output/HF/performanceKNNfull.csv')
