import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, cross_validate, RepeatedKFold, GridSearchCV
from sklearn.metrics import  roc_auc_score, brier_score_loss, roc_curve, recall_score
from sklearn.preprocessing import MinMaxScaler as scale
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics.scorer import make_scorer
import os

from algorithms.mlp import MLP


# import processed data
data = pd.read_csv("data/temp.csv")
data.pop("Unnamed: 0") # delete useless column
print(data.head())

preop = ['Sex', 'Race1', 'SMO_H', 'DB', 'HCHOL', 'PRECR', 'HYT', 'CBVD', 'PVD', 'LD', 'IE', 'IMSRX', 'MI', 'CHF', 'CHF_C', 'SHOCK', 'RESUS', 'ARRT', 'ARRT_A', 'ARRT_H', 'ARRT_V', 'MEDIN', 'MEDNI', 'MEDAC', 'POP', 'HTM', 'WKG', 'CATH', 'LMD', 'eGFR', 'MIN', 'CPB', 'NRF', 'age', 'STAT_1.0', 'STAT_2.0', 'STAT_3.0', 'STAT_4.0', 'EF_EST_1.0', 'EF_EST_2.0', 'EF_EST_3.0', 'EF_EST_4.0', 'TP_1.0', 'TP_2.0', 'TP_3.0', 'TP_4.0', 'NYHA_1.0', 'NYHA_2.0', 'NYHA_3.0', 'NYHA_4.0', 'CCS_0.0', 'CCS_1.0', 'CCS_2.0', 'CCS_3.0', 'CCS_4.0', 'DISVES_0.0', 'DISVES_1.0', 'DISVES_2.0', 'DISVES_3.0', 'AOPROC_0.0', 'AOPROC_1.0', 'AOPROC_3.0', 'AOPROC_5.0', 'AOPROC_6.0', 'AOPROC_7.0', 'AOPROC_8.0', 'AOPROC_9.0', 'AOPROC_11.0', 'AOPROC_12.0', 'AOPROC_14.0', 'AOPROC_15.0', 'AOPROC_16.0', 'AOPROC_17.0', 'AOPROC_18.0', 'AOPROC_19.0', 'AOPROC_20.0', 'AOPROC_22.0', 'AOPROC_24.0', 'AOPROC_99.0', 'MIPROC_0.0', 'MIPROC_2.0', 'MIPROC_3.0', 'MIPROC_4.0', 'MIPROC_5.0', 'DB_CON_1.0', 'DB_CON_2.0', 'DB_CON_3.0', 'DB_CON_4.0', 'LD_T_0.0', 'LD_T_1.0', 'LD_T_2.0', 'LD_T_3.0', 'LD_T_4.0', 'IABP_W_1.0']
pre = data[preop]
y = pre.pop("NRF")

ros = RandomOverSampler(random_state=1)
scaler = scale()
scaler.fit(pre)

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
                trainingIdx = np.concatenate(splits[:idx] + splits[idx+1:])
                Xidx_r, y_r = ros.fit_resample(trainingIdx.reshape((-1,1)), y[trainingIdx.astype(int)])
                train.append(Xidx_r.flatten())
                test.append(splits[idx])
        return list(zip(train, test))

rkf_search = RepeatedKFold(n_splits=5, n_repeats=1)
rkf = RepeatedKFold(n_splits=5, n_repeats=2)

metrics = {'auc': make_scorer(roc_auc_score, needs_threshold=True),
           'brier':make_scorer(brier_score_loss),
           'sens': make_scorer(recall_score, pos_label=1),
           'spec': make_scorer(recall_score, pos_label=0)} # what performance metrics to measure


# print("preop")
# param = {'layers':[2,5], 'nodes':[5,10], 'dropout':[0.4,0.8], 'epochs':[50]}
# gsearch = GridSearchCV(estimator = MLP(),
#                       param_grid = param, scoring='roc_auc', iid=False, cv=rkf_search, verbose=2)
#
#
# gsearch.fit(scaler.transform(pre.values), y.values)
# clf = gsearch.best_estimator_
# pd.DataFrame(gsearch.cv_results_).to_csv("output/MLP.csv")
#
# output = cross_validate(clf, scaler.transform(pre.values), y.values, scoring=metrics,cv=rkf, verbose=2,return_train_score=True)
#
# pd.DataFrame(output).to_csv('output/performanceMLP.csv')

#### FULL DATASET
# create full training and test data
data = pd.read_csv("data/temp.csv")
data.pop("Unnamed: 0") # delete useless column
print(data.head())

y = data.pop("NRF")

scaler = scale()
scaler.fit(data)


print("full")
param = {'layers':[2,5], 'nodes':[5,10], 'dropout':[0.4,0.8], 'epochs':[50]}
gsearch = GridSearchCV(estimator = MLP(),
                      param_grid = param, scoring='roc_auc', iid=False, cv=rkf_search, verbose=2)


gsearch.fit(scaler.transform(data.values), y.values)
clf = gsearch.best_estimator_
pd.DataFrame(gsearch.cv_results_).to_csv("output/MLPfull.csv")

output = cross_validate(clf, scaler.transform(data.values), y.values, scoring=metrics,cv=rkf, verbose=2,return_train_score=True)
pd.DataFrame(output).to_csv('output/performanceMLPfull.csv')
