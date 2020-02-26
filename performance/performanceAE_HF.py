import numpy as np
import pandas as pd
import tensorflow as tf
from algorithms.autoencoder import autoencoder
from sklearn.model_selection import train_test_split, cross_validate, RepeatedKFold, GridSearchCV
from sklearn.metrics import  roc_auc_score, brier_score_loss, roc_curve, recall_score
from sklearn.preprocessing import MinMaxScaler as scale
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics.scorer import make_scorer
import gc
import yaml


# import processed data
data = pd.read_csv("data/tempHF.csv")
data.pop("Unnamed: 0") # delete useless column
print(data.head())

preop = ['Sex', 'Race1', 'SMO_H', 'DB', 'HCHOL', 'PRECR', 'HYT', 'CBVD', 'PVD', 'LD', 'IE', 'IMSRX', 'MI', 'CHF', 'CHF_C', 'SHOCK', 'RESUS', 'ARRT', 'ARRT_A', 'ARRT_H', 'ARRT_V', 'MEDIN', 'MEDNI', 'MEDAC', 'POP', 'HTM', 'WKG', 'CATH', 'LMD', 'eGFR', 'MIN', 'CPB', 'HAEMOFIL', 'age', 'STAT_1.0', 'STAT_2.0', 'STAT_3.0', 'STAT_4.0', 'EF_EST_1.0', 'EF_EST_2.0', 'EF_EST_3.0', 'EF_EST_4.0', 'TP_1.0', 'TP_2.0', 'TP_3.0', 'TP_4.0', 'NYHA_1.0', 'NYHA_2.0', 'NYHA_3.0', 'NYHA_4.0', 'CCS_0.0', 'CCS_1.0', 'CCS_2.0', 'CCS_3.0', 'CCS_4.0', 'DISVES_0.0', 'DISVES_1.0', 'DISVES_2.0', 'DISVES_3.0', 'AOPROC_0.0', 'AOPROC_1.0', 'AOPROC_3.0', 'AOPROC_5.0', 'AOPROC_6.0', 'AOPROC_7.0', 'AOPROC_8.0', 'AOPROC_9.0', 'AOPROC_11.0', 'AOPROC_12.0', 'AOPROC_14.0', 'AOPROC_15.0', 'AOPROC_16.0', 'AOPROC_17.0', 'AOPROC_18.0', 'AOPROC_19.0', 'AOPROC_20.0', 'AOPROC_22.0', 'AOPROC_24.0', 'AOPROC_99.0', 'MIPROC_0.0', 'MIPROC_2.0', 'MIPROC_3.0', 'MIPROC_4.0', 'MIPROC_5.0', 'DB_CON_1.0', 'DB_CON_2.0', 'DB_CON_3.0', 'DB_CON_4.0', 'LD_T_0.0', 'LD_T_1.0', 'LD_T_2.0', 'LD_T_3.0', 'LD_T_4.0', 'IABP_W_1.0']
pre = data[preop]

# TENSORFLOW SETTINGS
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.7
sess = tf.Session(config=config)

np.random.seed(1)
ros = RandomOverSampler(random_state=1)

class oversampled_Kfold():
    def __init__(self, n_splits, n_repeats=1):
        self.n_splits = n_splits
        self.n_repeats = n_repeats

    def get_n_splits(self, X=1, y=1, groups=None):
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

rkf_search = oversampled_Kfold(n_splits=10, n_repeats=1)
rkf = oversampled_Kfold(n_splits=10, n_repeats=4)

metrics = {'auc': make_scorer(roc_auc_score, needs_threshold=True),
           'brier':make_scorer(brier_score_loss),
           'sens': make_scorer(recall_score, pos_label=1),
           'spec': make_scorer(recall_score, pos_label=0)} # what performance metrics to measure


###### PREOP ONLY DATASET #####
# create dataset of [x_samples, x_sameClass, x_diffClass]
#
# noaki = pre[pre.HAEMOFIL == 0]
# noaki.pop("HAEMOFIL")
# aki = pre[pre.HAEMOFIL == 1]
# aki.pop("HAEMOFIL")
#
# indexes_switch = np.random.choice(len(noaki), int(len(noaki)/2), replace=False)
#
# x_a = pd.concat([noaki,aki]).values
# x_b = pd.concat([noaki.sample(n=len(noaki),replace=True),aki.sample(n=len(aki),replace=True)]).values
# x_c = pd.concat([aki.sample(n=len(noaki),replace=True), noaki.sample(n=len(aki),replace=True)]).values
#
# y = np.full((len(x_a),1),1)
# y[len(noaki):] = np.full((len(aki),1),0)
#
# X = np.concatenate((x_a,x_b,x_c, y), axis=1)
# np.random.shuffle(X)
# y = X[:,-1]
# X = X[:,:-1]
# param = {'dim_red':[2,4,8], 'dropout':[0.4,0.8], 'epochs':[50]}


### design a custom grid search, output of algorithm  is either correct or incorrect class

# cv_results = {'params':[],'aucs':[], 'auc':np.array(())}
# save_model = 0
# auc_temp = 0
# for par_one in param['dim_red']:
#     for par_two in param['dropout']:
#         splits = rkf_search.split(X=X,y=y) # generate splits with ROS
#         aucs = np.array(())
#         for fold in range(rkf_search.get_n_splits()):
#             print({'dim_red':par_one, 'dropout':par_two})
#             ae = autoencoder(dropout=par_two, dim_red=par_one) # initialise algo
#             ae.fit(X=X[splits[fold][0]],y=y[splits[fold][0]]) # train algorithm on split data
#             prob_correct = ae.predict_proba(X=X[splits[fold][1]])
#             prob_aki = np.array([x if y == 1 else 1-x for x,y in zip(prob_correct.flatten(), y[splits[fold][1]])])
#             auc = roc_auc_score(y[splits[fold][1]].reshape((-1,1)), prob_aki.reshape((-1,1)))
#             print(auc)
#             aucs = np.append(aucs, auc)
#             if auc > auc_temp:
#                 save_model = ae
#                 auc_temp = auc
#         print(aucs.mean())
#         cv_results['params'].append({'dim_red':par_one, 'dropout':par_two})
#         cv_results['auc'] = np.append(cv_results['auc'] ,aucs.mean())
#         cv_results['aucs'].append({'folds':aucs})
#
#
# pd.DataFrame(cv_results).to_csv("output/AE_HF.csv")
#
# print("save_model auc:", auc_temp)
# save_model.epochs = 1
#
# cv_results = pd.read_csv('output/AE.csv')
# params = yaml.load(cv_results['params'][np.argmax(cv_results['auc'].values)])
# print("params:",params)
# aucs = np.array(())
# briers = np.array(())
# sens = np.array(())
# spec = np.array(())
# splits = rkf.split(X=X,y=y) # generate splits with ROS
# for fold in range(rkf.get_n_splits()):
#     save_model = autoencoder(dropout=params['dropout'], dim_red=params['dim_red']) # initialise algo
#     save_model.fit(X=X[splits[fold][0]],y=y[splits[fold][0]]) # train algorithm on split data
#     class_correct = save_model.predict(X=X[splits[fold][1]])
#     prob_correct = save_model.predict_proba(X=X[splits[fold][1]])
#
#     prob_aki = np.array([x if y == 1 else 1-x for x,y in zip(prob_correct, y[splits[fold][1]].reshape((-1,1)))])
#     class_aki = np.array([x if y == 1 else 1-x for x,y in zip(class_correct, y[splits[fold][1]].reshape((-1,1)))])
#
#     auc = roc_auc_score(y[splits[fold][1]].reshape((-1,1)), prob_aki.reshape((-1,1)))
#     brier = brier_score_loss(y[splits[fold][1]].reshape((-1,1)), prob_aki.reshape((-1,1)))
#     sen = recall_score(y[splits[fold][1]].reshape((-1,1)),class_aki.reshape((-1,1)), pos_label=1)
#     spe = recall_score(y[splits[fold][1]].reshape((-1,1)),class_aki.reshape((-1,1)), pos_label=0)
#
#     aucs = np.append(aucs, auc)
#     briers = np.append(briers, brier)
#     sens = np.append(sens, sen)
#     spec = np.append(spec, spe)
#
#
# print(aucs.mean(), briers.mean(), sens.mean(), spec.mean())
# final_results = {'auc':aucs,'briers':briers,'sens':sens,'spec':spec}
# pd.DataFrame(final_results).to_csv('output/performanceAE_HF.csv')

### FULL DATASET
# create full training and test data
# import processed data
data = pd.read_csv("data/tempHF.csv")
data.pop("Unnamed: 0") # delete useless column

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

print(data.head())

noaki = data[data.HAEMOFIL == 0]
noaki.pop("HAEMOFIL")
aki = data[data.HAEMOFIL == 1]
aki.pop("HAEMOFIL")

indexes_switch = np.random.choice(len(noaki), int(len(noaki)/2), replace=False)

x_a = pd.concat([noaki,aki]).values
x_b = pd.concat([noaki.sample(n=len(noaki),replace=True),aki.sample(n=len(aki),replace=True)]).values
x_c = pd.concat([aki.sample(n=len(noaki),replace=True), noaki.sample(n=len(aki),replace=True)]).values

y = np.full((len(x_a),1),1)
y[len(noaki):] = np.full((len(aki),1),0)

X = np.concatenate((x_a,x_b,x_c, y), axis=1)
np.random.shuffle(X)
y = X[:,-1]
X = X[:,:-1]

param = {'dim_red':[2,4,8], 'dropout':[0.4,0.8], 'epochs':[50]}

### design a custom grid search, output of algorithm  is either correct or incorrect class

cv_results = {'params':[],'aucs':[], 'auc':np.array(())}
save_model = 0
auc_temp = 0
for par_one in param['dim_red']:
    for par_two in param['dropout']:
        splits = rkf_search.split(X=X,y=y) # generate splits with ROS
        aucs = np.array(())
        for fold in range(rkf_search.get_n_splits()):
            print({'dim_red':par_one, 'dropout':par_two})
            ae = autoencoder(dropout=par_two, dim_red=par_one) # initialise algo
            ae.fit(X=X[splits[fold][0]],y=y[splits[fold][0]]) # train algorithm on split data
            prob_correct = ae.predict_proba(X=X[splits[fold][1]])
            prob_aki = np.array([x if y == 1 else 1-x for x,y in zip(prob_correct.flatten(), y[splits[fold][1]])])
            auc = roc_auc_score(y[splits[fold][1]].reshape((-1,1)), prob_aki)
            print(auc)
            aucs = np.append(aucs, auc)
            if auc > auc_temp:
                save_model = ae
                auc_temp = auc
        print(aucs.mean())
        cv_results['params'].append({'dim_red':par_one, 'dropout':par_two})
        cv_results['auc'] = np.append(cv_results['auc'] ,aucs.mean())
        cv_results['aucs'].append({'folds':aucs})


pd.DataFrame(cv_results).to_csv("output/AEfull_HF.csv")

print("save_model auc:", auc_temp)
save_model.epochs = 1

cv_results = pd.read_csv('output/AE.csv')
params = yaml.load(cv_results['params'][np.argmax(cv_results['auc'].values)])
print("params:",params)
aucs = np.array(())
briers = np.array(())
sens = np.array(())
spec = np.array(())
splits = rkf.split(X=X,y=y) # generate splits with ROS
for fold in range(rkf.get_n_splits()):
    save_model = autoencoder(dropout=params['dropout'], dim_red=params['dim_red']) # initialise algo
    save_model.fit(X=X[splits[fold][0]],y=y[splits[fold][0]]) # train algorithm on split data
    class_correct = save_model.predict(X=X[splits[fold][1]])
    prob_correct = save_model.predict_proba(X=X[splits[fold][1]])

    prob_aki = np.array([x if y == 1 else 1-x for x,y in zip(prob_correct, y[splits[fold][1]].reshape((-1,1)))])
    class_aki = np.array([x if y == 1 else 1-x for x,y in zip(class_correct, y[splits[fold][1]].reshape((-1,1)))])

    auc = roc_auc_score(y[splits[fold][1]].reshape((-1,1)), prob_aki.reshape((-1,1)))
    brier = brier_score_loss(y[splits[fold][1]].reshape((-1,1)), prob_aki.reshape((-1,1)))
    sen = recall_score(y[splits[fold][1]].reshape((-1,1)),class_aki.reshape((-1,1)), pos_label=1)
    spe = recall_score(y[splits[fold][1]].reshape((-1,1)),class_aki.reshape((-1,1)), pos_label=0)

    aucs = np.append(aucs, auc)
    briers = np.append(briers, brier)
    sens = np.append(sens, sen)
    spec = np.append(spec, spe)


print(aucs.mean(), briers.mean(), sens.mean(), spec.mean())
final_results = {'auc':aucs,'briers':briers,'sens':sens,'spec':spec}
pd.DataFrame(final_results).to_csv('output/performanceAEfull_HF.csv')
