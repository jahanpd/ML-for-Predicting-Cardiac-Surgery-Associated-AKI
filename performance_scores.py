import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import  roc_auc_score, brier_score_loss, classification_report, roc_curve
from sklearn.preprocessing import MinMaxScaler as scale
from imblearn.over_sampling import RandomOverSampler

# algorithms
from auxillary.scores import Ng_post as ng_post
from auxillary.scores import Ng_pre as ng_pre
from auxillary.scores import thakar


# import processed data
data = pd.read_csv("output/temp.csv")
data.pop("Unnamed: 0") # delete useless column
print(data.head())

preop = ['Sex', 'Race1', 'SMO_H', 'DB', 'HCHOL', 'PRECR', 'HYT', 'CBVD', 'PVD', 'LD', 'IE', 'IMSRX', 'MI', 'CHF', 'CHF_C', 'SHOCK', 'RESUS', 'ARRT', 'ARRT_A', 'ARRT_H', 'ARRT_V', 'MEDIN', 'MEDNI', 'MEDAC', 'POP', 'HTM', 'WKG', 'CATH', 'LMD', 'eGFR', 'MIN', 'CPB', 'NRF', 'age', 'STAT_1.0', 'STAT_2.0', 'STAT_3.0', 'STAT_4.0', 'EF_EST_1.0', 'EF_EST_2.0', 'EF_EST_3.0', 'EF_EST_4.0', 'TP_1.0', 'TP_2.0', 'TP_3.0', 'TP_4.0', 'NYHA_1.0', 'NYHA_2.0', 'NYHA_3.0', 'NYHA_4.0', 'CCS_0.0', 'CCS_1.0', 'CCS_2.0', 'CCS_3.0', 'CCS_4.0', 'DISVES_0.0', 'DISVES_1.0', 'DISVES_2.0', 'DISVES_3.0', 'AOPROC_0.0', 'AOPROC_1.0', 'AOPROC_3.0', 'AOPROC_5.0', 'AOPROC_6.0', 'AOPROC_7.0', 'AOPROC_8.0', 'AOPROC_9.0', 'AOPROC_11.0', 'AOPROC_12.0', 'AOPROC_14.0', 'AOPROC_15.0', 'AOPROC_16.0', 'AOPROC_17.0', 'AOPROC_18.0', 'AOPROC_19.0', 'AOPROC_20.0', 'AOPROC_22.0', 'AOPROC_24.0', 'AOPROC_99.0', 'MIPROC_0.0', 'MIPROC_2.0', 'MIPROC_3.0', 'MIPROC_4.0', 'MIPROC_5.0', 'DB_CON_1.0', 'DB_CON_2.0', 'DB_CON_3.0', 'DB_CON_4.0', 'LD_T_0.0', 'LD_T_1.0', 'LD_T_2.0', 'LD_T_3.0', 'LD_T_4.0', 'IABP_W_1.0']
pre = data[preop]
y = pre.pop("NRF")

ros = RandomOverSampler(random_state=1)
X_train, X_test, y_train, y_test = train_test_split(pre, y, test_size = 0.33, random_state=1)
X_train, y_train = ros.fit_resample(X_train, y_train)
scaler = scale()
scaler.fit(X_train)
combined = np.append(X_test.values, y_test.values.reshape((-1,1)), axis=1)

print("NG Pre")
aucs = np.array(())
briers = np.array(())
sens = np.array(())
spec = np.array(())
for n in range(32):
    # split into test and train data and normalise
    idx = np.random.randint(len(y_test),size=len(y_test))
    scores = [ng_pre(pd.DataFrame(combined[n,:-1].reshape((1,-1)),columns=list(pre))) for n in idx]
    scores = np.array(scores).reshape((-1,1))
    scorescaler = scale()
    scorescaler.fit(scores)
    scores = scorescaler.transform(scores)

    fpr, tpr, threshold = roc_curve(combined[idx,-1].reshape((-1,1)), scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    labellize = lambda x: 1 if x >= optimal_threshold else 0
    vfunc = np.vectorize(labellize)
    auc = roc_auc_score(combined[idx,-1].reshape((-1,1)), scores)
    brier = brier_score_loss(combined[idx,-1].reshape((-1,1)),vfunc(scores))
    report = classification_report(combined[idx,-1].reshape((-1,1)), vfunc(scores), output_dict=True)

    aucs = np.append(aucs, auc)
    briers = np.append(briers, brier)
    sens = np.append(sens, report["1.0"]['recall'])
    spec = np.append(spec, report["0.0"]['recall'])

np.savez_compressed('output/performanceNGpre',
                    auc=aucs, briers=briers, sens=sens, spec=spec)

print("Thakar Pre")
aucs = np.array(())
briers = np.array(())
sens = np.array(())
spec = np.array(())
for n in range(32):
    # split into test and train data and normalise
    idx = np.random.randint(len(y_test),size=len(y_test))
    scores = [thakar(pd.DataFrame(combined[n,:-1].reshape((1,-1)),columns=list(pre))) for n in idx]
    scores = np.array(scores).reshape((-1,1))
    scorescaler = scale()
    scorescaler.fit(scores)
    scorescaler = scale()
    scorescaler.fit(scores)
    scores = scorescaler.transform(scores)

    fpr, tpr, threshold = roc_curve(combined[idx,-1].reshape((-1,1)), scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    print(optimal_threshold)
    labellize = lambda x: 1 if x >= optimal_threshold else 0
    vfunc = np.vectorize(labellize)
    auc = roc_auc_score(combined[idx,-1].reshape((-1,1)), scores)
    brier = brier_score_loss(combined[idx,-1].reshape((-1,1)),vfunc(scores))
    report = classification_report(combined[idx,-1].reshape((-1,1)), vfunc(scores), output_dict=True)

    aucs = np.append(aucs, auc)
    briers = np.append(briers, brier)
    sens = np.append(sens, report["1.0"]['recall'])
    spec = np.append(spec, report["0.0"]['recall'])

np.savez_compressed('output/performanceThakpre',
                    auc=aucs, briers=briers, sens=sens, spec=spec)

#### FULL DATASET
# create full training and test data
y = data.pop("NRF")

ros = RandomOverSampler(random_state=1)

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size = 0.33, random_state=1)

X_train, y_train = ros.fit_resample(X_train, y_train)

scaler = scale()
scaler.fit(X_train)
combined = np.append(X_test.values, y_test.values.reshape((-1,1)), axis=1)

print("NG Post")
aucs = np.array(())
briers = np.array(())
sens = np.array(())
spec = np.array(())
for n in range(32):
    # split into test and train data and normalise
    idx = np.random.randint(len(y_test),size=len(y_test))
    scores = [ng_post(pd.DataFrame(combined[n,:-1].reshape((1,-1)),columns=list(data))) for n in idx]
    scores = np.array(scores).reshape((-1,1))
    scorescaler = scale()
    scorescaler.fit(scores)
    scorescaler = scale()
    scorescaler.fit(scores)
    scores = scorescaler.transform(scores)

    fpr, tpr, threshold = roc_curve(combined[idx,-1].reshape((-1,1)).reshape((-1,1)), scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    print(optimal_threshold)
    labellize = lambda x: 1 if x >= optimal_threshold else 0
    vfunc = np.vectorize(labellize)
    auc = roc_auc_score(combined[idx,-1].reshape((-1,1)), scores)
    brier = brier_score_loss(combined[idx,-1].reshape((-1,1)),vfunc(scores))
    report = classification_report(combined[idx,-1].reshape((-1,1)), vfunc(scores), output_dict=True)

    aucs = np.append(aucs, auc)
    briers = np.append(briers, brier)
    sens = np.append(sens, report["1.0"]['recall'])
    spec = np.append(spec, report["0.0"]['recall'])

np.savez_compressed('output/performanceNGpost',
                    auc=aucs, briers=briers, sens=sens, spec=spec)
