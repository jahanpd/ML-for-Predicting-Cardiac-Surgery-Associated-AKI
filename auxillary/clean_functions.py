import numpy as np
import pandas as pd
from scipy import stats

# routine to combine EF and EF_EST
def ejectionFraction(dataframe):
    estimate = dataframe["EF_EST"].values
    fraction = dataframe["EF"].values
    combinat = np.empty((len(estimate)))
    for index in range(len(dataframe)):
        if (np.isnan(estimate[index]) and np.isnan(fraction[index])):
            combinat[index] == np.nan
        if not np.isnan(estimate[index]) and  not np.isnan(fraction[index]):
            combinat[index]=estimate[index]
        if np.isnan(estimate[index]) and not np.isnan(fraction[index]):
            if fraction[index] < 30.0:
                combinat[index] = 4
            if 30 <= fraction[index] and fraction[index] < 45.0:
                combinat[index] = 3
            if 45 <= fraction[index] and fraction[index] <= 60.0:
                combinat[index] = 2
            if fraction[index] > 60.0:
                combinat[index] = 1
        if not np.isnan(estimate[index]) and np.isnan(fraction[index]):
            combinat[index]=estimate[index]
    dataframe["EF_EST"] = combinat
    dataframe["EF_EST"] = dataframe.EF_EST.apply(lambda x: np.nan if x not in [1,2,3,4] else x)
    # dataframe.EF_EST = dataframe.EF_EST.astype(pd.Int64Dtype())
    dataframe.pop('EF')
    return dataframe

# convert DOB to age
from datetime import date, datetime, timedelta

def age(dataframe):
    def calculate_age(born, admission):
        return admission.year - born.year - ((admission.month, admission.day) < (born.month, born.day))
    # convert data to datetime object
    dataframe.DOB = pd.to_datetime(dataframe.DOB, format="%m/%d/%Y")
    dataframe.DOA = pd.to_datetime(dataframe.DOA, format="%m/%d/%Y")
    ages = np.empty((len(dataframe)))
    for n in range(len(dataframe)):
        ages[n] = calculate_age(dataframe.DOB[n],dataframe.DOA[n])
    dataframe["age"] = ages
    dataframe.pop('DOB')
    return dataframe

def bias(dataset, cat_vars, num_vars):

    # Check if excluding NAs introduces bias to ech variable
    dataDropped = dataset.dropna()
    char = {}
    boolean=False
    for feat in cat_vars:
        prob = np.sum(dataset[feat].dropna().values)/len(dataset[feat].dropna().values)
        pvalue = stats.binom_test(x=int(np.sum(dataDropped[feat].values)),
                                  n=len(dataDropped[feat].values),
                                  p=prob)
        char.update({feat:[pvalue]})
        if pvalue < 0.05:
            boolean = True

    # pvalue for continuous variable
    for feat in num_vars:
        D, pvalue = stats.ks_2samp(dataset[feat].dropna().values,
                                    dataDropped[feat].values)
        char.update({feat:[pvalue]})
        if pvalue < 0.05:
            boolean = True

    return (pd.DataFrame(char, index=['P-value'])), boolean
