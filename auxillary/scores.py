import numpy as np

def Ng_pre(row):
    total = 0
    BMI = row.WKG.values/(np.power(row.HTM.values/100,2))
    if BMI > 30:
        total += 4
    if row.IE.values == 1:
        total += 6
    if row.DB.values == 1:
        total += 2
    if row["TP_2.0"].values == 1:
        total += 3
    if row["TP_3.0"].values == 1:
        total += 4
    if row["TP_4.0"].values == 1:
        total += 6
    if (row.PRECR.values > 70) and (row.PRECR.values <= 100):
        total += 3
    if (row.PRECR.values > 100) and (row.PRECR.values <= 120):
        total += 5
    if (row.PRECR.values > 120) and (row.PRECR.values <= 150):
        total += 8
    if (row.PRECR.values > 150):
        total += 9
    if row["STAT_2.0"].values == 1:
        total += 1
    if row["STAT_3.0"].values == 1:
        total += 6
    if row["STAT_4.0"].values == 1:
        total += 8
    if (row.eGFR.values > 60) and (row.PRECR.values <= 90):
        total += 1
    if (row.eGFR.values > 30) and (row.PRECR.values <= 60):
        total += 3
    if (row.eGFR.values <= 30):
        total += 3
    if row.CHF.values == 1:
        total += 2
    total += np.floor(row.age.values/10)
    if row.SHOCK.values == 1:
        total += 4
    return total

def Ng_post(row):
    total = 0
    BMI = row.WKG.values/(np.power(row.HTM.values/100,2))
    if BMI > 30:
        total += 2
    if row.IE.values == 1:
        total += 6
    if row.DB.values == 1:
        total += 3
    if row["IABP_W_1.0"].values == 1:
        total += 2
    if row["IABP_W_2.0"].values == 1:
        total += 8
    if row["IABP_W_3.0"].values == 1:
        total += 8
    if row.PERF.values > 180:
        total += 3
    if row.NRBC.values == 1:
        total += 2
    if row.Sex.values == 0:
        total += 1
    if ((row.RTT.values == 1) and (row.RBC.values == 1)):
        total += 11
    if row.RBC.values == 1:
        total += 7
    if row.HCHOL.values == 1:
        total -= 1
    if row.HYT.values == 1:
        total += 2
    if row.LD.values == 1:
        total += 2
    if row["TP_2.0"].values == 1:
        total += 3
    if row["TP_3.0"].values == 1:
        total += 2
    if row["TP_4.0"].values == 1:
        total += 5
    if (row.PRECR.values > 70) and (row.PRECR.values <= 100):
        total += 4
    if (row.PRECR.values > 100) and (row.PRECR.values <= 120):
        total += 6
    if (row.PRECR.values > 120) and (row.PRECR.values <= 150):
        total += 9
    if (row.PRECR.values > 150):
        total += 9
    if row["STAT_2.0"].values == 1:
        total += 0
    if row["STAT_3.0"].values == 1:
        total += 4
    if row["STAT_4.0"].values == 1:
        total += 5
    if (row.eGFR.values > 60) and (row.PRECR.values <= 90):
        total += 2
    if (row.eGFR.values > 30) and (row.PRECR.values <= 60):
        total += 3
    if (row.eGFR.values <= 30):
        total += 1
    if row.CHF.values == 1:
        total += 2
    total += np.floor(row.age.values/10)
    if row.SHOCK.values == 1:
        total += 4
    return total

def thakar(row):
    total = 0
    if row.Sex.values == 1:
        total += 1
    if row.CHF.values == 1:
        total += 1
    if row["EF_EST_4.0"].values == 1:
        total += 1
    if row["IABP_W_1.0"].values == 1:
        total += 2
    if row.LD.values == 1:
        total += 1
    if row["DB_CON_4.0"].values == 1:
        total += 1
    if row.POP.values == 1:
        total += 1
    if row["STAT_3.0"].values == 1:
        total += 2
    if row["STAT_4.0"].values == 1:
        total += 2
    if row["TP_2.0"].values == 1:
        total += 1
    if row["TP_3.0"].values == 1:
        total += 2
    if row["TP_4.0"].values == 1:
        total += 2
    if (row.PRECR.values > 106) and (row.PRECR.values < 186):
        total += 2
    if (row.PRECR.values >= 186):
        total += 5
    return total
