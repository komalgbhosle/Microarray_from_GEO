
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import copy, os, operator, collections, zipfile, math, ctypes, cProfile, glob
from itertools import combinations, repeat
from functools import reduce
from pathlib import Path
#import primaryFunctions as pf
#Machine Learning
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
#Statistics
import scipy.stats as stats
import statistics
#Multiprocessing
from multiprocessing import Pool
import time


# In[ ]:


def getpvalue4each(x, y):
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    return stats.ttest_ind(x, y).pvalue
#
def getlogfc4each(x, y):
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    return statistics.mean(y)-statistics.mean(x)
#
def getGPScore4each(row, minpVal):
    if(row['pvalue'] == 0):
        return row['abslogfc']*row['FScore']*abs(np.log(row['pvalue']+minpVal))
    else:
        return row['abslogfc']*row['FScore']*abs(np.log(row['pvalue']))
#
def getGroupedData(ExpData, SampInfo, col4Segregation, dict_GroupLabels):
    dict_ExpData = dict()
    print(SampInfo.head())
    for key in dict_GroupLabels:
        list_tmp_Col = ["Gene"]+list(SampInfo[SampInfo[col4Segregation]==key]['SampleID'])
        df_tmp_ExpData = ExpData[list_tmp_Col]
        df_tmp_ExpData.set_index(["Gene"], inplace=True)
        df_tmp_ExpData = df_tmp_ExpData.T
        dict_ExpData[key] = copy.deepcopy(df_tmp_ExpData)
    #
    return dict_ExpData
#
def prepareData4FS(ExpData, SampInfo, col4Segregation):
    SampleName = list(SampInfo['SampleID'])
    Labels = list(SampInfo[col4Segregation])
    #
    tmp_Labels = list()
    tmp_ExpData = pd.DataFrame()
    tmp_ExpData["Gene"] = ExpData["Gene"]
    #
    for i, sample in enumerate(SampleName):
        tmp_ExpData[sample] = ExpData[sample]
        tmp_Labels.append(Labels[i])
    #
    tmp_ExpData.set_index("Gene", inplace=True)
    tmp_ExpData = tmp_ExpData.T
    X = tmp_ExpData.astype(float).values
    Y = copy.deepcopy(tmp_Labels)
    featuresName = list(tmp_ExpData.columns)
    return X, Y, featuresName
#
def getImportantFeatures(ExpData, SampInfo, col4Segregation):
    #
    X, Y, featuresName = prepareData4FS(ExpData, SampInfo, col4Segregation)
    #
    rf = RandomForestClassifier(n_estimators=100,n_jobs=-1) 
    xgboost = XGBClassifier(nthread=30)
    ## Fit the model on your training data.
    df_featImp = pd.DataFrame(data = featuresName, columns=['Gene'])
    for i in range(20):
        xgboost.fit(X, Y)
        df_featImp['importance_XGB_'+str(i)] = xgboost.feature_importances_
        rf.fit(X, Y)
        df_featImp['importance_RF_'+str(i)] = rf.feature_importances_
    #
    df_featImp.set_index("Gene")
    df_featImp['FScore'] = df_featImp.mean(axis = 1, skipna = True)
    df_featImp = df_featImp.sort_values(by='FScore', ascending=False)
    df_featImp = df_featImp.loc[df_featImp['FScore'] > 0]
    meanVal = df_featImp['FScore'].quantile(.75)
    df_featImp = df_featImp.loc[df_featImp['FScore'] > meanVal]
    df_featImp.drop(df_featImp.columns.difference(['Gene','FScore']), 1, inplace=True)
    return df_featImp
#
def getVal4GP(ExpData, SampInfo, col4Segregation, dict_GroupLabels):
    #
    dict_ExpData = getGroupedData(ExpData, SampInfo, col4Segregation, dict_GroupLabels)
    #
    initialCondition_ExpData = pd.DataFrame()
    finalCondition_ExpData = pd.DataFrame()
    for key in dict_GroupLabels:
        if(dict_GroupLabels[key] == 0):
            finalCondition_ExpData = copy.deepcopy(dict_ExpData[key])
        elif(dict_GroupLabels[key] == 1):
            initialCondition_ExpData = copy.deepcopy(dict_ExpData[key])
    #
    df_val4GP = pd.DataFrame()
    df_val4GP["Gene"] = ExpData["Gene"]
    df_val4GP["pvalue"] = df_val4GP.apply(lambda row: getpvalue4each(initialCondition_ExpData[row["Gene"]],finalCondition_ExpData[row["Gene"]]), axis=1)
    df_val4GP = df_val4GP[df_val4GP['pvalue']<= 0.05]
    #
    initialCondition_ExpData = initialCondition_ExpData[list(df_val4GP["Gene"])]
    finalCondition_ExpData = finalCondition_ExpData[list(df_val4GP["Gene"])]
    #
    df_val4GP["logfc"] = df_val4GP.apply(lambda row: getlogfc4each(initialCondition_ExpData[row["Gene"]],finalCondition_ExpData[row["Gene"]]), axis=1)
    df_val4GP["abslogfc"] = df_val4GP["logfc"].abs()
    #
    tmp_df_FS = getImportantFeatures(ExpData, SampInfo, col4Segregation)
    df_val4GP = pd.merge(df_val4GP, tmp_df_FS, on="Gene")
    #
    minpvalue = min(list(filter((0).__ne__, list(df_val4GP['pvalue']))))
    df_val4GP["GP_Score"] = df_val4GP.apply(lambda row: getGPScore4each(row, minpvalue),axis=1)
    return df_val4GP
#


# In[ ]:


if __name__ == '__main__':
    ####For loop Added for each GSE ids############################################
    # unique_gse_ids = pd.read_csv("GSE_ids.csv")
    # unique_gse = unique_gse_ids.values
    # print(unique_gse)
    with open("/data/users-workspace/ruchika.sharma/Alzheimers/Microarray/unique.csv", "r") as f:
        ids = f.read()
    #print(ids)
    for gse_ids in ids.split("\n"):
        print(gse_ids)
        try:
            expression_data=pd.read_csv(f"/data/users-workspace/ruchika.sharma/Alzheimers/Microarray/{gse_ids}_ExpData.csv")
            sample_data=pd.read_csv(f"/data/users-workspace/ruchika.sharma/Alzheimers/Microarray/{gse_ids}_Sample_info.csv")
            #
            dict_GroupLabels = {'Control':1,'Knockout':0}#Reference will always have 1
            #
            #print("Going into function")
            df_deg = getVal4GP(ExpData = expression_data,
                               SampInfo = sample_data,
                               col4Segregation = "Condition",
                               dict_GroupLabels = dict_GroupLabels)
            #
            #print("Came out")
            #print(df_deg.head())
            df_deg.to_csv(f"data/users-workspace/ruchika.sharma/Alzheimers/Microarray/{gse_ids}_degs.csv",index=False)
        except Exception as e:
            print(gse_ids, "error 404")

        


