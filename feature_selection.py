#
import math, copy, time, random
#
import pandas as pd
import numpy as np
#
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
#
from utilities import utils
from utilities.fs_grid import sklearnhelper, parameteroptimization
#
#
#
def build_fs_data(expression_file_path=None,exp_filetype="csv",metadata_file_path=None,meta_filetype="csv",marker_column_name="Gene",condition_column_name="Condition",id_column_name="SampleID",final_point=["Crohns Disease"],initial_point=["Healthy"]):
    #
    data = utils.read_df(expression_file_path,exp_filetype)
    metadata = utils.read_df(metadata_file_path,meta_filetype)
    data, metadata = utils.check_exp_meta_data(data, metadata, marker_column_name, id_column_name)
    data = data.set_index(marker_column_name).T
    #
    feature_name = list(data.columns)
    sample_ids = list(data.index)
    #
    category = dict()
    for i,label in enumerate(metadata[condition_column_name].unique()):
        if label in final_point:
            category[label] = 1
        elif label in initial_point:
            category[label] = 0
    #category = {label:i for i,label in enumerate(metadata[condition_column_name].unique())}
    dict_metadata = {row[id_column_name]:row[condition_column_name] for i,row in metadata.iterrows()}
    #
    x = data.values
    y = [category[dict_metadata[sample]] for sample in sample_ids]
    #
    return {"feature":feature_name,"sample":sample_ids,"x":x,"y":y}
#
#
def feature_scoring(expression_file_path=None,exp_filetype="csv",metadata_file_path=None,meta_filetype="csv",marker_column_name="Gene",condition_column_name="Condition",id_column_name="SampleID",final_point=["Crohns Disease"],initial_point=["Healthy"]):
    data = build_fs_data(expression_file_path,"csv",
                         metadata_file_path,"csv",
                         marker_column_name,condition_column_name,id_column_name,final_point=final_point,initial_point=initial_point)
    #
    x_train, x_test, y_train, y_test = train_test_split(data["x"],data["y"], test_size=0.2, random_state=8)
    clf_init = parameteroptimization("RandomForest")
    clf_name = clf_init.clf
    clf_paramGrid = clf_init.paramGrid
    clf_obj = sklearnhelper(clf_name)
    #
    grid_obj = clf_obj.RandomizedSearchCV(clf_paramGrid, x_train, y_train, 150)
    clf_obj = sklearnhelper(clf_name,grid_obj.best_params_)
    clf_model = clf_obj.fit(x_train,y_train)
    fs_scoring = {feature:score for feature,score in zip(data['feature'],clf_model.feature_importances_)}
    precision, recall, f1_score = precision_recall_fscore_support(y_test, clf_model.predict(x_test), average="weighted")[:3]
    print("\nPrecision = {}\nRecall = {}\nF1-Score = {}\n".format(precision, recall, f1_score))
    return {"feature_score":fs_scoring,"precision":precision,"recall":recall,"f1_score":f1_score}
#
#
if __name__ == '__main__':
    print("Deep Dive Dear!!")