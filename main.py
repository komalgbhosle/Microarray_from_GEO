#
import math, copy, time, random, json
#
import pandas as pd
import numpy as np
#
from tqdm import tqdm
from scipy import stats
#
from statistical_tests import statistical_testing
from feature_selection import feature_scoring
from network_simulation import network_score
from utilities import utils
#
#
def power_mean(list_data,m):
    power_mean_internal = 0
    for value in list_data:
        power_mean_internal+=value**m
    return ((1/len(list_data))*power_mean_internal)**(1/m)
#
#
def calculate_confidence(data):
    data["updated_network"] = 1/(data["abs_network_score"]+1)
    data["log_confidence_score"] = -1*np.log(data.apply(lambda row: power_mean([row["mannwhitney_qvalue"],row["welch_ttest_qvalue"],row["student_ttest_qvalue"],row["updated_network"]],0.01),axis=1))
    data["confidence_score"] = data["log_confidence_score"]/data["log_confidence_score"].max()
    data["update_cs"] = [1]*(data.shape[0])
    return data
#
#
def prioritize_marker(expression_file_path=None,exp_filetype="csv",metadata_file_path=None,meta_filetype="csv",marker_column_name="Gene",condition_column_name="Condition",id_column_name="SampleID",final_point=["Crohns Disease"],initial_point=["Healthy"],final_point_name=None,initial_point_name=None,permutation_test=0):
    #
    start_time = time.time()
    result = statistical_testing(expression_file_path = expression_file_path,
                   metadata_file_path = metadata_file_path,
                   marker_column_name = marker_column_name,
                   condition_column_name = condition_column_name,
                   id_column_name = id_column_name,
                   final_point = final_point,
                   initial_point = initial_point,
                   permutation_test = permutation_test,
                   final_point_name=final_point_name,
                   initial_point_name=initial_point_name)
    print("Time taken to perform statistical testing  = ", time.time()-start_time)
    #
    start_time = time.time()
    result = network_score(dict_data = result,feature_weight_key = "logfc")
    print("Time taken to calculate network score  = ", time.time()-start_time)
    #
    start_time = time.time()
    result_fs = feature_scoring(expression_file_path = expression_file_path,
                   metadata_file_path = metadata_file_path,
                   marker_column_name = marker_column_name,
                   condition_column_name = condition_column_name,
                   id_column_name = id_column_name,
                   final_point = final_point,
                   initial_point = initial_point)
    for gene in result_fs["feature_score"]:
        result[gene]['fs_score'] = result_fs["feature_score"][gene]
    print("Time taken to calculate feature importance = ", time.time()-start_time)
    #
    start_time = time.time()
    tmp_df_scores = pd.DataFrame(result).T
    tmp_df_scores = calculate_confidence(tmp_df_scores)
    df_result = tmp_df_scores.sort_values("confidence_score",ascending=False)
    df_result.index.name = 'Gene'
    print("\nTime taken to summarize all score into CONFIDENCE SCORE = ", time.time()-start_time)
    #
    return df_result
#
#
if __name__ == '__main__':
    expression_file_path = "data/Transcriptomics_MA@MA@Crohns_Disease@Colon@TwoSteps@Healthy@TwoSteps@ExpData.csv"
    metadata_file_path = "data/Transcriptomics_MA@MA@Crohns_Disease@Colon@TwoSteps@Healthy@TwoSteps@SampInfo.csv"
    marker_column_name = "Gene"
    condition_column_name = "Condition"
    id_column_name="SampleID"
    final_point=["Crohns Disease"]
    initial_point=["Healthy"]
    final_point_name="Crohns Disease"
    initial_point_name="Healthy"
    permutation_test=0
    #
    start_time = time.time()
    result = prioritize_marker(expression_file_path = expression_file_path,
                   metadata_file_path = metadata_file_path,
                   marker_column_name = marker_column_name,
                   condition_column_name = condition_column_name,
                   id_column_name = id_column_name,
                   final_point = final_point,
                   initial_point = initial_point,
                   permutation_test = permutation_test,
                   final_point_name = final_point,
                   initial_point_name = initial_point)
    #
    result.to_csv("data/output.csv")
    print("\n\n---------!!!!TOTAL TIME TAKEN = ", time.time()-start_time,"seconds !!!!!!------------\n\n")
    #