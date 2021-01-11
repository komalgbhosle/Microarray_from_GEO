#
import math, copy, time, random, string
import numpy as np
import statsmodels.stats.multitest as multi
#
#
from tqdm import tqdm
from scipy import stats
from itertools import combinations,permutations
from multiprocessing import Pool, cpu_count
from itertools import repeat
#
from utilities import utils
#
def get_key_name(list_data1, list_data2):
    return "@".join(list_data2).replace(' ','_')+"-"+"@".join(list_data1).replace(' ','_')
def statistical_testing(expression_file_path=None,exp_filetype="csv",metadata_file_path=None,meta_filetype="csv",marker_column_name="Gene",condition_column_name="Condition",id_column_name="SampleID",final_point=["Crohns Disease"],initial_point=["Healthy"],final_point_name=None,initial_point_name=None,permutation_test=0,):
    #
    if not final_point_name:
        final_point_name=final_point[0]
    final_point_name = final_point_name.translate(str.maketrans('', '', string.punctuation))
    if not initial_point_name:
        initial_point_name=initial_point[0]
    initial_point_name = initial_point_name.translate(str.maketrans('', '', string.punctuation))
    #
    dict_df_degs = dict()
    list_pvalue_mann, list_pvalue_welch, list_pvalue_ttest = list(), list(), list()
    #
    dict_expdata = utils.read_df(expression_file_path,exp_filetype)
    dict_metadata = utils.read_df(metadata_file_path,meta_filetype)
    dict_expdata, dict_metadata = utils.check_exp_meta_data(dict_expdata, dict_metadata, marker_column_name, id_column_name)
    dict_expdata = utils.dataframe2dict(dict_expdata,marker_column_name)
    dict_metadata = utils.dataframe2dict(dict_metadata,id_column_name)
    #
    dict_cluster_condition = utils.get_id_by_condition(dict_input_data=dict_metadata,data_condition_keyname=condition_column_name,data_id_keyname=id_column_name)
    list_condition_sets = [(initial_point,final_point)]
    #
    for set1, set2 in list_condition_sets:
        key_name = get_key_name(set1, set2)
        #
        set1_ids = [_id for condition_set1 in set1 if condition_set1 in dict_cluster_condition for _id in list(dict_cluster_condition[condition_set1].keys())]
        set2_ids = [_id for condition_set2 in set2 if condition_set2 in dict_cluster_condition for _id in list(dict_cluster_condition[condition_set2].keys())]
        all_ids = set(list(set1_ids) + list(set2_ids))
        set1_len = len(set1_ids)
        set2_len = len(set2_ids)
        max_len = max(set1_len, set2_len)
        for gene in tqdm(dict_expdata, desc = key_name):
            dict_df_degs[gene] = dict()
            #
            set1_values = np.array([dict_expdata[gene][_id] for _id in set1_ids])
            set2_values = np.array([dict_expdata[gene][_id] for _id in set2_ids])
            #
            dict_df_degs[gene]['mean_'+str(initial_point_name).lower().replace(" ","_")] = np.mean(set1_values)
            dict_df_degs[gene]['mean_'+str(final_point_name).lower().replace(" ","_")] = np.mean(set2_values)
            dict_df_degs[gene]['logfc'] = np.mean(set2_values)-np.mean(set1_values)
            dict_df_degs[gene]['abslogfc'] = abs(np.mean(set2_values)-np.mean(set1_values))
            #
            try:
                dict_df_degs[gene]['mannwhitney_pvalue'] = stats.mannwhitneyu(set2_values, set1_values).pvalue
            except ValueError:
                dict_df_degs[gene]['mannwhitney_pvalue'] = 1
            dict_df_degs[gene]['student_ttest_pvalue'] = stats.ttest_ind(set2_values, set1_values).pvalue
            if np.isnan(dict_df_degs[gene]['student_ttest_pvalue']):
                dict_df_degs[gene]['student_ttest_pvalue'] = 1
            dict_df_degs[gene]["welch_ttest_pvalue"] = stats.ttest_ind(set2_values, set1_values, equal_var=False).pvalue
            if np.isnan(dict_df_degs[gene]["welch_ttest_pvalue"]):
                dict_df_degs[gene]["welch_ttest_pvalue"] = 1
            #
            list_pvalue_mann.append(dict_df_degs[gene]['mannwhitney_pvalue'])
            list_pvalue_welch.append(dict_df_degs[gene]["welch_ttest_pvalue"])
            list_pvalue_ttest.append(dict_df_degs[gene]['student_ttest_pvalue'])
        #
        list_fwer_mann, list_fdr_mann = utils.get_fwer(list_pvalue_mann), utils.get_fdr(list_pvalue_mann)
        list_fwer_welch, list_fdr_welch = utils.get_fwer(list_pvalue_welch), utils.get_fdr(list_pvalue_welch)
        list_fwer_ttest, list_fdr_ttest = utils.get_fwer(list_pvalue_ttest), utils.get_fdr(list_pvalue_ttest)
        #
        for i,gene_name in tqdm(enumerate(dict_df_degs)):
            dict_df_degs[gene_name]['mannwhitney_fwer'] = list_fwer_mann[i]
            dict_df_degs[gene_name]["welch_ttest_fwer"] = list_fwer_welch[i]
            dict_df_degs[gene_name]['student_ttest_fwer'] = list_fwer_ttest[i]
            #
            dict_df_degs[gene_name]['mannwhitney_qvalue'] = list_fdr_mann[i]
            dict_df_degs[gene_name]["welch_ttest_qvalue"] = list_fdr_welch[i]
            dict_df_degs[gene_name]['student_ttest_qvalue'] = list_fdr_ttest[i]
        #
        return dict_df_degs
#
if __name__ == '__main__':
    print("Deep Dive Dear!!")
