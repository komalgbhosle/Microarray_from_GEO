#
import math, copy, time, random
#
import pandas as pd
import numpy as np
import networkx as nx
#
from tqdm import tqdm
from scipy import stats
from itertools import combinations,permutations
#
from utilities import utils
#
def interaction_type(value):
    if '-->' in value or '..>' in value:
        return 1
    elif '--|' in value:
        return -1
    else:
        return 0.0001
#
def get_network_as_graph(path2relations):
    pathway_data = pd.read_csv(path2relations)[['PathwayName','source','destination','value']]
    pathway_data['interaction'] = pathway_data.apply(lambda row: interaction_type(row['value']),axis=1)
    #
    dict_networks=dict()
    for pathway in tqdm(list(pathway_data['PathwayName'].unique()), desc="Decoding Pathways"):
        tmp_df = pathway_data[pathway_data['PathwayName']==pathway]
        dict_networks[pathway] = nx.from_pandas_edgelist(tmp_df,source='source',target='destination',
                                                  edge_attr="interaction",create_using=nx.DiGraph())
    #
    return dict_networks
#
def network_score(dict_data,feature_weight_key):
    dict_out = dict()
    dict_network = get_network_as_graph("utilities/KEGG_FinalRelations.txt")
    for feature in tqdm(dict_data,desc="Node Imapct Analysis"):
        tmp_us = 0
        tmp_ds = 0
        for pathway in dict_network:
            if(feature in dict_network[pathway]):
                try:
                    tmp_us += utils.get_us_perturbation_factor(feature,dict_network[pathway],dict_data,feature_weight_key)/len(set(dict_network[pathway].nodes))

                except RecursionError:
                    tmp_us+=0
                #
                try:
                    tmp_ds += utils.get_ds_perturbation_factor(feature,dict_network[pathway],dict_data,feature_weight_key)/len(set(dict_network[pathway].nodes))

                except RecursionError:
                    tmp_ds+=0
        #
        dict_data[feature]['network_score'] = np.mean([tmp_us+tmp_ds])
        dict_data[feature]['abs_network_score'] = abs(np.mean([tmp_us+tmp_ds]))
    return dict_data
#
if __name__ == '__main__':
    print("Deep Dive Dear!!")