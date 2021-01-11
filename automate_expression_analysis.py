import pandas as pd
import os, sys, time
#
from glob import glob
from main import prioritize_marker
#
if __name__ == '__main__':
    #headers:
    #expression_file_path -> filepath to expression data
    #metadata_file_path -> filepath  to metadata
    #marker_column_name -> column name in expression data file which represents feature/marker name
    #condition_column_name -> column name in metadata file which represents various conditions or cohorts
    #id_column_name -> column name in metadata which represents unique sample id
    #final_point -> string of pipe "|" separated final cohort names
    #initial_point -> string of pipe "|" separated initial cohort names
    #final_point_name -> name of final cohort
    #initial_point_name -> name of initial cohort
    #output_filename -> filepath along with filename for the result generated during analysis
    input_files = pd.read_csv("metafile_data.csv")
    for i, row in input_files.iterrows():
        print("-------------Started working on ",i," row in the input file.----------")
        try:
            tmp_final_point = row["final_point"].split("|")
            tmp_initial_point = row["initial_point"].split("|")
        except:
            print("Please check the headers required for input file for row number: ",i)
            continue
        try:
            result = prioritize_marker(expression_file_path = row["expression_file_path"],
                                       metadata_file_path = row["metadata_file_path"],
                                       marker_column_name = row["marker_column_name"],
                                       condition_column_name = row["condition_column_name"],
                                       id_column_name = row["id_column_name"],
                                       final_point = tmp_final_point,
                                       initial_point = tmp_initial_point,
                                       final_point_name = row["final_point_name"],
                                       initial_point_name = row["initial_point_name"])
            print("Results received successfully!! Writing into the file.")
        except Exception as e: 
            print("Error while computing results ",e)
            continue
        try:
            result.to_csv(row["output_filename"],index=True)
        except Exception as e:
            print("Error while writing the output of row: ",i, e)
            continue