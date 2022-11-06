import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)


#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f)

input_path = os.path.abspath(config['input_folder_path'])
output_path = os.path.abspath(config['output_folder_path'])
#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    df = pd.DataFrame()
    # open a file and write the name of file that is read to it
    file_names = []
    logging.info(f"Reading files from {input_path}")

    with open(os.path.join(output_path, 'ingestedfiles.txt'), "w") as f:
        for file in os.listdir(input_path):
            if "csv" not in file:
                continue
            logging.info(f"Reading file {file}")
            file_path = os.path.join(input_path, file)
            # just write a relative path to the file
            f.write(os.path.join(*file_path.split(os.path.sep)[-3:]) + "\n")
            df_tmp = pd.read_csv(file_path)
            df = df.append(df_tmp, ignore_index=True)

    #Dropping duplicates
    df = df.drop_duplicates().reset_index(drop=1)
    # save the ingested data
    df.to_csv(os.path.join(output_path, 'finaldata.csv'), index=False)


if __name__ == '__main__':
    merge_multiple_dataframe()
