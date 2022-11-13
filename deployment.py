from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil
import logging


logging.basicConfig(level=logging.INFO)



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
output_model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


####################function for deployment
def copy_file(source, dest, filename):
    logging.info(f"Copying {filename} from {source} to {dest}")
    source = os.path.join(source, filename)
    dest = os.path.join(dest, filename)
    shutil.copy(source, dest)

def deploy_model_and_stats():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    logging.info("Deploy current model to production")
    copy_file(output_model_path, prod_deployment_path, 'trainedmodel.pkl')
    copy_file(output_model_path, prod_deployment_path, 'latestscore.txt')
    copy_file(dataset_csv_path, prod_deployment_path, 'ingestedfiles.txt')


if __name__ == '__main__':
    logging.info("Running Deployment")
    deploy_model_and_stats()