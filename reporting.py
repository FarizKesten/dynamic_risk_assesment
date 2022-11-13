import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import logging
import diagnostics

logging.basicConfig(level=logging.INFO)



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])



##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    logging.info("Load deployed model")
    model = pickle.load(open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb'))

    logging.info("Load testdata.csv")
    df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    y = df.pop('exited')
    X = df.drop(['corporation'], axis=1)

    logging.info("Create confusion matrix")
    pred = diagnostics.model_predictions(X)
    res = sns.heatmap(metrics.confusion_matrix(y, pred), annot=True, fmt='d', cmap='Blues')

    # save heatmap to file
    logging.info("Save confusion matrix to confusionmatrix.png")
    res.get_figure().savefig(os.path.join(output_model_path, 'confusionmatrix.png'))



if __name__ == '__main__':
    score_model()
