from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import logging

logging.basicConfig(level=logging.INFO)


#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])


#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    # load the test data
    logging.info("Load testdata.csv")
    df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))

    logging.info("Load trained model")
    model = pickle.load(open(os.path.join(output_model_path, 'trainedmodel.pkl'), 'rb'))

    logging.info("Prepare test data")
    y = df.pop('exited')
    X = df.drop(['corporation'], axis=1)

    logging.info("Predict test data")
    pred = model.predict(X)

    #calculate the f1 score
    f1_score = metrics.f1_score(y, pred)
    logging.info(f"f1 score = {f1_score}")
    logging.info("Save latest scores to latestscore.txt")
    with open(os.path.join(output_model_path, 'latestscore.txt'), 'w') as file:
        file.write(f"f1 score : {f1_score}")

    return f1_score


if __name__ == '__main__':
    logging.info("Determine model score")
    score_model()










