from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
# import create_prediction_model
import diagnostics
from scoring import score_model
# import predict_exited_from_saved_model
import json
import os



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
deployment_path = os.path.join(config['prod_deployment_path'])
test_data_path = os.path.join(config['test_data_path'])

prediction_model = None

#######################Prediction Endpoint

@app.route('/')
def index():
    return "Hello World!\n"


@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():
    #call the prediction function you created in Step 3
    file = request.get_json(force=True)['file']
    df = pd.read_csv(file)
    X = df.drop(['corporation', 'exited'], axis=1)

    preds = diagnostics.model_predictions(X)

    return jsonify(preds.tolist())

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():
    #check the score of the deployed model
    # with open(os.path.join(deployment_path, 'latestscore.txt'), 'r') as f:
    #     score = f.read().split(' ')[-1]
    score = score_model()
    return jsonify({'score': score})

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():
    #check means, medians, and modes for each column
    return jsonify(diagnostics.dataframe_summary())

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diag():
    #check timing and percent NA values
    missing_list    = diagnostics.calc_missing_data()
    execution_time  = diagnostics.execution_time()
    dependencies    = diagnostics.check_outdated_packages_list()
    return jsonify({'missing_list': missing_list,
                    'execution_time': execution_time,
                    'dependencies': dependencies})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
