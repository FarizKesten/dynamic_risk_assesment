
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

##################Function to get model predictions
def model_predictions(X):
    #read the deployed model and a test dataset, calculate predictions

    logging.info("Load deployed model")
    model = pickle.load(open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb'))

    logging.info("Predict data")
    return model.predict(X)


##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    #read data
    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv')).select_dtypes('number')

    logging.info('Calculate summary statistics')
    statistics = {}
    for col in df.columns:
        mean = df[col].mean()
        median = df[col].median()
        std = df[col].std()

        statistics[col] = {'mean': mean, 'median': median, 'std': std}

    return statistics


def calc_missing_data():
    logging.info("Load finaldata.csv")
    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    # calculate percentage in each column where the value is NA
    logging.info("Calculate missing data")
    missing_list = {}
    for col in df.columns:
        missing = df[col].isna().sum()
        total = df[col].shape[0]
        percentage = missing / total
        logging.info(f'Column {col} has {percentage:.2%} missing values')
        missing_list[col] = percentage
    return missing_list

def calculate_timing(script, loop=5):
    logging.info("get the mean execution time of %s, run for %s loops", script, loop)
    timing = []
    for i in range(loop):
        start = timeit.default_timer()
        _ = subprocess.run(['python', script], capture_output=True)
        timing.append(timeit.default_timer() - start)
    return np.mean(timing)

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    logging.info("Calculate execution time of training.py")
    training_time = calculate_timing('training.py')
    logging.info("Calculate execution time of ingestion.py")
    ingestion_time = calculate_timing('ingestion.py')
    return [training_time, ingestion_time]

##################Function to check dependencies
def check_outdated_packages_list():
    logging.info("Check outdated packages")
    dependencies = subprocess.run (
        ['pip', 'list', '--outdated', '../requirements.txt'],
        capture_output=True)

    dep = dependencies.stdout.decode('utf-8').split('\n')
    data = []
    for i in range(2, len(dep)):
        temp = ([x for x in dep[i].split(' ') if x != ''])
        if temp:
            data.append({'package': temp[0], 'current_version': temp[1], 'latest_version': temp[2]})

    return data


if __name__ == '__main__':
    logging.info("Load testdata.csv")
    df_test = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    X = df_test.drop(['exited', 'corporation'], axis=1)

    print("[RESULTS]: model prediction: ", model_predictions(X))
    print("[RESULTS]: calc missing data: ", calc_missing_data())
    print("[RESULTS]: dataframe summary: ", dataframe_summary())
    print("[RESULTS]: execution times[s]: ", execution_time())
    print("[RESULTS]: outdated-packages: ", check_outdated_packages_list())






