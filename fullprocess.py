import training
import scoring
import deployment
import diagnostics
import reporting
import ingestion

import logging
import json
import os

logging.basicConfig(level=logging.INFO)

def process():

    with open('config.json','r') as f:
        config = json.load(f)

    input_folder_path    = os.path.join(config['input_folder_path'])
    output_folder_path   = os.path.join(config['output_folder_path'])
    prod_deployment_path = os.path.join(config['prod_deployment_path'])

    ##################Check and read new data
    #first, read ingestedfiles.txt
    logging.info("Checking for new data")
    with open(os.path.join(output_folder_path, 'ingestedfiles.txt')) as file:
        ingested_files = {line.strip('\n') for line in file.readlines()[1:]}

    #second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    source_files = os.listdir(input_folder_path)
    #check if elements in source_files are not in ingested_files
    ##################Deciding whether to proceed, part 1
    #if you found new data, you should proceed. otherwise, do end the process here
    if len(set(source_files).difference(ingested_files)) == 0:
        logging.info("No new data found")
        return None

    logging.info("Ingesting new data")
    ingestion.merge_multiple_dataframe()

    ##################Checking for model drift
    #check whether the score from the deployed model is different from
    #the score from the model that uses the newest ingested data
    logging.info("Get the score of the trained model with the newest ingested data")
    file = os.path.join(output_folder_path, 'finaldata.csv')
    new_score = scoring.score_model(file, save_results=False)

    with open(os.path.join(prod_deployment_path, 'latestscore.txt')) as f:
        deployed_score = float(f.read().split(' ')[-1])

    logging.info("Checking for model drift")
    logging.info(f"Deployed score = {deployed_score}")
    logging.info(f"New score = {new_score}")

    ##################Deciding whether to proceed, part 2
    #if you found model drift, you should proceed. otherwise, do end the process here
    if (new_score <= deployed_score):
        logging.info("No model drift occurred")
        return None

    logging.info("Model drift occurred, retraining & rescoring")
    training.train_model()
    trained_score = scoring.score_model()

    ##################Re-deployment
    #if you found evidence for model drift, re-run the deployment.py script
    logging.info("Re-deploying model")
    deployment.deploy_model_and_stats()

    ##################Diagnostics and reporting
    #run diagnostics.py and reporting.py for the re-deployed model
    logging.info("Running diagnostics")
    reporting.score_model()
    logging.info("Re-run API")
    os.system("python apicalls.py")


if __name__ == '__main__':
    process()




