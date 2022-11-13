import requests
import json
import logging
import os
import requests

logging.basicConfig(level=logging.INFO)

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"

with open('config.json','r') as f:
    config = json.load(f)

test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


#Call each API endpoint and store the responses
logging.info("Post request /prediction")
response1 = requests.post(URL + "prediction",
                         json={"file": os.path.join(test_data_path,"testdata.csv")})

logging.info("Get request /scoring")
response2 = requests.get(URL + "scoring")

logging.info("Get request /summarystats")
response3 = requests.get(URL + "summarystats")

logging.info("Get request /diagnostics")
response4 = requests.get(URL + "diagnostics")

# #combine all API responses
# responses = #combine reponses here
#write the responses to your workspace

logging.info("Saving results to apireturns.txt")
with open(os.path.join(prod_deployment_path, 'apireturns.txt'), 'w') as file:
    file.write('Prediction result\n')
    file.write(response1.text)
    file.write('\nScoring result\n')
    file.write(response2.text)
    file.write('\nSummarystats result\n')
    file.write(response3.text)
    file.write('\nDiagnostics result\n')
    file.write(response4.text)






