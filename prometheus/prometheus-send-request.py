from prometheus_client import start_http_server, Counter
import requests
import json
import time
import pandas as pd

correct_predictions = Counter('correct_predictions', 'Number of correct predictions')
incorrect_predictions = Counter('incorrect_predictions', 'Number of incorrect predictions')

def query_model(data):
    url = "http://localhost:55218/v1/models/k8s-attacks-detection:predict"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()

if __name__ == "__main__":
    start_http_server(4433)  

    dataset_path = './validation/validation_dataset.csv'
    dataset = pd.read_csv(dataset_path).drop('Unnamed: 0', axis=1)
    
    model_input = dataset.drop('label', axis=1)
    expected_result = dataset['label']

    print(model_input.columns)
    
    for index, row in model_input.iterrows():
        row_values = row[1:].tolist()

        data = {"instances": [row_values]}

        response = query_model(data)
        print(f"Response for row {index}: {response}")
        
        if response['predictions'][0] == expected_result:
            correct_predictions.inc()
        else:
            incorrect_predictions.inc()

        time.sleep(1)


