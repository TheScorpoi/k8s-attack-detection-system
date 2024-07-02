import streamlit as st
import pandas as pd
import time
import json
import requests
import matplotlib.pyplot as plt
import logging

DATASET_PATH = '../data/validation/validation_dataset.csv'
URL_MODEL = 'http://localhost:49192/v1/models/k8s-attacks-detection:predict'

logging.basicConfig(level=logging.INFO)

def query_model(data):
    """
    Query the ML model with the validation data

    Args:
        data (list): list of features (row) to be predicted
    Returns:
        response (json): prediction of the ML model
    """
    headers = {"Content-Type": "application/json"}
    response = requests.post(URL_MODEL, headers=headers, data=json.dumps(data))
    #logging.info(f"Queried model with data: {data}")
    return response.json()

def plot_counts(correct_count, incorrect_count, placeholder):
    """
    Plot the correct vs incorrect predictions

    Args:
        correct_count : current count of correct predictions
        incorrect_count : current count of incorrect predictions
        placeholder : placeholder to plot the graph
    """
    plt.clf()
    plt.bar(['Correct', 'Incorrect'], [correct_count, incorrect_count])
    plt.title('Correct vs Incorrect predictions')
    plt.xlabel('Predictions')
    plt.ylabel('Counts')
    plt.ylim(0, max(correct_count, incorrect_count) + 1)
    placeholder.pyplot(plt) 

def main():
    st.title("K8S Attacks Detection - Validation")
    st.write("This page is used to validate the model. The model will be tested against the validation dataset.")
    st.write("Every second a request will be sent to the ML model with a row of the validation dataset.\nThe model will make a prediction and the result will be compared with the expected value. The results will be plotted in the graph below.")
    st.write("\n\nThe model is deployed in the following URL: " + URL_MODEL)

    #init session state variables
    if 'position' not in st.session_state:
        st.session_state.position = 0
    if 'correct_count' not in st.session_state:
        st.session_state.correct_count = 0
    if 'incorrect_count' not in st.session_state:
        st.session_state.incorrect_count = 0

    #load validation dataset
    dataset = pd.read_csv(DATASET_PATH)
    model_input = dataset.drop('label', axis=1)
    expected_results = dataset['label']

    #define where the plot will be placed
    position_placeholder = st.empty()
    plot_placeholder = st.empty()

    while True:
        if st.session_state.position < len(model_input):
            row = model_input.iloc[st.session_state.position]
            row_values = row.tolist()

            prediction = query_model(data={"instances": [row_values]})
            prediction = prediction['predictions'][0]
            expected_value = expected_results.iloc[st.session_state.position]
            logging.info(f"Expected value: {expected_value} -- Predicted value: {prediction}")

            if prediction == expected_value:
                st.session_state.correct_count += 1
            else:
                st.session_state.incorrect_count += 1

            plot_counts(st.session_state.correct_count, st.session_state.incorrect_count, plot_placeholder)

            st.session_state.position += 1
            position_placeholder.markdown(f"## #Predictions: {st.session_state.position}")

            #every 0.35 second the page will be updated and a request will be sent to the ML model
            time.sleep(0.35)
        else:
            break

    if st.button('Reset'):
        st.session_state.position = 0
        st.session_state.correct_count = 0
        st.session_state.incorrect_count = 0
        plt.clf()
        plot_placeholder.pyplot(plt)

if __name__ == "__main__":
    main()
