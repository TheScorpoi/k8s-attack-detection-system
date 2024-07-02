import hopsworks
import pandas as pd
import numpy as np
import great_expectations as ge
from great_expectations.core import ExpectationSuite, ExpectationConfiguration

from pre_process_data import PreProcessData
from feature_validation import FeatureValidation

fs = None

def login_feature_store() -> None:
    """
    Login to Hopsworks Feature Store. Using the hopsworks free trial (https://c.app.hopsworks.ai)
    """
    global fs
    project = hopsworks.login()
    fs = project.get_feature_store()

def read_data(path: str) -> pd.DataFrame:
    """
    Read data from csv file.
    Do some data cleaning and transformation.

    Args:
        path (str): _description_

    Returns:
        _type_: _description_
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.lstrip('_')
    df.rename(columns={'source_@timestamp': 'source_timestamp'}, inplace=True)
    df = df.dropna(subset=['source_network_transport'])
    
    return df

def pre_process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre process data before upload to feature store.

    Args:
        df (pd.DataFrame): data to be pre-processed

    Returns:
        data(pd.DataFrame): data pre-processed
    """
    ppd = PreProcessData(df)
    ppd.arrange_columns_names().network_transport_to_numerical() \
        .ip_address_to_numerical().source_flow_final_to_numerical() \
        .source_flow_id_to_numerical().convert_timestamp_to_numerical() \
        .convert_uint64_to_int()

    return ppd.data

def create_fg(df: pd.DataFrame, featuregroup_name: str, featuregroup_version: int, primary_key: list):
    """
    Create Feature Group and upload data to the Feature Store.

    Args:
        df (pd.DataFrame): _description_
        featuregroup_name (str): _description_
        featuregroup_version (int): _description_
        primary_key (list): _description_

    Returns:
        traffic_fg (_type_): feature group object 
    """
    global fs

    traffic_fg = fs.get_or_create_feature_group(
        name=featuregroup_name,
        version=featuregroup_version,
        description="Benign and malicious traffic from February 2022",
        primary_key=primary_key,
        online_enabled=False,
        time_travel_format="HUDI",
        statistics_config={"enabled": True, "histograms": True, "correlations": True, "exact_uniqueness": True}
    )
    
    feature_validation = FeatureValidation(df)
    feature_validation.register_expectation_suite(feature_group=traffic_fg)
    
    traffic_fg.insert(df)
    return traffic_fg
    
def update_feature_descriptions(featuregroup_name: str, featuregroup_version: int) -> None:
    """
    Update feature group descriptions.
    """
    
    traffic_fg = fs.get_feature_group(featuregroup_name, version=featuregroup_version)
    
    feature_descriptions = [
    {"name": "source_flow_id_encoded", "description": "Flow ID"},
    {"name": "source_flow_final", "description": "Flow final status"},
    {"name": "source_source_ip", "description": "Source IP address"},
    {"name": "source_destination_ip", "description": "Destination IP address"},
    {"name": "source_network_bytes", "description": "Number of bytes transferred in the transaction"},
    {"name": "source_network_transport", "description": "Type of network transport used"},
    {"name": "source_timestamp", "description": "Timestamp of the transaction"},
    {"name": "source_event_duration", "description": "Duration of the transaction"},
    {"name": "source_destination_port", "description": "Destination port of the transaction"},
    {"name": "source_source_port", "description": "Source port of the transaction"},
    {"name": "label", "description": "Label to classify the transaction as benign or malicious"},
    ]

    for desc in feature_descriptions: 
        traffic_fg.update_feature_description(desc["name"], desc["description"])

def create_training_data():
    """
    Normalize numerical features.
    Create Feature View from Feature Group.
    Split data between train and test (80/20) and upload to Feature View.    
    """
    min_max_scaler = fs.get_transformation_function(name="min_max_scaler")
    
    numerical_features = [
        "source_destination_port", "source_source_port", 
        "source_network_bytes", "source_event_duration",
        "source_flow_id_encoded", "source_flow_final",
        "source_source_ip", "source_destination_ip",
        "source_network_transport", "source_timestamp"
    ]
    
    transformation_functions = {}
    for feature in numerical_features:
        transformation_functions[feature] = min_max_scaler

    traffic_fg = fs.get_feature_group('benign_and_malicious_traffic_february', version=1)
    query = traffic_fg.select_all(include_primary_key=True)
        
    feature_view = fs.get_or_create_feature_view(
        name = 'benign_and_malicious_traffic_february_view',
        version = 1,
        labels=["label"],
        transformation_functions=transformation_functions,
        query=query
    )

    #split data into train and test and upload to feature view
    version, job = feature_view.create_train_validation_test_split(test_size=0.2, validation_size=0.1)


if __name__ == "__main__":
    
    login_feature_store()
    
    df = read_data("../data/629887f7-f6aa-4d77-b0db-83822a92c582_1_all/elastic_february2022_data.csv")
    
    #pre-process data
    df = pre_process_data(df)
    
    #df.to_csv("../data/629887f7-f6aa-4d77-b0db-83822a92c582_1_all/final.csv", index=False)
    #print(df.head(-1))
    
    #crate feature group inside feature store
    fg = create_fg(df, "benign_and_malicious_traffic_february", 1, ["source_flow_id_encoded"])
    
    #update feature descriptions
    update_feature_descriptions(featuregroup_name="benign_and_malicious_traffic_february", featuregroup_version=1)
    
    #create feature view to then generate training dataaet (train, test, validation)
    create_training_data()