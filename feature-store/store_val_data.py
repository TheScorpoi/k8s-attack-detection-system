from upload_data import login_feature_store

fs = None

def login_feature_store() -> None:
    """
    Login to Hopsworks Feature Store. Using the hopsworks free trial (https://c.app.hopsworks.ai)
    """
    import hopsworks
    
    global fs
    project = hopsworks.login()
    fs = project.get_feature_store()

def get_val_data_from_fs():
    import pandas as pd
    
    fv = fs.get_feature_view('benign_and_malicious_traffic_february_view', version=1)
    x_train, x_val, x_test, y_train, y_val, y_test = fv.get_train_validation_test_split(training_dataset_version=3)
    
    x_val_df = pd.DataFrame(x_val)
    y_val_df = pd.DataFrame(y_val)
    
    validation_dataset = pd.concat([x_val_df, y_val_df], axis=1)
    
    validation_dataset.to_csv("../data/validation/validation_dataset.csv")

if __name__ == "__main__":
    login_feature_store()
    
    get_val_data_from_fs()
    
    