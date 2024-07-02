import kfp
from kfp import dsl
from kfp.dsl import (component, Output, Input, Dataset, ClassificationMetrics, Metrics, Model)


@component(
    packages_to_install=['requests'],
    base_image='python:3.8'
)
def load_data(output_data: Output[Dataset]):
    import requests
    url = 'https://raw.githubusercontent.com/TheScorpoi/test/main/preprocessed_data.csv'

    data = requests.get(url).content

    with open(output_data.path, 'wb') as writer:
        writer.write(data)

    print("Data downloaded and written to:", output_data.path)

@component(
    packages_to_install=['mlflow'],
    base_image='python:3.8',
)
def create_mlflow_experiment(mlflow_user: str, mlflow_password: str, mlflow_tracking_uri: str) -> str:
    import mlflow
    import os
    from datetime import datetime
    
    os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_user
    os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_password
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    client = mlflow.tracking.MlflowClient()
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"kubeflow-pipeline-{current_time}"

    experiment_id = client.create_experiment(experiment_name)

    print(f"Created a new MLflow experiment: {experiment_name} with ID: {experiment_id}")

    return experiment_name


@component(
    packages_to_install=['scikit-learn', 'pandas', 'mlflow'],
    base_image='python:3.8',
)
def train_logistict_regression(
    mlflow_user: str,
    mlflow_password: str,
    mlflow_tracking_uri: str,
    experiment_name: str,
    model: Output[Model],
    metrics: Output[Metrics],
    metric: Output[ClassificationMetrics],
    df: Input[Dataset]
):
    
    import pandas as pd
    import mlflow
    import mlflow.sklearn
    import json
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    
    os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_user
    os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_password
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # inside the IT network i cant access the HopsWorks machine so i have to download the data from githuh, 
    # and then i will do the train_test_split here and not use the one from feature store
    df = pd.read_csv(df.path)
    X = df.drop('label', axis=1)
    y = df['label']
    class_labels = y.unique().tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = StandardScaler()
    rf = LogisticRegression(random_state=42)
    pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', rf)])

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        pipeline_rf.fit(X_train, y_train)

        y_pred = pipeline_rf.predict(X_test)
        accuracy_value = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        metrics.log_metric('accuracy', accuracy_value)
        metric.log_confusion_matrix(matrix=cm.tolist(), categories=class_labels)

        mlflow.log_param("random_state", 42)
        mlflow.log_metric('accuracy', accuracy_value)
        mlflow.sklearn.log_model(pipeline_rf, "model")

        confusion_matrix_path = "confusion_matrix.json"
        with open(confusion_matrix_path, 'w') as cm_file:
            json.dump({'matrix': cm.tolist(), 'categories': class_labels}, cm_file)

        mlflow.log_artifact(confusion_matrix_path)

    model_uri = mlflow.get_artifact_uri("model")
    print("Model saved to:", model_uri)
    

@component(
    packages_to_install=['scikit-learn', 'pandas', 'mlflow'],
    base_image='python:3.8',
)
def train_SVC(
    mlflow_user: str,
    mlflow_password: str,
    mlflow_tracking_uri: str,
    experiment_name: str,
    model: Output[Model],
    metrics: Output[Metrics],
    metric: Output[ClassificationMetrics],
    df: Input[Dataset]
):
    
    import pandas as pd
    import mlflow
    import mlflow.sklearn
    import json
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_user
    os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_password
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # inside the IT network i cant access the HopsWorks machine so i have to download the data from githuh, 
    # and then i will do the train_test_split here and not use the one from feature store
    df = pd.read_csv(df.path)
    X = df.drop('label', axis=1)
    y = df['label']
    class_labels = y.unique().tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = StandardScaler()
    rf = SVC(random_state=42)
    pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', rf)])

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        pipeline_rf.fit(X_train, y_train)

        y_pred = pipeline_rf.predict(X_test)
        accuracy_value = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        metrics.log_metric('accuracy', accuracy_value)
        metric.log_confusion_matrix(matrix=cm.tolist(), categories=class_labels)

        mlflow.log_param("random_state", 42)
        mlflow.log_metric('accuracy', accuracy_value)
        mlflow.sklearn.log_model(pipeline_rf, "model")

        confusion_matrix_path = "confusion_matrix.json"
        with open(confusion_matrix_path, 'w') as cm_file:
            json.dump({'matrix': cm.tolist(), 'categories': class_labels}, cm_file)

        mlflow.log_artifact(confusion_matrix_path)

    model_uri = mlflow.get_artifact_uri("model")
    print("Model saved to:", model_uri)
    

@component(
    packages_to_install=['scikit-learn', 'pandas', 'mlflow'],
    base_image='python:3.8',
)
def train_decision_tree(
    mlflow_user: str,
    mlflow_password: str,
    mlflow_tracking_uri: str,
    experiment_name: str,
    model: Output[Model],
    metrics: Output[Metrics],
    metric: Output[ClassificationMetrics],
    df: Input[Dataset]
):
    
    import pandas as pd
    import mlflow
    import mlflow.sklearn
    import json
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    
    os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_user
    os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_password
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # inside the IT network i cant access the HopsWorks machine so i have to download the data from githuh, 
    # and then i will do the train_test_split here and not use the one from feature store
    df = pd.read_csv(df.path)
    X = df.drop('label', axis=1)
    y = df['label']
    class_labels = y.unique().tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = StandardScaler()
    rf = DecisionTreeClassifier(random_state=42)
    pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', rf)])

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        pipeline_rf.fit(X_train, y_train)

        y_pred = pipeline_rf.predict(X_test)
        accuracy_value = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        metrics.log_metric('accuracy', accuracy_value)
        metric.log_confusion_matrix(matrix=cm.tolist(), categories=class_labels)

        mlflow.log_param("random_state", 42)
        mlflow.log_metric('accuracy', accuracy_value)
        mlflow.sklearn.log_model(pipeline_rf, "model")

        confusion_matrix_path = "confusion_matrix.json"
        with open(confusion_matrix_path, 'w') as cm_file:
            json.dump({'matrix': cm.tolist(), 'categories': class_labels}, cm_file)

        mlflow.log_artifact(confusion_matrix_path)

    model_uri = mlflow.get_artifact_uri("model")
    print("Model saved to:", model_uri)

@component(
    packages_to_install=['scikit-learn', 'pandas', 'mlflow'],
    base_image='python:3.8',   
)
def train_knn(
    mlflow_user: str,
    mlflow_password: str,
    mlflow_tracking_uri: str,
    experiment_name: str,
    model: Output[Model],
    metrics: Output[Metrics],
    metric: Output[ClassificationMetrics],
    df: Input[Dataset]
):
    
    import pandas as pd
    import mlflow
    import mlflow.sklearn
    import os
    import json
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix

    os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_user
    os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_password
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # inside the IT network i cant access the HopsWorks machine so i have to download the data from githuh, 
    # and then i will do the train_test_split here and not use the one from feature store
    df = pd.read_csv(df.path)
    X = df.drop('label', axis=1)
    y = df['label']
    class_labels = y.unique().tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = StandardScaler()
    knn = KNeighborsClassifier()
    pipeline_knn = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', knn)])
    
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        pipeline_knn.fit(X_train, y_train)
        
        y_pred = pipeline_knn.predict(X_test)
        
        accuracy_value = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # kubeflow pipeline metrics
        metrics.log_metric('accuracy', accuracy_value)
        metric.log_confusion_matrix(matrix=cm.tolist(), categories=class_labels)
        
        mlflow.log_param("random_state", 42)
        mlflow.log_metric('accuracy', accuracy_value)
        mlflow.sklearn.log_model(pipeline_knn, "model")
        
        confusion_matrix_path = "confusion_matrix.json"
        with open(confusion_matrix_path, 'w') as cm_file:
            json.dump({'matrix': cm.tolist(), 'categories': class_labels}, cm_file)
            
        mlflow.log_artifact(confusion_matrix_path)
        
        
    model_uri = mlflow.get_artifact_uri("model")
    print("Model saved to:", model_uri)


@component(
    packages_to_install=['mlflow'],
    base_image='python:3.8',
)
def select_best_model(
    mlflow_user: str,
    mlflow_password: str, 
    mlflow_tracking_uri: str,
    experiment_name: str,
    metric_name: str
) -> dict:
    import mlflow
    import os

    os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_user
    os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_password
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)

    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = client.search_runs([experiment.experiment_id])

    best_run = None
    best_metric = float("-inf")
    
    for run in runs:
        metric_value = run.data.metrics.get(metric_name)

        if metric_value and metric_value > best_metric:
            best_metric = metric_value
            best_run = run
    if not best_run:
        raise ValueError("No suitable run found.")

    model_uri = f"runs:/{best_run.info.run_id}/model"
    model_name="model1"
    
    #register best model in MLFlow model registry
    result = mlflow.register_model(
        model_uri,
        name=model_name,
    )
    
    model_version = result.version
    print("result version: " + result.version)

    print(f"Selected best model from run {best_run.info.run_id} with {metric_name}: {best_metric}")
    return {"model_name": model_name, "model_version": model_version, "artifact_uri": model_uri}
        
@component(
    packages_to_install=["google-cloud-storage==2.14.0", "mlflow"],
    base_image='python:3.8',
)        
def upload_best_model_to_bucket(
    google_auth: str,
    model_info: dict,
    mlflow_user: str,
    mlflow_password: str,
    mlflow_tracking_uri: str,
) -> str:
    import json
    import os
    import mlflow
    from google.cloud import storage
    from google.oauth2 import service_account
    from mlflow.tracking import MlflowClient
    
    credentials = service_account.Credentials.from_service_account_info(json.loads(google_auth))
    storage_client = storage.Client(credentials=credentials, project=credentials.project_id)

    model_name = model_info['model_name']
    model_version = model_info['model_version']
    
    os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_user
    os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_password
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    #! check if it runs with the model_info['artifact_uri'] from the component before
    client = MlflowClient()
    model_version_details = client.get_model_version(model_name, model_version)
    artifact_uri = model_version_details.source

    local_model_dir = "/tmp/model_artifacts"
    os.makedirs(local_model_dir, exist_ok=True)
    mlflow.artifacts.download_artifacts(artifact_uri, dst_path=local_model_dir)
    
    model_file_path = os.path.join(local_model_dir, 'model','model.pkl')
    destination_blob_name = f"{model_name}_v{model_version}.pkl"

    def upload_blob(bucket_name, source_file_name, destination_blob_name):
        """Uploads a file to the bucket."""
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        print(f"File {source_file_name} uploaded to {destination_blob_name}.")

    upload_blob('models-aveiro', model_file_path, destination_blob_name)

    model_url = "https://storage.googleapis.com/models-aveiro/" + destination_blob_name
    return model_url

@component(
    packages_to_install=["kserve"],
    base_image='python:3.9',
)
def deploy_model_with_kserve(
    model_storage_uri: str,
    kserve_inference_service_name: str,
    namespace: str = "kserve-test",
):
    from kubernetes import client 
    from kserve import KServeClient
    from kserve import constants
    from kserve import utils
    from kserve import V1beta1InferenceService
    from kserve import V1beta1InferenceServiceSpec
    from kserve import V1beta1PredictorSpec
    from kserve import V1beta1SKLearnSpec

    kserve_client = KServeClient()

    isvc = V1beta1InferenceService(
        api_version=constants.KSERVE_GROUP + '/v1beta1',
        kind=constants.KSERVE_KIND,
        metadata=client.V1ObjectMeta(
            name=kserve_inference_service_name, 
            namespace=namespace, 
            annotations={'serving.kserve.io/enable-prometheus-scraping': "true"}
        ),
        spec=V1beta1InferenceServiceSpec(
            predictor=V1beta1PredictorSpec(
                sklearn=V1beta1SKLearnSpec(storage_uri=model_storage_uri)
            )
        )
    )

    kserve_client.create(isvc, namespace=namespace)
    print(f"InferenceService {kserve_inference_service_name} deployed in namespace {namespace}.")

@dsl.pipeline(
    name='Kubernetes Attacks Detection - Kubeflow Pipeline',
    description='Kubeflow pipeline to train and deploy a model to detect Kubernetes attacks'
)
def k8s_attacks_detection_pipeline():
    from kubernetes import client, config
    import base64
    
    MLFLOW_USER = "MLFLOW_TRACKING_USERNAME"
    MLFLOW_PASSWORD = "MLFLOW_TRACKING_PASSWORD"
    MLFLOW_TRACKING_URI = "http://mlflow-release-tracking.mlflow.svc.cluster.local"
    
    config.load_kube_config()

    v1 = client.CoreV1Api()
    secret = v1.read_namespaced_secret("mlflow-secret", namespace="kubeflow-user-example-com")
    secret_mlflow_user = base64.b64decode(secret.data[MLFLOW_USER]).decode('utf-8')
    secret_mlflow_password = base64.b64decode(secret.data[MLFLOW_PASSWORD]).decode('utf-8')
    
    secret_google = v1.read_namespaced_secret("google-cloud-key", namespace="kubeflow-user-example-com")
    google_cloud_key_json = base64.b64decode(secret_google.data['key.json']).decode('utf-8')
    
    load_data_from_fs = load_data()
    
    generate_experiment_name = create_mlflow_experiment(
        mlflow_tracking_uri=MLFLOW_TRACKING_URI, 
        mlflow_user=secret_mlflow_user,
        mlflow_password=secret_mlflow_password)
    
    logistic_regression = train_logistict_regression(
        mlflow_user=secret_mlflow_user,
        mlflow_password=secret_mlflow_password,
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=generate_experiment_name.output,
        df=load_data_from_fs.output
    ).after(load_data_from_fs)
    
    svc = train_SVC(
        mlflow_user=secret_mlflow_user,
        mlflow_password=secret_mlflow_password,
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=generate_experiment_name.output,
        df=load_data_from_fs.output
    ).after(load_data_from_fs)
    
    knn = train_knn(
        mlflow_user=secret_mlflow_user,
        mlflow_password=secret_mlflow_password,
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=generate_experiment_name.output,
        df=load_data_from_fs.output
    ).after(load_data_from_fs)
    
    decision_tree = train_decision_tree(
        mlflow_user=secret_mlflow_user,
        mlflow_password=secret_mlflow_password,
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=generate_experiment_name.output,
        df=load_data_from_fs.output
    ).after(load_data_from_fs)
    
    select_model_task = select_best_model(
        mlflow_user=secret_mlflow_user,
        mlflow_password=secret_mlflow_password,
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=generate_experiment_name.output,
        metric_name='accuracy',
    ).after(logistic_regression, svc, knn, decision_tree)
    
    update_best_model_bucket_task = upload_best_model_to_bucket(
        google_auth=google_cloud_key_json,
        model_info=select_model_task.output,
        mlflow_user=secret_mlflow_user,
        mlflow_password=secret_mlflow_password,
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
    ).after(select_model_task)
    
    deploy_model = deploy_model_with_kserve(
        model_storage_uri=update_best_model_bucket_task.output,
        kserve_inference_service_name="k8s-attacks-detection",
        namespace="kubeflow-user-example-com"
    ).after(update_best_model_bucket_task)

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(
        pipeline_func=k8s_attacks_detection_pipeline, package_path='k8s_attacks_detection_pipeline-gh-data.yaml')