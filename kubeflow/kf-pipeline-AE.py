import kfp
from kfp import dsl
from kfp.dsl import (component, Output, InputPath, OutputPath, Input, Artifact, Dataset, ClassificationMetrics, Metrics, Model)
from typing import Dict

@component(
    packages_to_install=['hopsworks'],
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
    experiment_name = f"kubeflow-pipeline-autoencoder-{current_time}"

    experiment_id = client.create_experiment(experiment_name)

    print(f"Created a new MLflow experiment: {experiment_name} with ID: {experiment_id}")

    return experiment_name


@component(
    packages_to_install=['tensorflow-cpu', 'pandas', 'numpy', 'scikit-learn'],
    base_image='tensorflow/tensorflow:1.6.0'
)
def train_autoencoder(
    df: Input[Dataset],
    encoded_train: Output[Dataset],
    encoded_test: Output[Dataset],
    y_train: Output[Dataset],
    y_test: Output[Dataset]
):
    
    """
    The CPUs are not compatible with AVX2 instructions, so this pipeline will not work in this kubeflow pipeline.
    """
    
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn import model_selection
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers.legacy import Adam
    from tensorflow.keras.initializers import glorot_normal
    
    data = pd.read_csv(df.path)
    X = data.drop('label', axis=1)
    y = data['label']

    x_train, x_test, y_train_data, y_test_data = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train_norm = scaler.transform(x_train)
    x_test_norm = scaler.transform(x_test)

    input_dim = x_train_norm.shape[1]
    latent_dim = 6

    tf.keras.utils.set_random_seed(42)

    # Encoder
    input_img = Input(shape=(input_dim,))
    encoded = Dense(10, activation='swish', kernel_initializer=glorot_normal())(input_img)
    encoded = Dense(8, activation='swish', kernel_initializer=glorot_normal())(encoded)
    encoded = Dense(latent_dim, activation='relu', kernel_initializer=glorot_normal())(encoded)
    encoder = Model(input_img, encoded)

    # Decoder
    decoder_input = Input(shape=(latent_dim,))
    decoded_layer = Dense(8, activation='swish', kernel_initializer=glorot_normal())(decoder_input)
    decoded_layer = Dense(10, activation='swish', kernel_initializer=glorot_normal())(decoded_layer)
    decoded_layer = Dense(input_dim, activation='linear', kernel_initializer=glorot_normal())(decoded_layer)
    decoder = Model(decoder_input, decoded_layer)

    # Autoencoder
    autoencoder_output = decoder(encoder(input_img))
    autoencoder = Model(input_img, autoencoder_output)
    autoencoder.compile(optimizer=Adam(), loss='mse')
    history = autoencoder.fit(x_train_norm, x_train_norm, epochs=128, batch_size=512, shuffle=True, validation_data=(x_test_norm, x_test_norm))
    encoder.compile(optimizer=Adam(), loss='mse')
    encoded_train_data = encoder.predict(x_train_norm)
    encoded_test_data = encoder.predict(x_test_norm)
    
    np.savetxt(encoded_train.path, encoded_train_data, delimiter=',')
    np.savetxt(encoded_test.path, encoded_test_data, delimiter=',')
    y_train_data.to_csv(y_train.path, index=False)
    y_test_data.to_csv(y_test.path, index=False)


@component(
    packages_to_install=['scikit-learn', 'pandas', 'numpy','mlflow'],
    base_image='python:3.8',
)
def train_random_forest(
    mlflow_user: str,
    mlflow_password: str,
    mlflow_tracking_uri: str,
    experiment_name: str,
    model: Output[Model],
    metrics: Output[Metrics],
    metric: Output[ClassificationMetrics],
    encoded_train: Input[Dataset],
    encoded_test: Input[Dataset],
    y_train: Input[Dataset],
    y_test: Input[Dataset]
):
    
    import pandas as pd
    import numpy as np
    import mlflow
    import mlflow.sklearn
    import json
    import os
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    
    os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_user
    os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_password
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    X_train = np.loadtxt(encoded_train.path, delimiter=',')
    X_test = np.loadtxt(encoded_test.path, delimiter=',')
    y_train = pd.read_csv(y_train.path)
    y_test = pd.read_csv(y_test.path)

    class_labels = y_test.unique().tolist()

    rf = RandomForestClassifier(random_state=42)
    pipeline_rf = Pipeline(steps=[('classifier', rf)])

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
    packages_to_install=['scikit-learn', 'pandas', 'numpy','mlflow'],
    base_image='python:3.8',
)
def train_gbr(
    mlflow_user: str,
    mlflow_password: str,
    mlflow_tracking_uri: str,
    experiment_name: str,
    model: Output[Model],
    metrics: Output[Metrics],
    metric: Output[ClassificationMetrics],
    encoded_train: Input[Dataset],
    encoded_test: Input[Dataset],
    y_train: Input[Dataset],
    y_test: Input[Dataset]
):
    
    import pandas as pd
    import numpy as np
    import mlflow
    import mlflow.sklearn
    import json
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    
    os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_user
    os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_password
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    X_train = np.loadtxt(encoded_train.path, delimiter=',')
    X_test = np.loadtxt(encoded_test.path, delimiter=',')
    y_train = pd.read_csv(y_train.path)
    y_test = pd.read_csv(y_test.path)

    class_labels = y_test.unique().tolist()

    gbr = GradientBoostingClassifier(random_state=42)
    pipeline = Pipeline(steps=[('classifier', gbr)])

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        accuracy_value = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        metrics.log_metric('accuracy', accuracy_value)
        metric.log_confusion_matrix(matrix=cm.tolist(), categories=class_labels)

        mlflow.log_param("random_state", 42)
        mlflow.log_metric('accuracy', accuracy_value)
        mlflow.sklearn.log_model(pipeline, "model")

        confusion_matrix_path = "confusion_matrix.json"
        with open(confusion_matrix_path, 'w') as cm_file:
            json.dump({'matrix': cm.tolist(), 'categories': class_labels}, cm_file)

        mlflow.log_artifact(confusion_matrix_path)

    model_uri = mlflow.get_artifact_uri("model")
    print("Model saved to:", model_uri)

@component(
    packages_to_install=['scikit-learn', 'pandas', 'numpy','mlflow'],
    base_image='python:3.8',
)
def train_lr(
    mlflow_user: str,
    mlflow_password: str,
    mlflow_tracking_uri: str,
    experiment_name: str,
    model: Output[Model],
    metrics: Output[Metrics],
    metric: Output[ClassificationMetrics],
    encoded_train: Input[Dataset],
    encoded_test: Input[Dataset],
    y_train: Input[Dataset],
    y_test: Input[Dataset]
):
    
    import pandas as pd
    import numpy as np
    import mlflow
    import mlflow.sklearn
    import json
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_user
    os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_password
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    X_train = np.loadtxt(encoded_train.path, delimiter=',')
    X_test = np.loadtxt(encoded_test.path, delimiter=',')
    y_train = pd.read_csv(y_train.path)
    y_test = pd.read_csv(y_test.path)

    class_labels = y_test.unique().tolist()

    lr = LogisticRegression(random_state=42)
    pipeline = Pipeline(steps=[('classifier', lr)])

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        accuracy_value = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        metrics.log_metric('accuracy', accuracy_value)
        metric.log_confusion_matrix(matrix=cm.tolist(), categories=class_labels)

        mlflow.log_param("random_state", 42)
        mlflow.log_metric('accuracy', accuracy_value)
        mlflow.sklearn.log_model(pipeline, "model")

        confusion_matrix_path = "confusion_matrix.json"
        with open(confusion_matrix_path, 'w') as cm_file:
            json.dump({'matrix': cm.tolist(), 'categories': class_labels}, cm_file)

        mlflow.log_artifact(confusion_matrix_path)

    model_uri = mlflow.get_artifact_uri("model")
    print("Model saved to:", model_uri)

@component(
    packages_to_install=['scikit-learn', 'pandas', 'numpy','mlflow'],
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
    encoded_train: Input[Dataset],
    encoded_test: Input[Dataset],
    y_train: Input[Dataset],
    y_test: Input[Dataset]
):
    
    import pandas as pd
    import numpy as np
    import mlflow
    import mlflow.sklearn
    import json
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    
    os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_user
    os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_password
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    X_train = np.loadtxt(encoded_train.path, delimiter=',')
    X_test = np.loadtxt(encoded_test.path, delimiter=',')
    y_train = pd.read_csv(y_train.path)
    y_test = pd.read_csv(y_test.path)

    class_labels = y_test.unique().tolist()

    knn = KNeighborsClassifier(random_state=42)
    pipeline = Pipeline(steps=[('classifier', knn)])

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        accuracy_value = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        metrics.log_metric('accuracy', accuracy_value)
        metric.log_confusion_matrix(matrix=cm.tolist(), categories=class_labels)

        mlflow.log_param("random_state", 42)
        mlflow.log_metric('accuracy', accuracy_value)
        mlflow.sklearn.log_model(pipeline, "model")

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
    mlflow.register_model(
        model_uri,
        name=model_name,
    )

    print(f"Selected best model from run {best_run.info.run_id} with {metric_name}: {best_metric}")
    return {"model_name": model_name, "model_version": "1", "artifact_uri": model_uri}
        
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
    name='Kubernetes Attacks Detection -  Kubeflow Pipeline',
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
    
    load_data_from_fs = load_data()
    
    generate_experiment_name = create_mlflow_experiment(
        mlflow_tracking_uri=MLFLOW_TRACKING_URI, 
        mlflow_user=secret_mlflow_user,
        mlflow_password=secret_mlflow_password)
    
    train_autoencoder_task = train_autoencoder(    
        df=load_data_from_fs.output
    ).after(load_data_from_fs)
    
    random_forest = train_random_forest(
        mlflow_user=secret_mlflow_user,
        mlflow_password=secret_mlflow_password,
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=generate_experiment_name.output,
        encoded_train=train_autoencoder_task.outputs['encoded_train'],
        encoded_test=train_autoencoder_task.outputs['encoded_test'],
        y_train=train_autoencoder_task.outputs['y_train'],
        y_test=train_autoencoder_task.outputs['y_test']
    ).after(train_autoencoder_task)
    
    gbr_task = train_gbr(
        mlflow_user=secret_mlflow_user,
        mlflow_password=secret_mlflow_password,
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=generate_experiment_name.output,
        encoded_train=train_autoencoder_task.outputs['encoded_train'],
        encoded_test=train_autoencoder_task.outputs['encoded_test'],
        y_train=train_autoencoder_task.outputs['y_train'],
        y_test=train_autoencoder_task.outputs['y_test']
    ).after(train_autoencoder_task)
        
    lr_task = train_lr(
        mlflow_user=secret_mlflow_user,
        mlflow_password=secret_mlflow_password,
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=generate_experiment_name.output,
        encoded_train=train_autoencoder_task.outputs['encoded_train'],
        encoded_test=train_autoencoder_task.outputs['encoded_test'],
        y_train=train_autoencoder_task.outputs['y_train'],
        y_test=train_autoencoder_task.outputs['y_test']
    ).after(train_autoencoder_task)
    
    knn_task = train_knn(
        mlflow_user=secret_mlflow_user,
        mlflow_password=secret_mlflow_password,
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=generate_experiment_name.output,
        encoded_train=train_autoencoder_task.outputs['encoded_train'],
        encoded_test=train_autoencoder_task.outputs['encoded_test'],
        y_train=train_autoencoder_task.outputs['y_train'],
        y_test=train_autoencoder_task.outputs['y_test']
    ).after(train_autoencoder_task)
    
    select_model_task = select_best_model(
        mlflow_user=secret_mlflow_user,
        mlflow_password=secret_mlflow_password,
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=generate_experiment_name.output,
        metric_name='accuracy',
    ).after(random_forest, gbr_task, lr_task, knn_task)
    
    deploy_model = deploy_model_with_kserve(
        model_storage_uri="https://storage.googleapis.com/models-aveiro/model.pkl",
        kserve_inference_service_name="k8s-attacks-detection",
        namespace="kserve-test"
    ).after(select_model_task)

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(
        pipeline_func=k8s_attacks_detection_pipeline, package_path='k8s_attacks_detection_pipeline-AE.yaml')