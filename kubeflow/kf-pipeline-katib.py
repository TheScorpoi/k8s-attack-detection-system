import kfp
from kfp import dsl
#https://aniruddha-choudhury49.medium.com/mlops-kubeflow-with-feature-store-feast-in-google-cloud-platform-with-batch-sources-4c34fcea714d

from preprocess import load_data
from train import train_model
from utils import aggregate_metrics

from kfp import dsl
from kfp.dsl import (component, Output, Input, Artifact, Dataset, ClassificationMetrics, Metrics, Model)

from kubeflow.katib import V1beta1TrialTemplate, V1beta1TrialParameterSpec
from kubeflow.katib import V1beta1AlgorithmSpec, V1beta1ObjectiveSpec, V1beta1ParameterSpec
from kubeflow.katib import V1beta1FeasibleSpace, V1beta1ExperimentSpec, V1beta1TrialTemplate


@component(
    packages_to_install=['hopsworks'],
    base_image='python:3.8'
)
def load_data(fg_secret: int, dataset: str, data: Output[Dataset]):
    import hopsworks
    
    #! fazer isto com secrets do k8s
    project = hopsworks.login()
    fs = project.get_feature_store()
    fg = fs.get_feature_group('benign_and_malicious_traffic_february', version=fg_secret)
    
    data = fg.read()
    
    with open(dataset, 'w') as f:
        f.write(data)

@component(
    packages_to_install=['scikit-learn'],
    base_image='python:3.8',   
)
def train_random_forest(model: Output[Model],metrics: Output[Metrics], metric: Output[ClassificationMetrics], df: Input[Dataset]):
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_curve, accuracy_score, confusion_matrix
    from sklearn.metrics import roc_curve
    
    X = df.drop('label', axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = StandardScaler()

    rf = RandomForestClassifier(random_state=42)

    pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', rf)])

    pipeline_rf.fit(X_train, y_train)
    y_pred = pipeline_rf.predict(X_test)

    aggregate_metrics(y_test, y_pred, metrics, metric, model, model_name='random_forest')


@component(
    packages_to_install=['scikit-learn'],
    base_image='python:3.8',   
)
def train_svc(model: Output[Model], metrics: Output[Metrics], metric: Output[ClassificationMetrics], df: Input[Dataset]):
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    
    X = df.drop('label', axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = StandardScaler()

    svm = SVC(random_state=42)

    pipeline_svm = Pipeline(steps=[('preprocessor', svm), ('classifier', svm)])

    pipeline_svm.fit(X_train, y_train)
    y_pred = pipeline_svm.predict(X_test)

    aggregate_metrics(y_test, y_pred, metrics, metric, model, model_name='svm')
    
    
@component(
    base_image='python:3.8',
    packages_to_install=['kubernetes', 'kubeflow-katib']
)
def katib_experiment_launcher(experiment_name: str, namespace: str):
    def create_trial_template():
        trial_parameters = [
            V1beta1TrialParameterSpec(
                name="n_estimators",
                description="Number of trees in the forest",
                reference="n_estimators"
            ),
            V1beta1TrialParameterSpec(
                name="max_depth",
                description="Maximum depth of the tree",
                reference="max_depth"
            ),
            V1beta1TrialParameterSpec(
                name="C",
                description="Regularization parameter for SVM",
                reference="C"
            ),
            V1beta1TrialParameterSpec(
                name="gamma",
                description="Kernel coefficient for SVM",
                reference="gamma"
            ),
            # Add other parameters if needed
        ]

        trial_spec = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "training-container",
                                "image": "your-training-image",  # Replace with your training image
                                "command": [
                                    "python",
                                    "/path/to/your/training/script.py",
                                    "--n_estimators=${trialParameters.n_estimators}",
                                    "--max_depth=${trialParameters.max_depth}",
                                    "--C=${trialParameters.C}",
                                    "--gamma=${trialParameters.gamma}"
                                    # Add other parameter arguments as needed
                                ]
                            }
                        ],
                        "restartPolicy": "Never"
                    }
                }
            }
        }

        return V1beta1TrialTemplate(
            primary_container_name="training-container",
            trial_parameters=trial_parameters,
            trial_spec=trial_spec
        )

    def create_katib_experiment_spec():
        objective = V1beta1ObjectiveSpec(
            type="maximize",
            goal=0.95,
            objective_metric_name="accuracy",
        )

        algorithm = V1beta1AlgorithmSpec(
            algorithm_name="random",
        )

        parameters_rf = [
            V1beta1ParameterSpec(
                name="n_estimators",
                parameter_type="int",
                feasible_space=V1beta1FeasibleSpace(min="100", max="500"),
            ),
            V1beta1ParameterSpec(
                name="max_depth",
                parameter_type="int",
                feasible_space=V1beta1FeasibleSpace(min="5", max="20"),
            ),
            # Add other Random Forest parameters if needed
        ]

        parameters_svm = [
            V1beta1ParameterSpec(
                name="C",
                parameter_type="double",
                feasible_space=V1beta1FeasibleSpace(min="0.1", max="10"),
            ),
            V1beta1ParameterSpec(
                name="gamma",
                parameter_type="double",
                feasible_space=V1beta1FeasibleSpace(min="0.001", max="0.1"),
            ),
            # Add other SVM parameters if needed
        ]

        trial_template = create_trial_template()

        return V1beta1ExperimentSpec(
            max_trial_count=30,
            parallel_trial_count=3,
            max_failed_trial_count=5,
            algorithm=algorithm,
            objective=objective,
            parameters=parameters_rf + parameters_svm,
            trial_template=trial_template,
        )

    # Creating Katib experiment spec
    experiment_spec = create_katib_experiment_spec()

    # Creating Katib experiment object
    experiment = V1beta1Experiment(
        api_version="kubeflow.org/v1beta1",
        kind="Experiment",
        metadata=V1ObjectMeta(
            name=experiment_name,
            namespace=namespace,
        ),
        spec=experiment_spec
    )

    # Using KatibClient to create the experiment in Kubernetes
    katib_client = KatibClient()
    katib_client.create_experiment(experiment)


@dsl.pipeline(
    name='Kubernetes Attacks Detection -  Kubeflow Pipeline',
    description='Kubeflow pipeline to train and deploy a model to detect Kubernetes attacks'
)
def k8s_attacks_detection_pipeline():
    load_data_from_fs = load_data(dataset="data.csv", fg_secret=1)
    random_forest = train_random_forest(df=load_data_from_fs.output).after(load_data_from_fs)
    svm = train_svc(df=load_data_from_fs.output).after(load_data_from_fs)
    automl = katib_experiment_launcher().after(random_forest, svm)


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(
        pipeline_func=k8s_attacks_detection_pipeline, package_path='k8s_attacks_detection_pipeline-automl.yaml')