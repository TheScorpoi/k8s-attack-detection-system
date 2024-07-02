#curl -X POST -H "Content-Type: application/json" -d @input.json http://localhost:57282/v1/models/aas-k8s-attack-model-deploy-test-kubeflow-endpoints:predict
curl -X POST -H "Content-Type: application/json" -d @input.json http://localhost:49192/v1/models/k8s-attacks-detection:predict
