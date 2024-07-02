Deploy kubernetes cluster with k3s: https://docs.k3s.io/quick-start


https://docs.gitlab.com/charts/installation/deployment.html

to access mlflow (from inside the cluster): mlflow-release-tracking.mlflow.svc.cluster.local

mlflow creds: 
user
uxnsYkzVaU


kubeflow: http://10.255.32.77 (istio-ingressgateway)


curl -X POST -H "Content-Type: application/json" -d @input.json http://localhost:54819/v1/models/k8s-attacks-detection:predict