# AssureMOSS Kubernetes Run-time Monitoring Dataset
This dataset contains NetFlow data that is collected from a Kubernetes cluster. The cluster is used to monitor the microservice applications that are running on the cluster. The goal is to use the (NetFlow) logs to learn a state machine model that models the normal network behaviour within the cluster. The state machine model is then used to monitor and detect potential anomalies that might occur during the runtime of the cluster. This dataset contains both benign data (produced by real-life users) and malicious data (produced by launching several attacks against the clusters). The label of each flow is included in the dataset.

## Collection of Data
For the collection of data, we have set up the cluster with three main microservice applications and several different services. The three main microservice applications are the following: 
- Guestbook Application: a front-end application that allows users to read messages that were written by other users and write messages of their own. The messages are all presented on a web page.
- Joomla Blog Application: an application that allows users to write and read blog posts. This application is very similar to a WordPress blog.
- OpenSSH server: a tool that allows users to connect to the cluster remotely using an SSH connection. The user can execute commands once they have successfully logged in.

We set up several experiments in which real-life users are allowed to use the deployed applications normally. Their network behaviours are used to collect benign data. For the collection of malicious data, we launch several attacks during an experiment. The three microservice applications are the main targets of the attacks. Elasticsearch (specifically Packetbeat) is used to collect and extract the NetFlow data from the cluster. As real-life users are involved in our experiments, we have anonymised all IP addresses in the dataset to prevent the identification of the users.   

## Description of files
In this repository, you will find different CSV files. Each CSV file contains labelled NetFlow data collected from the cluster. The NetFlow data collected during the experiments have almost the same fields as the original NetFlow v9 format; in our dataset, we have only included a subset of the original fields as we did not use all original fields to learn our model. From the list of files, only two files (`elastic_may21_benign_data.csv` and `elastic_may21_malicious_data.csv`) do not contain network behaviours that were produced by real-life users. The former file contains only normal data that was produced by the applications and services and the latter file contains both data produced by the applications, services and attacks.

Each file with a name in the following format, `elastic_MONTHYEAR_data.csv`, contains NetFlow data collected from a particular experiment. Throughout the project, we will carry out more experiments and we will add the collected data to this repository. 

## Publications
This dataset is mentioned in the following publications written by the authors: 
- [Learning State Machines to Monitor and Detect Anomalies on a Kubernetes Cluster](https://doi.org/10.1145/3538969.3543810). This paper will appear later on ACM after the conference has been held. A preprint of this paper is available on [ArXiv](https://arxiv.org/abs/2207.12087). This paper contains information on the architecture of the cluster, descriptions of the attacks that we have used in our experiments and some initial performance results of our model on this dataset.
- [Encoding NetFlows for State-Machine Learning](https://arxiv.org/abs/2207.03890). This paper presents an encoding that we have used for the learning of our models.


## License
This dataset is licensed under the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license.

## Contact Information
For any questions regarding the dataset, please send a mail to Clinton Cao (c.s.cao@tudelft.nl) and/or Agathe Blaise (agathe.blaise@thalesgroup.com)

## Ackowledgements
This work is funded under the Assurance and certification in secure Multi-party Open Software and Services [(AssureMOSS) Project](https://assuremoss.eu/en/), with the support of the European Commission and H2020 Program, under Grant Agreement No. 952647. This dataset is collected for the work that is done in [Work Package 4 - Continuous run-time assessment of deployed applications and services](https://assuremoss.eu/en/project-structure/WP4-Continuous-run-time-assessment-of-deployed-applications-and-services) 
