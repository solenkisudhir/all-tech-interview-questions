# Machine Learning Pipeline

What is Machine Learning Pipeline?
----------------------------------

**_A Machine Learning pipeline is a process of automating the workflow of a complete machine learning task_**. It can be done by enabling a sequence of data to be transformed and correlated together in a model that can be analyzed to get the output. A typical pipeline includes raw data input, features, outputs, model parameters, ML models, and Predictions. Moreover, an ML Pipeline contains multiple sequential steps that perform everything ranging from data extraction and pre-processing to model training and deployment in Machine learning in a modular approach. It means that **_in the pipeline, each step is designed as an independent module, and all these modules are tied together to get the final result._**

The ML pipeline is a high-level API for MLlib within the "spark.ml" package. A typical pipeline contains various stages. However, there are two main pipeline stages:

![Machine Learning Pipeline](https://static.javatpoint.com/tutorial/machine-learning/images/machine-learning-pipeline.png)

1.  **Transformer:** It takes a dataset as an input and creates an augmented dataset as output. For example, A tokenizer works as Transformer, which takes a text dataset, and transforms it into tokenized words.
2.  **Estimator:** An estimator is an algorithm that fits on the input dataset to generate a model, which is a transformer. For example, regression is an Estimator that trains on a dataset with labels and features and produces a logistic regression model.

Importance of Machine Learning Pipeline
---------------------------------------

To understand the importance of a Machine learning pipeline, let's first understand a typical workflow of an ML task:

![Machine Learning Pipeline](https://static.javatpoint.com/tutorial/machine-learning/images/machine-learning-pipeline2.png)

A typical workflow consists of **Ingestion, Data cleaning, Data pre-processing, Modelling, and deployment.**

In ML workflow, all these steps are run together with the same script. It means the same script will be used to extract data, clean data, model, and deploy. However, it may generate issues while trying to scale an ML model. These issues involve:

*   If we need to deploy multiple versions of the same model, we need to run the complete workflow cycle multiple times, even when the very first step, i.e., ingestion and preparation, are exactly similar in each model.
*   If we want to expand our model, we need to copy and paste the code from the beginning of the process, which is an inefficient and bad way of software development.
*   If we want to change the configuration of any part of the workflow, we need to do it manually, which is a much more time-consuming process.

For solving all the above problems, we can use a Machine learning pipeline. With the ML pipeline, each part of the workflow acts as an **independent module**. So whenever we need to change any part, we can choose that specific module and use that as per our requirement.

We can understand it with an example. Building any ML model requires a huge amount of data to train the model. As data is collected from different resources, it is necessary to clean and pre-process the data, which is one of the crucial steps of an ML project. However, whenever a new dataset is included, we need to perform the same pre-processing step before using it for training, and it becomes a time-consuming and complex process for ML professionals.

To solve such issues, ML pipelines can be used, which can remember and automate the complete pre-processing steps in the same order.

Machine Learning Pipeline Steps
-------------------------------

On the basis of the use cases of the ML model and the requirement of the organization, each machine learning pipeline may be different to some extent. However, each pipeline follows/works upon the general workflow of Machine learning, or there are some common stages that each ML pipeline includes. Each stage of the pipeline takes the output from its preceding stage, which acts as the input for that particular stage. A typical ML pipeline includes the following stages:

![Machine Learning Pipeline](https://static.javatpoint.com/tutorial/machine-learning/images/machine-learning-pipeline3.png)

### 1\. Data Ingestion

Each ML pipeline starts with the Data ingestion step. In this step, the data is processed into a well-organized format, which could be suitable to apply for further steps. This step does not perform any feature engineering; rather, this may perform the versioning of the input data.

### 2\. Data Validation

The next step is data validation, which is required to perform before training a new model. Data validation focuses on statistics of the new data, e.g., range, number of categories, distribution of categories, etc. In this step, data scientists can detect if any anomaly present in the data. There are various data validation tools that enable us to compare different datasets to detect anomalies.

### 3\. Data Pre-processing

Data pre-processing is one of the most crucial steps for each ML lifecycle as well as the pipeline. We cannot directly input the collected data to train the model without pr-processing it, as it may generate an abrupt result.

The pre-processing step involves preparing the raw data and making it suitable for the ML model. The process includes different sub-steps, such as Data cleaning, feature scaling, etc. The product or output of the data pre-processing step becomes the final dataset that can be used for model training and testing. There are different tools in ML for data pre-processing that can range from simple Python scripts to graph models.

### 4\. Model Training & Tuning

The model training step is the core of each ML pipeline. In this step, the model is trained to take the input (pre-processed dataset) and predicts an output with the highest possible accuracy.

However, there could be some difficulties with larger models or with large training data sets. So, for this, efficient distribution of the model training or model tuning is required.

This issue of the model training stage can be solved with pipelines as they are scalable, and a large number of models can be processed concurrently.

### 5\. Model Analysis

After model training, we need to determine the optimal set of parameters by using the loss of accuracy metrics. Apart from this, an in-depth analysis of the model's performance is crucial for the final version of the model. The in-depth analysis includes calculating other metrics such as precision, recall, AUC, etc. This will also help us in determining the dependency of the model on features used in training and explore how the model's predictions would change if we altered the features of a single training example.

### 6\. Model Versioning

The model versioning step keeps track of which model, set of hyperparameters, and datasets have been selected as the next version to be deployed. For various situations, there could occur a significant difference in model performance just by applying more/better training data and without changing any model parameter. Hence, it is important to document all inputs into a new model version and track them.

### 7\. Model Deployment

After training and analyzing the model, it's time to deploy the model. An ML model can be deployed in three ways, which are:

*   Using the Model server,
*   In a Browser
*   On Edge device

However, the common way to deploy the model is using a model server. Model servers allow to host multiple versions simultaneously, which helps to run A/B tests on models and can provide valuable feedback for model improvement.

### 8\. Feedback Loop

Each pipeline forms a closed-loop to provide feedback. With this close loop, data scientists can determine the effectiveness and performance of the deployed models. This step could be automated or manual depending on the requirement. **Except for the two manual review steps (the model analysis and the feedback step), we can automate the entire pipeline.**

Benefits of Machine Learning Pipelines
--------------------------------------

Some of the benefits of using pipelines for the ML workflows are as follows:

*   **Unattended runs**  
    The pipeline allows to schedule different steps to run in parallel in a reliable and unattended way. It means you can focus on other tasks simultaneously when the process of data modeling and preparation is going on.
*   **Easy Debugging**  
    Using pipeline, there is a separate function for each task(such as different functions for data cleaning and data modeling). Therefore, it becomes easy to debug the complete code and find out the issues in a particular step.
*   **Easy tracking and versioning**  
    We can use a pipeline to explicitly name and version the data sources, inputs, and output rather than manually tracking data and outputs for each iteration.
*   **Fast execution**  
    As we discussed above, in the ML pipeline, each part of the workflow acts as an independent element, which allows the software to run faster and generate an efficient and high-quality output.
*   **Collaboration**  
    Using pipelines, data scientists can collaborate over each phase of the ML design process and can also work on different pipeline steps simultaneously.
*   **Reusability**  
    We can create pipeline templates for particular scenarios and can reuse them as per requirement. For example, creating a template for retraining and batch scoring.
*   **Heterogeneous Compute**  
    We can use multiple pipelines which are reliably coordinated over heterogeneous computer resources as well as different storage locations. It allows making efficient use of resources by running separate pipelines steps on different computing resources, e.g., GPUs, Data Science VMs, etc.

Considerations while building a Machine Learning Pipeline
---------------------------------------------------------

*   **Create each step as reusable components:**  
    We should consider all the steps that involve in an ML workflow for creating an ML model. Start building a pipeline with how data is collected and pre-processed, and continue till the end. It is recommended to limit the scope of each component to make it easier to understand and iterate.
*   **Always codify tests into components:**  
    Testing should be considered an inherent part of the pipeline. If you, in a manual process, do some checks on how the input data and the model predictions should look like, you should codify this into a pipeline.
*   **Put all the steps together:**  
    We must put all the steps together and define the order in which components of the workflow are processed, including how input and outputs run through the pipeline.
*   **Automate as per needed:**  
    When we create a pipeline, it already makes the workflow automated as it manages and handles the different running steps of workflow without any human intervention. However, various people's aim is to make the complete pipeline automated when specific criteria are met. For example, you may monitor model drift in production to trigger a re-training run or - simply do it more periodically, like daily.

ML Pipeline Tools
-----------------

There are different tools in Machine learning for building a Pipeline. Some are given below along with their usage:



* Steps while building the pipeline: Obtaining the Data
  * Tools: Managing the Database - PostgreSQL, MongoDB, DynamoDB, MySQL. Distributed Storage - Apache Hadoop, Apache Spark/Apache Flink.
* Steps while building the pipeline: Scrubbing / Cleaning the Data
  * Tools: Scripting Language - SAS, Python, and R. Processing in a Distributed manner - MapReduce/ Spark, Hadoop. Data Wrangling Tools - R, Python Pandas
* Steps while building the pipeline: Exploring / Visualizing the Data to find the patterns and trends
  * Tools: Python, R, MATLAB, and Weka.
* Steps while building the pipeline: Modeling the data to make the predictions
  * Tools: Machine Learning algorithms - Supervised, Unsupervised, Reinforcement, Semi-Supervised, and Semi-unsupervised learning. Important libraries - Python (Scikit learn) / R (CARET)
* Steps while building the pipeline: Interpreting the result
  * Tools: Data Visualization Tools -ggplot, Seaborn, D3.JS, Matplotlib, Tableau.


* * *

