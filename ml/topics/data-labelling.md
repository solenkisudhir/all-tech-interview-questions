# Data Labelling in Machine Learning
**_Data labeling is the way of identifying the raw data and adding suitable labels or tags to that data to specify what this data is about, which allows ML models to make an accurate prediction_**. In this topic, we will understand in detail Data Labelling, including the importance of data labeling in Machine Learning, different approaches, how data labeling works, etc. However, before starting, let's first understand what the labels are and how these are different from features in Machine Learning.

![Data Labelling in Machine Learning](https://static.javatpoint.com/tutorial/machine-learning/images/data-labelling-in-machine-learning.png)

Labels and Features in Machine Learning
---------------------------------------

### Labels in Machine Learning

Labels are also known as tags, which are used to give an identification to a piece of data and tell some information about that element. Labels are also referred to as the final output for a prediction. For example, as in the below image, we have labels such as a cat and dog, etc. For audio, labels could be the words that are said. This set of labels lets the ML model learn the dataset, as when we train a model with supervised techniques, we provide the model with a labeled dataset. With this labeled training dataset, the ML model easily predicts the accurate result when given the test dataset.

![Data Labelling in Machine Learning](https://static.javatpoint.com/tutorial/machine-learning/images/data-labelling-in-machine-learning2.gif)

### Features in Machine Learning

Features are the individual independent variables that work as input for the ML system. For an ML dataset, a column can be understood as a feature. ML models use these features to make predictions. Further, we can also get the new features from the old features using feature engineering methods.

We can understand **_the difference between both of them with a simple example of an image dataset of animals. So, height, weight, color, etc., are the features. Whereas it is a cat or dog, these are the labels._**

Now let's understand the main topic, i.e., Data Labelling

What is Data Labelling?
-----------------------

If we input a vast amount of raw data to a Machine Learning model and expect it to learn from it, then it is not enough. As it will give an abrupt result, so it is necessary to pre-process the data, and one of the parts of the pre-processing data stage is Data Labelling. In the data labeling process, we provide some identification to raw data (that may include an image, audio, text) and add some tags to it. These tags tell which class of object the data belongs to, which helps the ML model learn from this data and make the most accurate prediction.

Hence, we can define it as, "**_Data labelling is a process of adding some meaning to different types of datasets, so that it can be properly used to train a Machine Learning Model. Data labelling is also called as Data Annotation (however, there is minor difference between both of them)."_**

Data Labelling is required in the case of Supervised Learning, as in supervised learning techniques, we input the labeled data set into the model.

Labeled Data vs. Unlabelled Data
--------------------------------

In data labeling, data is labeled, but in machine learning, both labeled and unlabelled data are used. So, what is the difference between them?

*   Labeled data is data that has some predefined tags such as name, type, or number. For example, an image has an apple or banana. At the same time, unlabelled data contains no tags or no specified name.
*   Labeled data is used in Supervised Learning techniques, whereas Unlabelled data is used in Unsupervised Learning.
*   Labeled data is difficult to get, whereas Unalabled data is easy to acquire.

#### Note: Semi-supervised learning uses combined data, i.e., labeled and unlabelled data, to train the model, which reduces the difficulties of getting labeled data.

How does Data Labelling Work?
-----------------------------

Nowadays, most machine learning models use supervised learning techniques that map the input variable to an output variable and make predictions. For supervised learning, we need the labeled dataset to train the model so that it can make accurate predictions. The Data labeling begins with a process of **"Human-in-the-loop"** or **HITL** participation, where humans are asked to make a judgment for the given unlabelled data. For example, a human labeler may be asked to tag an image dataset, where "does the image contain a cat" is true.

![Data Labelling in Machine Learning](https://static.javatpoint.com/tutorial/machine-learning/images/data-labelling-in-machine-learning3.png)

With these human-provided labels, an ML model learns from the data and underlying patterns, which is known as the Model training process, and the trained model then can be used to make a prediction with new data/test data.

Approaches to Data Labelling
----------------------------

Data labeling is an important step while building the high-performance Machine Learning Model. Although the process of data labeling appears easy and simple, it is a bit critical to implement. Therefore, in order to use data labeling techniques, companies should consider multiple factors to find the best approach to labeling. Some common data labeling approaches are given as follows:

*   **Internal/In-house data labeling**  
    In-house data labeling is performed by data scientists or data engineers of the organization. It is one of the highest quality possible labeling approaches with greater accuracy and simplified tracking. However, it is a time-consuming approach, and it is suitable for companies with extensive resources.
*   **Synthetic Labeling**  
    In this approach, new project data is generated from the pre-existing dataset, which increases the quality of the data, and also time efficiency. However, this approach needs high computing power and resources, which enhances the overall cost.
*   **Programmatic Labeling**  
    Programmatic labeling is an automated process that reduces time consumption and requirement for human annotation as it uses a script. However, besides an automated process, it needs HITL as a part of the QA process to check the possible technical problems.
*   **Outsourcing**  
    Outsourcing is another popular approach to data labeling, where a team of external labelers is put together in which; most of them are freelancers. This approach can be the best choice for high-level temporary projects; however, it may be a time-consuming approach to develop and manage the freelance-oriented workflow. Although there are various freelancing platforms, such as Upwork, which provide complete candidate information to make the selection process easier, hiring managed data labeling teams provides pre-assessed staff and pre-built data labeling tools.
*   **Crowdsourcing**  
    Crowdsourcing is one of the fastest and most cost-effective approaches, as it has micro-tasking capabilities and web-based distribution. It obtains annotated data from a large number of freelancers who are registered on a crowdsourcing platform. The datasets that need to be annotated mostly contain data such as images of plants, animals, natural environment, which do not need additional expertise for the annotation. One of the popular examples of crowdsourced data labeling is Recaptcha.

Benefits and Challenges of Data Labelling
-----------------------------------------

Being an important concept of machine learning, data labeling has different benefits at the same time and also has some challenges. It can make an accurate prediction but is also a costly approach. Below are some benefits and challenges of Data labeling:

### Benefits

*   **Precise Predictions:** With accurate data labeling, models can be trained with better quality data and hence generate the expected output. Otherwise, if we provide poor data to the model, then it will generate abrupt results.
*   **Better Data Usability:** Data labeling techniques make the data more usable within a model**.** For example, the categorical variables can be reclassified as binary variables to make them more consumable for a model. Therefore, with the aggregation of data, the model can be optimized by reducing the number of variables. Further, high-quality data is always a top priority, whether it is to build computer vision models (i.e., putting bounding boxes around objects) or NLP models (i.e., classifying text for social sentiment).

### Challenges

Data labeling has various challenges, and some of the most common challenges are:

*   **Costly and time-consuming:**  
    Being one of the crucial steps of building Machine Learning models, data labeling is time consuming and costly process. For a completely automated process also, engineering teams will still need to set up data pipelines prior to data processing, and manual labeling will almost always be costly and time-consuming.
*   **Possibilities of Human-Error:**  
    The labeling processes and approaches are prone to human errors, including coding errors or manual entry errors, which degrades the quality of data. The low-quality data leads to inaccurate data processing and modeling. Hence, in order to maintain data quality, quality assurance checks are essential.

Use Cases of Data Labelling
---------------------------

As data labeling is one of the important concepts of machine learning, it has various use cases. Some of them are given below:

*   **Computer Vision**  
    Computer vision is a field of AI, which creates a computer vision model to derive meaningful information from an image, video, or any other visual input. It does this using training data that enable the computer model to identify key points in an image and detect the location of an object.  
    While creating a computer vision model, firstly, we need to label the images, pixels, or key points or create a border that fully encloses a digital image, known as a bounding box, to get the training dataset. For example, an image can be classified as content(what the image contains or what it is about), quality type(product vs. lifestyle), or pixels. This training dataset can then be used to train the computer vision model, which finds insights from the image and make a prediction.
*   **Natural Language Processing**  
    Natural language processing is a branch of computer science, and more specifically, the branch of Artificial Intelligence, which enable computers to understand the text and spoken work in order to communicate with a human. NLP models can be used for sentiment analysis, entity name recognition, and optical character recognition. For the NLP model, firstly, we need to manually identify the important part of the text and add specific labels/tags to them in order to generate the training dataset.
*   **Audio Processing**  
    Audio processing is a technique to process and transform each type of sound, including speeches, wildlife noises, alarms, breaking glass, etc., into a structured form so that this audio dataset can be used for Machine learning applications. For audio processing, firstly, we need to transcribe it into written text manually, and then we can find detailed information about the audio by adding tags and categorizing the data. This labeled and categorized dataset can now be used as the training dataset.

Best Practices for Data Labelling
---------------------------------

There are various techniques that help in improving the efficiency and accuracy of data labeling. Some of these techniques are as follows:

### Active Learning

The active Learning technique makes the data labeling more efficient by identifying the most appropriate dataset to be labeled by humans using different ML algorithms and Semi-supervised learning. Active Learning approaches include:

*   Membership query synthesis
*   Pool-based sampling
*   Stream-based selective sampling

### Transfer Learning

Using transfer learning, one or more pre-trained models from one dataset are applied to another. This can also include multi-task learning, in which tasks are learned back-to-back.

### Label Auditing

The label auditing technique is used to verify the accuracy of labels and update them as per requirement.

### Consensus

This technique calculates the rate of agreement between different labelers, either human or machine, on the given dataset. This is calculated as the sum of aggreging labels to the total number of labels per asset.

### Intuitive and streamlined task interfaces

It minimizes cognitive load and context switching for human labelers.

* * *