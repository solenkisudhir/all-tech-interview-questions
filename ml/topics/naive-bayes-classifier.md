# Naive Bayes Classifier in Machine Learning

Naïve Bayes Classifier Algorithm
--------------------------------

*   Naïve Bayes algorithm is a supervised learning algorithm, which is based on **Bayes theorem** and used for solving classification problems.
*   It is mainly used in _text classification_ that includes a high-dimensional training dataset.
*   Naïve Bayes Classifier is one of the simple and most effective Classification algorithms which helps in building the fast machine learning models that can make quick predictions.
*   **It is a probabilistic classifier, which means it predicts on the basis of the probability of an object**.
*   Some popular examples of Naïve Bayes Algorithm are **spam filtration, Sentimental analysis, and classifying articles**.

Why is it called Naïve Bayes?
-----------------------------

The Naïve Bayes algorithm is comprised of two words Naïve and Bayes, Which can be described as:

*   **Naïve**: It is called Naïve because it assumes that the occurrence of a certain feature is independent of the occurrence of other features. Such as if the fruit is identified on the bases of color, shape, and taste, then red, spherical, and sweet fruit is recognized as an apple. Hence each feature individually contributes to identify that it is an apple without depending on each other.
*   **Bayes**: It is called Bayes because it depends on the principle of [Bayes' Theorem](https://www.javatpoint.com/bayes-theorem-in-artifical-intelligence).

Bayes' Theorem:
---------------

*   Bayes' theorem is also known as **Bayes' Rule** or **Bayes' law**, which is used to determine the probability of a hypothesis with prior knowledge. It depends on the conditional probability.
*   The formula for Bayes' theorem is given as:

![Naïve Bayes Classifier Algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/naive-bayes-classifier-algorithm.png)

**Where,**

**P(A|B) is Posterior probability**: Probability of hypothesis A on the observed event B.

**P(B|A) is Likelihood probability**: Probability of the evidence given that the probability of a hypothesis is true.

**P(A) is Prior Probability**: Probability of hypothesis before observing the evidence.

**P(B) is Marginal Probability**: Probability of Evidence.

Working of Naïve Bayes' Classifier:
-----------------------------------

Working of Naïve Bayes' Classifier can be understood with the help of the below example:

Suppose we have a dataset of **weather conditions** and corresponding target variable "**Play**". So using this dataset we need to decide that whether we should play or not on a particular day according to the weather conditions. So to solve this problem, we need to follow the below steps:

1.  Convert the given dataset into frequency tables.
2.  Generate Likelihood table by finding the probabilities of given features.
3.  Now, use Bayes theorem to calculate the posterior probability.

**Problem**: If the weather is sunny, then the Player should play or not?

**Solution**: To solve this, first consider the below dataset:


|   |Outlook |Play|
|---|--------|----|
|0  |Rainy   |Yes |
|1  |Sunny   |Yes |
|2  |Overcast|Yes |
|3  |Overcast|Yes |
|4  |Sunny   |No  |
|5  |Rainy   |Yes |
|6  |Sunny   |Yes |
|7  |Overcast|Yes |
|8  |Rainy   |No  |
|9  |Sunny   |No  |
|10 |Sunny   |Yes |
|11 |Rainy   |No  |
|12 |Overcast|Yes |
|13 |Overcast|Yes |


**Frequency table for the Weather Conditions:**


|Weather |Yes|No |
|--------|---|---|
|Overcast|5  |0  |
|Rainy   |2  |2  |
|Sunny   |3  |2  |
|Total   |10 |5  |


**Likelihood table weather condition:**


|Weather |No       |Yes       |          |
|--------|---------|----------|----------|
|Overcast|0        |5         |5/14= 0.35|
|Rainy   |2        |2         |4/14=0.29 |
|Sunny   |2        |3         |5/14=0.35 |
|All     |4/14=0.29|10/14=0.71|          |


**Applying Bayes'theorem:**

**P(Yes|Sunny)= P(Sunny|Yes)\*P(Yes)/P(Sunny)**

P(Sunny|Yes)= 3/10= 0.3

P(Sunny)= 0.35

P(Yes)=0.71

So P(Yes|Sunny) = 0.3\*0.71/0.35= **0.60**

**P(No|Sunny)= P(Sunny|No)\*P(No)/P(Sunny)**

P(Sunny|NO)= 2/4=0.5

P(No)= 0.29

P(Sunny)= 0.35

So P(No|Sunny)= 0.5\*0.29/0.35 = **0.41**

So as we can see from the above calculation that **P(Yes|Sunny)>P(No|Sunny)**

**Hence on a Sunny day, Player can play the game.**

### Advantages of Naïve Bayes Classifier:

*   Naïve Bayes is one of the fast and easy ML algorithms to predict a class of datasets.
*   It can be used for Binary as well as Multi-class Classifications.
*   It performs well in Multi-class predictions as compared to the other Algorithms.
*   It is the most popular choice for **text classification problems**.

### Disadvantages of Naïve Bayes Classifier:

*   Naive Bayes assumes that all features are independent or unrelated, so it cannot learn the relationship between features.

### Applications of Naïve Bayes Classifier:

*   It is used for **Credit Scoring**.
*   It is used in **medical data classification**.
*   It can be used in **real-time predictions** because Naïve Bayes Classifier is an eager learner.
*   It is used in Text classification such as **Spam filtering** and **Sentiment analysis**.

Types of Naïve Bayes Model:
---------------------------

There are three types of Naive Bayes Model, which are given below:

*   **Gaussian**: The Gaussian model assumes that features follow a normal distribution. This means if predictors take continuous values instead of discrete, then the model assumes that these values are sampled from the Gaussian distribution.
*   **Multinomial**: The Multinomial Naïve Bayes classifier is used when the data is multinomial distributed. It is primarily used for document classification problems, it means a particular document belongs to which category such as Sports, Politics, education, etc.  
    The classifier uses the frequency of words for the predictors.
*   **Bernoulli**: The Bernoulli classifier works similar to the Multinomial classifier, but the predictor variables are the independent Booleans variables. Such as if a particular word is present or not in a document. This model is also famous for document classification tasks.

Python Implementation of the Naïve Bayes algorithm:
---------------------------------------------------

Now we will implement a Naive Bayes Algorithm using Python. So for this, we will use the "**user\_data**" **dataset**, which we have used in our other classification model. Therefore we can easily compare the Naive Bayes model with the other models.

### Steps to implement:

*   Data Pre-processing step
*   Fitting Naive Bayes to the Training set
*   Predicting the test result
*   Test accuracy of the result(Creation of Confusion matrix)
*   Visualizing the test set result.

### 1) Data Pre-processing step:

In this step, we will pre-process/prepare the data so that we can use it efficiently in our code. It is similar as we did in [data-pre-processing](https://www.javatpoint.com/data-preprocessing-machine-learning). The code for this is given below:

In the above code, we have loaded the dataset into our program using "**dataset = pd.read\_csv('user\_data.csv')**. The loaded dataset is divided into training and test set, and then we have scaled the feature variable.

The output for the dataset is given as:

![Naïve Bayes Classifier Algorithm 1](https://static.javatpoint.com/tutorial/machine-learning/images/naive-bayes-classifier-algorithm-1.png)

### 2) Fitting Naive Bayes to the Training Set:

After the pre-processing step, now we will fit the Naive Bayes model to the Training set. Below is the code for it:

In the above code, we have used the **GaussianNB classifier** to fit it to the training dataset. We can also use other classifiers as per our requirement.

**Output:**

```
Out[6]: GaussianNB(priors=None, var_smoothing=1e-09)

```


### 3) Prediction of the test set result:

Now we will predict the test set result. For this, we will create a new predictor variable **y\_pred**, and will use the predict function to make the predictions.

**Output:**

![Naïve Bayes Classifier Algorithm 2](https://static.javatpoint.com/tutorial/machine-learning/images/naive-bayes-classifier-algorithm-2.png)

The above output shows the result for prediction vector **y\_pred** and real vector y\_test. We can see that some predications are different from the real values, which are the incorrect predictions.

### 4) Creating Confusion Matrix:

Now we will check the accuracy of the Naive Bayes classifier using the Confusion matrix. Below is the code for it:

**Output:**

![Naïve Bayes Classifier Algorithm 3](https://static.javatpoint.com/tutorial/machine-learning/images/naive-bayes-classifier-algorithm-3.png)

As we can see in the above confusion matrix output, there are 7+3= 10 incorrect predictions, and 65+25=90 correct predictions.

### 5) Visualizing the training set result:

Next we will visualize the training set result using Naïve Bayes Classifier. Below is the code for it:

**Output:**

![Naïve Bayes Classifier Algorithm 4](https://static.javatpoint.com/tutorial/machine-learning/images/naive-bayes-classifier-algorithm-4.png)

In the above output we can see that the Naïve Bayes classifier has segregated the data points with the fine boundary. It is Gaussian curve as we have used **GaussianNB** classifier in our code.

### 6) Visualizing the Test set result:

**Output:**

![Naïve Bayes Classifier Algorithm 5](https://static.javatpoint.com/tutorial/machine-learning/images/naive-bayes-classifier-algorithm-5.png)

The above output is final output for test set data. As we can see the classifier has created a Gaussian curve to divide the "purchased" and "not purchased" variables. There are some wrong predictions which we have calculated in Confusion matrix. But still it is pretty good classifier.

* * *

