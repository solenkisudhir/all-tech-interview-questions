# What is a Confusion Matrix in Machine Learning?
In machine learning, Classification is used to split data into categories. But after cleaning and preprocessing the data and training our model, how do we know if our classification model performs well? That is where a confusion matrix comes into the picture.Â 

A confusion matrix is used to measure the performance of a classifier in depth. In this simple guide to Confusion Matrix, we will get to understand and learn confusion matrices better.

What Are Confusion Matrices, and Why Do We Need Them?
-----------------------------------------------------

Classification Models have multiple categorical outputs. Most error measures will calculate the total error in our model, but we cannot find individual instances of errors in our model. The model might misclassify some categories more than others, but we cannot see this using a standard accuracy measure.

Furthermore, suppose there is a significant class imbalance in the given data. In that case, i.e., a class has more instances of data than the other classes, a model might predict the majority class for all cases and have a high accuracy score; when it is not predicting the minority classes. This is where confusion matrices are useful.

A confusion matrix presents a table layout of the different outcomes of the prediction and results of a classification problem and helps visualize its outcomes.

It plots a table of all the predicted and actual values of a classifier.

![basic layout](https://www.simplilearn.com/ice9/assets/form_opacity.png)

Figure 1: Basic layout of a Confusion Matrix

How to Create a 2x2 Confusion Matrix?
-------------------------------------

We can obtain four different combinations from the predicted and actual values of a classifier:

![confusion matrix](https://www.simplilearn.com/ice9/assets/form_opacity.png)

Figure 2: Confusion Matrix

*   True Positive: The number of times our actual positive values are equal to the predicted positive. You predicted a positive value, and it is correct.
*   False Positive: The number of times our model wrongly predicts negative values as positives. You predicted a negative value, and it is actually positive.
*   True Negative: The number of times our actual negative values are equal to predicted negative values. You predicted a negative value, and it is actually negative.
*   False Negative: The number of times our model wrongly predicts negative values as positives. You predicted a negative value, and it is actually positive.

Confusion Matrix Metrics
------------------------

![classifier](https://www.simplilearn.com/ice9/assets/form_opacity.png)

Figure 3: Confusion Matrix for a classifier

Consider a confusion matrix made for a classifier that classifies people based on whether they speak English or Spanish.

From the above diagram, we can see that:

True Positives (TP) = 86

True Negatives (TN) = 79

False Positives (FP) = 12

False Negatives (FN) = 10

Just from looking at the matrix, the performance of our model is not very clear. To find how accurate our model is, we use the following metrics:

*   Accuracy: The accuracy is used to find the portion of correctly classified values. It tells us how often our classifier is right. It is the sum of all true values divided by total values.

![accuracy](https://www.simplilearn.com/ice9/assets/form_opacity.png)

Figure 4: Accuracy

In this case:

Accuracy = (86 +79) / (86 + 79 + 12 + 10) = 0.8823 = 88.23%

*   Precision: Precision is used to calculate the model's ability to classify positive values correctly. It is the true positives divided by the total number of predicted positive values.

![precision](https://www.simplilearn.com/ice9/assets/form_opacity.png)

Figure 5: Precision

In this case,

Precision = 86 / (86 + 12) = 0.8775 = 87.75%

*   Recall: It is used to calculate the model's ability to predict positive values. "How often does the model predict the correct positive values?". It is the true positives divided by the total number of actual positive values.Â Â 

![recall](https://www.simplilearn.com/ice9/assets/form_opacity.png)Â Â 

Figure 6: RecallÂ  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â 

In this case,

Recall = 86 / (86 + 10) = 0.8983 = 89.83%

*   F1-Score: It is the harmonic mean of Recall and Precision. It is useful when you need to take both Precision and Recall into account.

![f1score](https://www.simplilearn.com/ice9/assets/form_opacity.png)

Figure 7: F1-Score

In this case,

F1-Score = (2\* 0.8775 \* 0.8983) / (0.8775 + 0.8983) = 0.8877 = 88.77%

Scaling a Confusion Matrix
--------------------------

To scale a confusion matrix, increase the number of rows and columns. All the True Positives will be along the diagonal. The other values will be False Positives or False Negatives.

Â ![scaling up](https://www.simplilearn.com/ice9/assets/form_opacity.png)

Figure 12: Scaling down our dataset

Now that we understand what a confusion matrix is and its inner working, let's explore how we find the accuracy of a model with a hands-on demo on confusion matrix with Python.

Confusion Matrix With Python
----------------------------

We'll build a logistic regression model using a heart attack dataset to predict if a patient is at risk of a heart attack.Â 

Depicted below is the dataset that we'll be using for this demonstration.

![heart attack](https://www.simplilearn.com/ice9/assets/form_opacity.png)

Figure 9: Heart Attack DatasetÂ  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â Â 

Letâ€™s import the necessary libraries to create our model.Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  ![importing](https://www.simplilearn.com/ice9/assets/form_opacity.png)Â  Â 

Figure 10: Importing Confusion Matrix in pythonÂ  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â Â 

We can import the confusion matrix function from sklearn.metrics. Letâ€™s split our dataset into the input features and target output dataset.Â 

![splitting data](https://www.simplilearn.com/ice9/assets/form_opacity.png) Â Â 

Figure 11: Splitting data into variables and target dataset

As we can see, our data contains a massive range of values, some are single digits, and some have three numbers. To make our calculations more straightforward, we will scale our data and reduce it to a small range of values using the Standard Scaler.

![scaling down](https://www.simplilearn.com/ice9/assets/form_opacity.png)

Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â 

Figure 12: Scaling down our dataset

Now, let's split our dataset into two: one to train our model and another to test our model. To do this, we use train\_test\_split imported from sklearn. Using a Logistic Regression Model, we will perform Classification on our train data and predict our test data to check the accuracy.

![performing classification](https://www.simplilearn.com/ice9/assets/form_opacity.png)

Figure 13: Performing classification

To find the accuracy of a confusion matrix and all other metrics, we can import accuracy\_score and classification\_report from the same library.

Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â ![accuracy classifier](https://www.simplilearn.com/ice9/assets/form_opacity.png)Â  Â  Â  Â  Â  Â Â 

Figure 14: Accuracy of classifier

The accuracy\_score gives us the accuracy of our classifier

![data](https://www.simplilearn.com/ice9/assets/form_opacity.png)

Figure 15: Confusion Matrix for data

Using the predicted values(pred) and our actual values(y\_test), we can create a confusion matrix with the confusion\_matrix function.

Then, using the ravel() method of our confusion\_matrix function, we can get the True Positive, True Negative, False Positive, and False Negative values.

![extracting](https://www.simplilearn.com/ice9/assets/form_opacity.png)

Figure 16: Extracting matrix value

![confusion metrics](https://www.simplilearn.com/ice9/assets/form_opacity.png)

Â Figure 17: Confusion Matrix MetricsÂ  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â 

Finally, using the classification\_report, we can find the values of various metrics of our confusion matrix.

> Looking forward to making a move to the programming field? Take up the [Caltech Post Graduate Program In AI And Machine Learning](https://www.simplilearn.com/artificial-intelligence-masters-program-training-course?source=GhPreviewCoursepages)Â and begin your career as a professional Python programmer

Conclusion
----------

In this article - The Best Guide to Confusion Matrix, we have looked at what a confusion matrix is and why we use confusion matrices. We then looked at how to create a 2X2 confusion matrix and calculate the confusion matrix metrics using it. We took a look at how confusion matrices can be scaled up to include more than two classification classes and finally got hands-on experience with confusion matrices by implementing them in [Python](https://www.simplilearn.com/learn-the-basics-of-python-article "Python").Â 

Was this article on the confusion matrix useful to you? Do you have any doubts or questions for us? Mention them in this article's comments section, and we'll have our experts answer them for you at the earliest!
