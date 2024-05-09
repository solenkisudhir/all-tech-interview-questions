# Machine Learning Random Forest Algorithm
Random Forest is a popular machine learning algorithm that belongs to the supervised learning technique. It can be used for both Classification and Regression problems in ML. It is based on the concept of **ensemble learning,** which is a process of _combining multiple classifiers to solve a complex problem and to improve the performance of the model._

As the name suggests, _**"Random Forest is a classifier that contains a number of decision trees on various subsets of the given dataset and takes the average to improve the predictive accuracy of that dataset."**_ Instead of relying on one decision tree, the random forest takes the prediction from each tree and based on the majority votes of predictions, and it predicts the final output.

**The greater number of trees in the forest leads to higher accuracy and prevents the problem of overfitting.**

The below diagram explains the working of the Random Forest algorithm:

![Random Forest Algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/random-forest-algorithm.png)

#### Note: To better understand the Random Forest Algorithm, you should have knowledge of the Decision Tree Algorithm.

Assumptions for Random Forest
-----------------------------

Since the random forest combines multiple trees to predict the class of the dataset, it is possible that some decision trees may predict the correct output, while others may not. But together, all the trees predict the correct output. Therefore, below are two assumptions for a better Random forest classifier:

*   There should be some actual values in the feature variable of the dataset so that the classifier can predict accurate results rather than a guessed result.
*   The predictions from each tree must have very low correlations.

Why use Random Forest?
----------------------

Below are some points that explain why we should use the Random Forest algorithm:

*   It takes less training time as compared to other algorithms.
*   It predicts output with high accuracy, even for the large dataset it runs efficiently.
*   It can also maintain accuracy when a large proportion of data is missing.

How does Random Forest algorithm work?
--------------------------------------

Random Forest works in two-phase first is to create the random forest by combining N decision tree, and second is to make predictions for each tree created in the first phase.

The Working process can be explained in the below steps and diagram:

**Step-1:** Select random K data points from the training set.

**Step-2:** Build the decision trees associated with the selected data points (Subsets).

**Step-3:** Choose the number N for decision trees that you want to build.

**Step-4:** Repeat Step 1 & 2.

**Step-5:** For new data points, find the predictions of each decision tree, and assign the new data points to the category that wins the majority votes.

The working of the algorithm can be better understood by the below example:

**Example:** Suppose there is a dataset that contains multiple fruit images. So, this dataset is given to the Random forest classifier. The dataset is divided into subsets and given to each decision tree. During the training phase, each decision tree produces a prediction result, and when a new data point occurs, then based on the majority of results, the Random Forest classifier predicts the final decision. Consider the below image:

![Random Forest Algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/random-forest-algorithm2.png)

Applications of Random Forest
-----------------------------

There are mainly four sectors where Random forest mostly used:

1.  **Banking:** Banking sector mostly uses this algorithm for the identification of loan risk.
2.  **Medicine:** With the help of this algorithm, disease trends and risks of the disease can be identified.
3.  **Land Use:** We can identify the areas of similar land use by this algorithm.
4.  **Marketing:** Marketing trends can be identified using this algorithm.

Advantages of Random Forest
---------------------------

*   Random Forest is capable of performing both Classification and Regression tasks.
*   It is capable of handling large datasets with high dimensionality.
*   It enhances the accuracy of the model and prevents the overfitting issue.

Disadvantages of Random Forest
------------------------------

*   Although random forest can be used for both classification and regression tasks, it is not more suitable for Regression tasks.

Python Implementation of Random Forest Algorithm
------------------------------------------------

Now we will implement the Random Forest Algorithm tree using Python. For this, we will use the same dataset "user\_data.csv", which we have used in previous classification models. By using the same dataset, we can compare the Random Forest classifier with other classification models such as [Decision tree Classifier,](https://www.javatpoint.com/machine-learning-decision-tree-classification-algorithm) [KNN,](https://www.javatpoint.com/k-nearest-neighbor-algorithm-for-machine-learning) [SVM,](https://www.javatpoint.com/machine-learning-support-vector-machine-algorithm) [Logistic Regression,](https://www.javatpoint.com/logistic-regression-in-machine-learning) etc.

Implementation Steps are given below:

*   Data Pre-processing step
*   Fitting the Random forest algorithm to the Training set
*   Predicting the test result
*   Test accuracy of the result (Creation of Confusion matrix)
*   Visualizing the test set result.

### 1.Data Pre-Processing Step:

Below is the code for the pre-processing step:

In the above code, we have pre-processed the data. Where we have loaded the dataset, which is given as:

![Random Forest Algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/random-forest-algorithm3.png)

### 2\. Fitting the Random Forest algorithm to the training set:

Now we will fit the Random forest algorithm to the training set. To fit it, we will import the **RandomForestClassifier** class from the **sklearn.ensemble** library. The code is given below:

In the above code, the classifier object takes below parameters:

*   **n\_estimators=** The required number of trees in the Random Forest. The default value is 10. We can choose any number but need to take care of the overfitting issue.
*   **criterion=** It is a function to analyze the accuracy of the split. Here we have taken "entropy" for the information gain.

**Output:**

```
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=10,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)

```


### 3\. Predicting the Test Set result

Since our model is fitted to the training set, so now we can predict the test result. For prediction, we will create a new prediction vector y\_pred. Below is the code for it:

**Output:**

The prediction vector is given as:

![Random Forest Algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/random-forest-algorithm4.png)

By checking the above prediction vector and test set real vector, we can determine the incorrect predictions done by the classifier.

### 4\. Creating the Confusion Matrix

Now we will create the confusion matrix to determine the correct and incorrect predictions. Below is the code for it:

**Output:**

![Random Forest Algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/random-forest-algorithm5.png)

As we can see in the above matrix, there are **4+4= 8 incorrect predictions** and **64+28= 92 correct predictions.**

### 5\. Visualizing the training Set result

Here we will visualize the training set result. To visualize the training set result we will plot a graph for the Random forest classifier. The classifier will predict yes or No for the users who have either Purchased or Not purchased the SUV car as we did in [Logistic Regression.](https://www.javatpoint.com/logistic-regression-in-machine-learning) Below is the code for it:

**Output:**

![Random Forest Algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/random-forest-algorithm6.png)

The above image is the visualization result for the Random Forest classifier working with the training set result. It is very much similar to the Decision tree classifier. Each data point corresponds to each user of the user\_data, and the purple and green regions are the prediction regions. The purple region is classified for the users who did not purchase the SUV car, and the green region is for the users who purchased the SUV.

So, in the Random Forest classifier, we have taken 10 trees that have predicted Yes or NO for the Purchased variable. The classifier took the majority of the predictions and provided the result.

### 6\. Visualizing the test set result

Now we will visualize the test set result. Below is the code for it:

**Output:**

![Random Forest Algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/random-forest-algorithm7.png)

The above image is the visualization result for the test set. We can check that there is a minimum number of incorrect predictions (8) without the Overfitting issue. We will get different results by changing the number of trees in the classifier.

* * *