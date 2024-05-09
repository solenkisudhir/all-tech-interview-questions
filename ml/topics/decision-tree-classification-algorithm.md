# Decision Tree Algorithm in Machine Learning
Decision Tree Classification Algorithm
--------------------------------------

*   Decision Tree is a **Supervised learning technique** that can be used for both classification and Regression problems, but mostly it is preferred for solving Classification problems. It is a tree-structured classifier, where **internal nodes represent the features of a dataset, branches represent the decision rules** and **each leaf node represents the outcome.**
*   In a Decision tree, there are two nodes, which are the **Decision Node** and **Leaf Node.** Decision nodes are used to make any decision and have multiple branches, whereas Leaf nodes are the output of those decisions and do not contain any further branches.
*   The decisions or the test are performed on the basis of features of the given dataset.
*   **_It is a graphical representation for getting all the possible solutions to a problem/decision based on given conditions._**
*   It is called a decision tree because, similar to a tree, it starts with the root node, which expands on further branches and constructs a tree-like structure.
*   In order to build a tree, we use the **CART algorithm,** which stands for **Classification and Regression Tree algorithm.**
*   A decision tree simply asks a question, and based on the answer (Yes/No), it further split the tree into subtrees.
*   Below diagram explains the general structure of a decision tree:

#### Note: A decision tree can contain categorical data (YES/NO) as well as numeric data.

![Decision Tree Classification Algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/decision-tree-classification-algorithm.png)

Why use Decision Trees?
-----------------------

There are various algorithms in Machine learning, so choosing the best algorithm for the given dataset and problem is the main point to remember while creating a machine learning model. Below are the two reasons for using the Decision tree:

*   Decision Trees usually mimic human thinking ability while making a decision, so it is easy to understand.
*   The logic behind the decision tree can be easily understood because it shows a tree-like structure.

Decision Tree Terminologies
---------------------------

*   **Root Node:** Root node is from where the decision tree starts. It represents the entire dataset, which further gets divided into two or more homogeneous sets.
*   **Leaf Node:** Leaf nodes are the final output node, and the tree cannot be segregated further after getting a leaf node.
*   **Splitting:** Splitting is the process of dividing the decision node/root node into sub-nodes according to the given conditions.
*   **Branch/Sub Tree:** A tree formed by splitting the tree.
*   **Pruning:** Pruning is the process of removing the unwanted branches from the tree.
*   **Parent/Child node:** The root node of the tree is called the parent node, and other nodes are called the child nodes.

**How does the Decision Tree algorithm Work?**

In a decision tree, for predicting the class of the given dataset, the algorithm starts from the root node of the tree. This algorithm compares the values of root attribute with the record (real dataset) attribute and, based on the comparison, follows the branch and jumps to the next node.

For the next node, the algorithm again compares the attribute value with the other sub-nodes and move further. It continues the process until it reaches the leaf node of the tree. The complete process can be better understood using the below algorithm:

*   **Step-1:** Begin the tree with the root node, says S, which contains the complete dataset.
*   **Step-2:** Find the best attribute in the dataset using **Attribute Selection Measure (ASM).**
*   **Step-3:** Divide the S into subsets that contains possible values for the best attributes.
*   **Step-4:** Generate the decision tree node, which contains the best attribute.
*   **Step-5:** Recursively make new decision trees using the subsets of the dataset created in step -3. Continue this process until a stage is reached where you cannot further classify the nodes and called the final node as a leaf node.

**Example:** Suppose there is a candidate who has a job offer and wants to decide whether he should accept the offer or Not. So, to solve this problem, the decision tree starts with the root node (Salary attribute by ASM). The root node splits further into the next decision node (distance from the office) and one leaf node based on the corresponding labels. The next decision node further gets split into one decision node (Cab facility) and one leaf node. Finally, the decision node splits into two leaf nodes (Accepted offers and Declined offer). Consider the below diagram:

![Decision Tree Classification Algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/decision-tree-classification-algorithm2.png)

Attribute Selection Measures
----------------------------

While implementing a Decision tree, the main issue arises that how to select the best attribute for the root node and for sub-nodes. So, to solve such problems there is a technique which is called as **Attribute selection measure or ASM.** By this measurement, we can easily select the best attribute for the nodes of the tree. There are two popular techniques for ASM, which are:

*   **Information Gain**
*   **Gini Index**

### 1\. Information Gain:

*   Information gain is the measurement of changes in entropy after the segmentation of a dataset based on an attribute.
*   It calculates how much information a feature provides us about a class.
*   According to the value of information gain, we split the node and build the decision tree.
*   A decision tree algorithm always tries to maximize the value of information gain, and a node/attribute having the highest information gain is split first. It can be calculated using the below formula:

**Entropy:** Entropy is a metric to measure the impurity in a given attribute. It specifies randomness in data. Entropy can be calculated as:

```
Entropy(s)= -P(yes)log2 P(yes)- P(no) log2 P(no)
```


**Where,**

*   **S= Total number of samples**
*   **P(yes)= probability of yes**
*   **P(no)= probability of no**

### 2\. Gini Index:

*   Gini index is a measure of impurity or purity used while creating a decision tree in the CART(Classification and Regression Tree) algorithm.
*   An attribute with the low Gini index should be preferred as compared to the high Gini index.
*   It only creates binary splits, and the CART algorithm uses the Gini index to create binary splits.
*   Gini index can be calculated using the below formula:

Pruning: Getting an Optimal Decision tree
-----------------------------------------

_Pruning is a process of deleting the unnecessary nodes from a tree in order to get the optimal decision tree._

A too-large tree increases the risk of overfitting, and a small tree may not capture all the important features of the dataset. Therefore, a technique that decreases the size of the learning tree without reducing accuracy is known as Pruning. There are mainly two types of tree **pruning** technology used:

*   **Cost Complexity Pruning**
*   **Reduced Error Pruning.**

Advantages of the Decision Tree
-------------------------------

*   It is simple to understand as it follows the same process which a human follow while making any decision in real-life.
*   It can be very useful for solving decision-related problems.
*   It helps to think about all the possible outcomes for a problem.
*   There is less requirement of data cleaning compared to other algorithms.

Disadvantages of the Decision Tree
----------------------------------

*   The decision tree contains lots of layers, which makes it complex.
*   It may have an overfitting issue, which can be resolved using the **Random Forest algorithm.**
*   For more class labels, the computational complexity of the decision tree may increase.

Python Implementation of Decision Tree
--------------------------------------

Now we will implement the Decision tree using Python. For this, we will use the dataset "**user\_data.csv**," which we have used in previous classification models. By using the same dataset, we can compare the Decision tree classifier with other classification models such as [KNN](https://www.javatpoint.com/k-nearest-neighbor-algorithm-for-machine-learning) [SVM,](https://www.javatpoint.com/machine-learning-support-vector-machine-algorithm) [LogisticRegression,](https://www.javatpoint.com/logistic-regression-in-machine-learning) etc.

Steps will also remain the same, which are given below:

*   **Data Pre-processing step**
*   **Fitting a Decision-Tree algorithm to the Training set**
*   **Predicting the test result**
*   **Test accuracy of the result(Creation of Confusion matrix)**
*   **Visualizing the test set result.**

### 1\. Data Pre-Processing Step:

Below is the code for the pre-processing step:

In the above code, we have pre-processed the data. Where we have loaded the dataset, which is given as:

![Decision Tree Classification Algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/decision-tree-classification-algorithm3.png)

### 2\. Fitting a Decision-Tree algorithm to the Training set

Now we will fit the model to the training set. For this, we will import the **DecisionTreeClassifier** class from **sklearn.tree** library. Below is the code for it:

In the above code, we have created a classifier object, in which we have passed two main parameters;

*   **"criterion='entropy':** Criterion is used to measure the quality of split, which is calculated by information gain given by entropy.
*   **random\_state=0":** For generating the random states.

Below is the output for this:

```
Out[8]: 
DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
max_features=None, max_leaf_nodes=None,
min_impurity_decrease=0.0, min_impurity_split=None,
min_samples_leaf=1, min_samples_split=2,
min_weight_fraction_leaf=0.0, presort=False,
                       random_state=0, splitter='best')

```


### 3\. Predicting the test result

Now we will predict the test set result. We will create a new prediction vector **y\_pred.** Below is the code for it:

**Output:**

In the below output image, the predicted output and real test output are given. We can clearly see that there are some values in the prediction vector, which are different from the real vector values. These are prediction errors.

![Decision Tree Classification Algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/decision-tree-classification-algorithm4.png)

### 4\. Test accuracy of the result (Creation of Confusion matrix)

In the above output, we have seen that there were some incorrect predictions, so if we want to know the number of correct and incorrect predictions, we need to use the confusion matrix. Below is the code for it:

**Output:**

![Decision Tree Classification Algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/decision-tree-classification-algorithm5.png)

In the above output image, we can see the confusion matrix, which has **6+3= 9 incorrect predictions** and**62+29=91 correct predictions. Therefore, we can say that compared to other classification models, the Decision Tree classifier made a good prediction.**

### 5\. Visualizing the training set result:

Here we will visualize the training set result. To visualize the training set result we will plot a graph for the decision tree classifier. The classifier will predict yes or No for the users who have either Purchased or Not purchased the SUV car as we did in [Logistic Regression.](https://www.javatpoint.com/logistic-regression-in-machine-learning) Below is the code for it:

**Output:**

![Decision Tree Classification Algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/decision-tree-classification-algorithm6.png)

The above output is completely different from the rest classification models. It has both vertical and horizontal lines that are splitting the dataset according to the age and estimated salary variable.

As we can see, the tree is trying to capture each dataset, which is the case of overfitting.

### 6\. Visualizing the test set result:

Visualization of test set result will be similar to the visualization of the training set except that the training set will be replaced with the test set.

**Output:**

![Decision Tree Classification Algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/decision-tree-classification-algorithm7.png)

As we can see in the above image that there are some green data points within the purple region and vice versa. So, these are the incorrect predictions which we have discussed in the confusion matrix.

* * *