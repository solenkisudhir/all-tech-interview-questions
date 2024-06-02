# How to Handle Imbalanced Classes in Machine Learning
In machine learning, “imbalanced classes” is a familiar problem particularly occurring in classification when we have datasets with an unequal ratio of data points in each class.

Training of model becomes much trickier as typical accuracy is no longer a reliable metric for measuring the performance of the model. Now if the number of data points in minority class is much less, then it may end up being completely ignored during training.

Problems with the imbalanced data
---------------------------------

Unbalanced class distributions present a barrier, even though many [machine learning](https://www.geeksforgeeks.org/machine-learning/) algorithms work best when there are nearly equal numbers of samples in each class. A model may appear to have high accuracy in these situations if it primarily predicts the majority class. In such cases, having high accuracy becomes deceptive. Sadly, the minority class—which is frequently the main focus of model creation—is ignored by this strategy. In the event that 99% of the data pertains to the majority class, for example, simple classification models such as logistic regression or decision trees may find it difficult to recognize and precisely forecast occurrences from the minority class.

Class Imbalance Handling in Machine Learning
--------------------------------------------

[Resampling](https://www.geeksforgeeks.org/introduction-to-resampling-methods/), which modifies the sample distribution, is a frequently used technique for handling very unbalanced datasets. This can be accomplished by either over-sampling, which adds more examples from the minority class, or under-sampling, which removes samples from the majority class. One method for reducing the difficulties caused by severely skewed datasets is resampling, which balances the class distribution.

Using strategies like over- and under-sampling to balance classes has advantages, but there are also disadvantages.

A fundamental method of [over-sampling](https://www.geeksforgeeks.org/introduction-to-resampling-methods/) is to replicate random records from the minority class, which may cause overfitting.

On the other hand, information loss may occur from the simple technique of eliminating random records from the majority class in an undersampled situation.

In [Up-sampling](https://www.geeksforgeeks.org/spatial-resolution-down-sampling-and-up-sampling-in-image-processing/), samples from minority classes are randomly duplicated so as to achieve equivalence with the majority class. There are many methods used for achieving this.

### ****1\. Using Random Under-Sampling****

When observations from the majority class are eliminated until the majority and minority classes are balanced, this is known as [undersampling](https://www.geeksforgeeks.org/ml-handling-imbalanced-data-with-smote-and-near-miss-algorithm-in-python/).

Undersampling has advantages when working with large datasets, especially ones with millions of rows, but there is a risk that important information will be lost during the removal process.

****Example :**** 

Python3`   ```
# Importing scikit-learn, pandas library
from sklearn.utils import resample
from sklearn.datasets import make_classification
import pandas as pd

# Making DataFrame having 100
# dummy samples with 4 features 
# Divided in 2 classes in a ratio of 80:20 
X, y = make_classification(n_classes=2, 
                           weights=[0.8, 0.2],
                           n_features=4, 
                           n_samples=100, 
                           random_state=42)

df = pd.DataFrame(X, columns=['feature_1',
                              'feature_2',
                              'feature_3',
                              'feature_4'])
df['balance'] = y
print(df)

# Let df represent the dataset
# Dividing majority and minority classes
df_major = df[df.balance == 0]
df_minor = df[df.balance == 1]

# Upsampling minority class
df_minor_sample = resample(df_minor,
                           
                           # Upsample with replacement
                           replace=True,    
                           
                           # Number to match majority class
                           n_samples=80,   
                           random_state=42)

# Combine majority and upsampled minority class
df_sample = pd.concat([df_major, df_minor_sample])

# Display count of data points in both class
print(df_sample.balance.value_counts())

```
     `

****Output:****

```
    feature_1  feature_2  feature_3  feature_4  balance
0   -1.053839  -1.027544  -0.329294   0.826007        1
1    1.569317   1.306542  -0.239385  -0.331376        0
2   -0.658926  -0.357633   0.723682  -0.628277        0
3   -0.136856   0.460938   1.896911  -2.281386        0
4   -0.048629   0.502301   1.778730  -2.171053        0
..        ...        ...        ...        ...      ...
95  -2.241820  -1.248690   2.357902  -2.009185        0
96   0.573042   0.362054  -0.462814   0.341294        1
97  -0.375121  -0.149518   0.588465  -0.575002        0
98   1.042518   1.058239   0.461945  -0.984846        0
99  -0.121203  -0.043997   0.204211  -0.203119        0
[100 rows x 5 columns]
0    80
1    80
Name: balance, dtype: int64
```


****Explanation :**** 

*   Firstly, we’ll divide the data points from each class into separate DataFrames.
*   After this, the minority class is resampled ****with replacement**** by setting the number of data points equivalent to that of the majority class.
*   In the end, we’ll concatenate the original majority class DataFrame and up-sampled minority class DataFrame.

### ****2\. Using RandomOverSampler:****

Oversampling is the process of adding more copies to the minority class. When dealing with constrained data resources, this approach is helpful. Overfitting and decreased generalization performance on the test set are potential drawbacks of oversampling, though.

This can be done with the help of the [RandomOverSampler](https://www.geeksforgeeks.org/ml-handling-imbalanced-data-with-smote-and-near-miss-algorithm-in-python/) method present in imblearn. This function randomly generates new data points belonging to the minority class with replacement (by default).

> ****Syntax:**** RandomOverSampler(sampling\_strategy=’auto’, random\_state=None, shrinkage=None)
> 
> ****Parameters:****
> 
> *   ****sampling\_strategy:**** Sampling Information for dataset.Some Values are- ‘minority’: only minority class ‘not minority’: all classes except minority class, ‘not majority’: all classes except majority class, ‘all’: all classes,  ‘auto’: similar to ‘not majority’, Default value is ‘auto’
> *   ****random\_state:**** Used for shuffling the data. If a positive non-zero number is given then it shuffles otherwise not. Default value is None.
> *   ****shrinkage:**** Parameter controlling the shrinkage. Values are: float: Shrinkage factor applied on all classes. dict: Every class will have a specific shrinkage factor. None: Shrinkage= 0. Default value is None.

****Implementation of RandomOverSampler****

Python3`   ```
# Importing imblearn,scikit-learn library
from imblearn.over_sampling import RandomOverSampler
from sklearn.datasets import make_classification

# Making Dataset having 100
# dummy samples with 4 features 
# Divided in 2 classes in a ratio of 80:20 
X, y = make_classification(n_classes=2, 
                           weights=[0.8, 0.2],
                           n_features=4, 
                           n_samples=100, 
                           random_state=42)

# Printing number of samples in
# each class before Over-Sampling
t = [(d) for d in y if d==0]
s = [(d) for d in y if d==1]
print('Before Over-Sampling: ')
print('Samples in class 0: ',len(t))
print('Samples in class 1: ',len(s))

# Over Sampling Minority class
OverS = RandomOverSampler(random_state=42)

# Fit predictor (x variable)
# and target (y variable) using fit_resample()
X_Over, Y_Over = OverS.fit_resample(X, y)

# Printing number of samples in
# each class after Over-Sampling
t = [(d) for d in Y_Over if d==0]
s = [(d) for d in Y_Over if d==1]
print('After Over-Sampling: ')
print('Samples in class 0: ',len(t))
print('Samples in class 1: ',len(s))

```
     `

****Output:****

```
Before Over-Sampling: 
Samples in class 0:  80
Samples in class 1:  20
After Over-Sampling: 
Samples in class 0:  80
Samples in class 1:  80
```


*   This code illustrates how to use imbalanced-learn’s RandomOverSampler to address class imbalance in a dataset.
*   By creating artificial samples for the minority class, it improves the balance of the class distribution.
*   For comparison, the number of samples in each class is printed both before and after oversampling.

Balancing data with the Imbalanced-Learn module in Python
---------------------------------------------------------

In the world of fixing imbalanced data, there are some smart tricks. Scientists have come up with advanced methods to handle this issue.

For example, one clever way involves grouping together the majority class data and then carefully removing some of it. This helps keep the important details while making things more balanced. Another cool technique for adding more data to the minority class is not just making exact copies but tweaking them a bit to create a more diverse set.

To try out these methods, we can use a handy Python library called [imbalanced-learn](https://www.geeksforgeeks.org/imbalanced-learn-module-in-python/). It plays well with scikit-learn and is part of some cool projects in the world of scikit-learn.

```
import imblearn
```


### 3\. Random Under-Sampling with Imblearn

There’s a library called imblearn, which is super helpful for fixing imbalanced datasets and making your models work better.

One good thing in imblearn is RandomUnderSampler. It’s a quick and simple way to even out the data by randomly choosing some data from the classes we want to balance. Basically, it grabs a bunch of samples from the majority class (or classes) in a random way.

> ****Syntax:**** RandomUnderSampler(sampling\_strategy=’auto’, random\_state=None, replacement=False)
> 
> ****Parameters:****
> 
> *   ****sampling\_strategy:**** Sampling Information for dataset.
> *   ****random\_state:**** Used for shuffling the data. If positive non zero number is given then it shuffles otherwise not. Default value is None.
> *   ****replacement:**** Implements resampling with or without replacement. Boolean type of value. Default value is False.

#### Implementation of Random Under-Sampling with Imblearn

Python3`   ```
# Importing imblearn library
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import make_classification

# Making Dataset having
# 100 dummy samples with 4 features
# Divided in 2 classes in a ratio of 80:20
X, y = make_classification(n_classes=2,
                           weights=[0.8, 0.2],
                           n_features=4,
                           n_samples=100,
                           random_state=42)

# Printing number of samples
# in each class before Under-Sampling
t = [(d) for d in y if d == 0]
s = [(d) for d in y if d == 1]
print('Before Under-Sampling: ')
print('Samples in class 0: ', len(t))
print('Samples in class 1: ', len(s))

# Down-Sampling majority class
UnderS = RandomUnderSampler(random_state=42,
                            replacement=True)

# Fit predictor (x variable)
# and target (y variable) using fit_resample()
X_Under, Y_Under = UnderS.fit_resample(X, y)

# Printing number of samples in
# each class after Under-Sampling
t = [(d) for d in Y_Under if d == 0]
s = [(d) for d in Y_Under if d == 1]
print('After Under-Sampling: ')
print('Samples in class 0: ', len(t))
print('Samples in class 1: ', len(s))

```
     `

****Output:****

```
Before Under-Sampling: 
Samples in class 0:  80
Samples in class 1:  20
After Under-Sampling: 
Samples in class 0:  20
Samples in class 1:  20
```


*   This code illustrates how to rectify class imbalance in a dataset using RandomUnderSampler from imbalanced-learn.
*   By removing samples at random from the majority class, it improves the balance of the class distribution.
*   For comparison, the number of samples in each class is printed both before and after undersampling.

### 4\. Random Over-Sampling with Imblearn

To address imbalanced data, one approach is to create more examples for the minority classes. A simple way to do this is by randomly selecting and duplicating existing samples. The RandomOverSampler provides a method to implement this strategy.

#### Implementation of Random Over-Sampling with Imblearn

Python3`   ```
# Importing imblearn library for over-sampling
from imblearn.over_sampling import RandomOverSampler
from sklearn.datasets import make_classification

# Making a dataset with 100 dummy samples and 4 features, divided into 2 classes in a ratio of 80:20
X, y = make_classification(n_classes=2, weights=[
                           0.8, 0.2], n_features=4, n_samples=100, random_state=42)

# Printing the number of samples in each class before Over-Sampling
class_0_samples = sum(1 for label in y if label == 0)
class_1_samples = sum(1 for label in y if label == 1)
print('Before Over-Sampling:')
print('Samples in class 0:', class_0_samples)
print('Samples in class 1:', class_1_samples)

# Applying random over-sampling
over_sampler = RandomOverSampler(random_state=42)
X_over, y_over = over_sampler.fit_resample(X, y)

# Printing the number of samples in each class after Over-Sampling
class_0_samples_after = sum(1 for label in y_over if label == 0)
class_1_samples_after = sum(1 for label in y_over if label == 1)
print('After Over-Sampling:')
print('Samples in class 0:', class_0_samples_after)
print('Samples in class 1:', class_1_samples_after)

```
     `

****Output:****

```
Before Over-Sampling:
Samples in class 0: 80
Samples in class 1: 20
After Over-Sampling:
Samples in class 0: 80
Samples in class 1: 80
```


*   This code corrects class imbalance in a dataset by using RandomOverSampler from imbalanced-learn.
*   In order to achieve a more balanced distribution, samples from the minority class are randomly duplicated.
*   For comparison, the number of samples in each class is printed both before and after oversampling.

### ****5\. Synthetic Minority Oversampling Technique (SMOTE)****

[SMOTE](https://www.geeksforgeeks.org/ml-handling-imbalanced-data-with-smote-and-near-miss-algorithm-in-python/) is used to generate artificial/synthetic samples for the minority class. This technique works by randomly choosing a sample from a minority class and determining K-Nearest Neighbors for this sample, then the artificial sample is added between the picked sample and its neighbors. This function is present in imblearn module.

> ****Syntax:**** SMOTE(sampling\_strategy=’auto’, random\_state=None, k\_neighbors=5, n\_jobs=None)
> 
> ****Parameters:****
> 
> *   ****sampling\_strategy:**** Sampling Information for dataset
> *   ****random\_state:**** Used for shuffling the data. If positive non zero number is given then it shuffles otherwise not. Default value is None.
> *   ****k\_neighbors:**** Number count of nearest neighbours used to generate artificial/synthetic samples. Default value is 5
> *   ****n\_jobs:**** Number of CPU cores to be used. Default value is None. None here means 1 not 0.

#### Working of SMOTE Algorithm

An algorithm called SMOTE (Synthetic Minority Over-sampling Technique) is used to rectify dataset class imbalances. To put it briefly, SMOTE generates synthetic samples for the minority class. Here is a quick rundown of how it functions:

*   ****Identify minority class instances:**** Determine which dataset instances belong to the minority class.
*   ****Select a Minority Instance:**** Select a minority instance at random from the dataset.
*   ****Find Nearest Neighbors:**** Determine which members of the minority class are the selected instance’s [k-nearest neighbors.](https://www.geeksforgeeks.org/k-nearest-neighbors-with-python-ml/)
*   ****Generate Synthetic Samples:**** By dividing the selected instance by the distance between it and its closest neighbors, create synthetic instances. Usually, to accomplish this, a synthetic instance is made along the line that connects the selected instance and the neighbor, and a random neighbor is chosen.

****Implementation of SMOTE(Synthetic Minority Oversampling Technique)****

Python3`   ```
# Importing imblearn, scikit-learn library
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification

# Making Dataset having
# 100 dummy samples with 4 features 
# Divided in 2 classes in a ratio of 80:20 
X, y = make_classification(n_classes=2, 
                           weights=[0.8, 0.2],
                           n_features=4, 
                           n_samples=100, 
                           random_state=42)

# Printing number of samples in
# each class before Over-Sampling
t = [(d) for d in y if d==0]
s = [(d) for d in y if d==1]
print('Before Over-Sampling: ')
print('Samples in class 0: ',len(t))
print('Samples in class 1: ',len(s))


# Making an instance of SMOTE class 
# For oversampling of minority class
smote = SMOTE()

# Fit predictor (x variable)
# and target (y variable) using fit_resample()
X_OverSmote, Y_OverSmote = smote.fit_resample(X, y)

# Printing number of samples
# in each class after Over-Sampling
t = [(d) for d in Y_OverSmote if d==0]
s = [(d) for d in Y_OverSmote if d==1]
print('After Over-Sampling: ')
print('Samples in class 0: ',len(t))
print('Samples in class 1: ',len(s))

```
     `

****Output:****

```
Before Over-Sampling: 
Samples in class 0:  80
Samples in class 1:  20
After Over-Sampling: 
Samples in class 0:  80
Samples in class 1:  80
```


****Explanation:****

*   Minority class is given as input vector.
*   Determine its K-Nearest Neighbours
*   Pick one of these neighbors and place an artificial sample point anywhere between the neighbor and sample point under consideration.
*   Repeat till the dataset gets balanced.

### ****6\.****  NearMiss

[NearMiss](https://www.geeksforgeeks.org/ml-handling-imbalanced-data-with-smote-and-near-miss-algorithm-in-python/) is an under-sampling technique that aims to equalize the class distribution by considering the distances between instances, ensuring that the majority class becomes comparable to the minority class.

#### Implementation of NearMiss

Python3`   ```
# Importing necessary libraries
from imblearn.under_sampling import NearMiss
from sklearn.datasets import make_classification

# Creating a synthetic imbalanced dataset
X, y = make_classification(n_classes=2, weights=[
                           0.8, 0.2], n_features=4, n_samples=100, random_state=42)

# Printing number of samples in each class before Under-Sampling
print('Before Under-Sampling: ')
print('Samples in class 0:', sum(y == 0))
print('Samples in class 1:', sum(y == 1))

# Creating an instance of NearMiss
nm = NearMiss()

# Fit predictor (X) and target (y) using fit_resample()
X_nearmiss, y_nearmiss = nm.fit_resample(X, y)

# Printing number of samples in each class after Under-Sampling
print('After Under-Sampling: ')
print('Samples in class 0:', sum(y_nearmiss == 0))
print('Samples in class 1:', sum(y_nearmiss == 1))

```
     `

****Output:****

```
Before Under-Sampling: 
Samples in class 0: 80
Samples in class 1: 20
After Under-Sampling: 
Samples in class 0: 20
Samples in class 1: 20
```


This code performs under-sampling on an imbalanced dataset using the NearMiss algorithm from imbalanced-learn. In order to create a more balanced distribution, NearMiss chooses samples from the majority class that are near to the minority class. For comparison, the number of samples in each class is printed both before and after undersampling.

### 7\. Cost Sensitive Training (Penalize Algorithm)

Cost-sensitive training is a machine learning technique in which the algorithm is trained by taking into account the various costs connected to various kinds of errors. The cost of incorrectly classifying instances of one class may differ from the cost of incorrectly classifying instances of the other class in a typical [binary classification](https://www.geeksforgeeks.org/getting-started-with-classification/) problem. The goal of cost-sensitive training is to incorporate these expenses into the model-training procedure.

#### Implementation of Cost Sensitive Training

Python3`   ```
from sklearn.model_selection import train_test_split
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.datasets import make_classification

# Creating a synthetic imbalanced dataset
X, y = make_classification(n_classes=2, weights=[
                           0.9, 0.1], n_features=10, n_samples=1000, random_state=42)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Creating and training a BalancedRandomForestClassifier
clf = BalancedRandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Making predictions on the test set
y_pred = clf.predict(X_test)

# Calculate ROC-AUC score
roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

```
     `

****Output:****

```
ROC-AUC Score: 0.8714
Accuracy: 0.8650
```


This code shows how to handle imbalanced data using the BalancedRandomForestClassifier from imbalanced-learn. After dividing the dataset into training and testing sets, it synthesizes an imbalanced dataset and trains the classifier. Next, for model evaluation, the [accuracy](https://www.geeksforgeeks.org/techniques-to-evaluate-accuracy-of-classifier-in-data-mining/) and [ROC-AUC](https://www.geeksforgeeks.org/calculate-roc-auc-for-classification-algorithm-such-as-random-forest/) score are computed and printed.

### Advantages and Disadvantages of Over Sampling

#### Advantages

*   ****Enhanced Model Performance****: Enhances the model’s capacity to identify patterns in data from minority classes.
*   ****More Robust Models:**** It becomes more robust, especially when handling unbalanced datasets.
*   ****Reduced Risk of Information Loss:**** Oversampling helps to keep potentially important data from being lost.

#### Disadvantages

*   ****Increased Complexity:**** When the dataset grows, so do the computational requirements.
*   ****Potential Overfitting:**** The over-sampled data may introduce noise into the model fitting process.
*   ****Algorithm Sensitivity:**** Generalization may suffer from some algorithms’ sensitivity to repeated occurrences.

### Advantages and Disadvantages of Under Sampling

#### Advantages

*   ****Reduced Complexity:**** Under-sampling streamlines the dataset and speeds up calculations.
*   ****Prevents Overfitting:**** helps avoid overfitting, particularly in cases where the dominant class is the majority.
*   ****Simpler Models:**** results in less complex models that are simpler to understand.

#### Disadvantages

*   ****Loss of Information:**** Information loss may occur from removing instances of the majority class.
*   ****Risk of Bias:**** Undersampling may cause bias in how the original data are represented.
*   ****Potential for Instability:**** The model might become unstable as it grows more susceptible to changes.

### Frequently Asked Questions (FAQs)

****1\. Why is handling imbalanced classes important in machine learning?****

> Handling imbalanced classes is crucial because most machine learning algorithms are designed to maximize accuracy. In datasets with imbalanced classes, a model may achieve high accuracy by simply predicting the majority class, but fail to capture patterns in the minority class, which is often the primary focus.

****2\. What are common problems associated with imbalanced classes?****

> Common problems include biased model performance, where the model is more accurate for the majority class but performs poorly for the minority class. This is misleading and fails to address the purpose of the model. Additionally, standard evaluation metrics like accuracy can be misleading in imbalanced datasets.

****3\. How does class imbalance affect machine learning algorithms?****

> Because algorithms strive to maximize overall accuracy, class imbalance can result in biased models that favor the majority class. Poor performance on the minority class follows, and it’s possible that the model misses significant patterns in the data.

****4\. What are some techniques for handling imbalanced classes?****

> Resampling techniques that are frequently used include SMOTE (Synthetic Minority Over-sampling Technique) and random under- and over-sampling. Altering the algorithm’s parameters to penalize incorrectly classifying the minority class is an additional strategy.
