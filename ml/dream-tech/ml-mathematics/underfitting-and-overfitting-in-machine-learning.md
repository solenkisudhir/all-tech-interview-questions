# ML | Underfitting and Overfitting

When we talk about the Machine Learning model, we actually talk about how well it performs and its accuracy which is known as prediction errors. Let us consider that we are designing a machine learning model. A model is said to be a good machine learning model if it generalizes any new input data from the problem domain in a proper way. This helps us to make predictions about future data, that the data model has never seen. Now, suppose we want to check how well our machine learning model learns and generalizes to the new data. For that, we have overfitting and underfitting, which are majorly responsible for the poor performances of the [machine learning](https://www.geeksforgeeks.org/machine-learning/) algorithms.

Bias and Variance in Machine Learning
-------------------------------------

*   [****Bias****](https://www.geeksforgeeks.org/bias-vs-variance-in-machine-learning/): Bias refers to the error due to overly simplistic assumptions in the learning algorithm. These assumptions make the model easier to comprehend and learn but might not capture the underlying complexities of the data. It is the error due to the model’s inability to represent the true relationship between input and output accurately. When a model has poor performance both on the training and testing data means high bias because of the simple model, indicating underfitting.
*   [****Variance****](https://www.geeksforgeeks.org/bias-vs-variance-in-machine-learning/): Variance, on the other hand, is the error due to the model’s sensitivity to fluctuations in the training data. It’s the variability of the model’s predictions for different instances of training data. High variance occurs when a model learns the training data’s noise and random fluctuations rather than the underlying pattern. As a result, the model performs well on the training data but poorly on the testing data, indicating overfitting.

![Bias and Variance-Geeksforgeeks](https://media.geeksforgeeks.org/wp-content/uploads/20230829151403/Bias-and-Variance-in-Machine-Learning.webp)

Bias and Variance

****Underfitting in Machine Learning****
----------------------------------------

A [statistical model](https://www.geeksforgeeks.org/difference-between-statistical-model-and-machine-learning/) or a machine learning algorithm is said to have underfitting when a model is too simple to capture data complexities. It represents the inability of the model to learn the training data effectively result in poor performance both on the training and testing data. In simple terms, an underfit model’s are inaccurate, especially when applied to new, unseen examples. It mainly happens when we uses very simple model with overly simplified assumptions. To address underfitting problem of the model, we need to use more complex models, with enhanced feature representation, and less regularization.

****Note: The underfitting model has High bias and low variance.****

### ****Reasons for**** ****Underfitting****

1.  The model is too simple, So it may be not capable to represent the complexities in the data.
2.  The input features which is used to train the model is not the adequate representations of underlying factors influencing the target variable.
3.  The size of the training dataset used is not enough.
4.  Excessive regularization are used to prevent the overfitting, which constraint the model to capture the data well.
5.  Features are not scaled.

### ****Techniques to Reduce Underfitting****

1.  Increase model complexity.
2.  Increase the number of features, performing [feature engineering](https://www.geeksforgeeks.org/what-is-feature-engineering/).
3.  Remove noise from the data.
4.  Increase the number of [epochs](https://www.geeksforgeeks.org/epoch-in-machine-learning/) or increase the duration of training to get better results.

****Overfitting in Machine Learning****
---------------------------------------

A [statistical model](https://www.geeksforgeeks.org/difference-between-statistical-model-and-machine-learning/) is said to be overfitted when the model does not make accurate predictions on testing data. When a model gets trained with so much data, it starts learning from the noise and inaccurate data entries in our data set. And when testing with test data results in High variance. Then the model does not categorize the data correctly, because of too many details and noise. The causes of overfitting are the non-parametric and non-linear methods because these types of machine learning algorithms have more freedom in building the model based on the dataset and therefore they can really build unrealistic models. A solution to avoid overfitting is using a linear algorithm if we have linear data or using the parameters like the maximal depth if we are using decision trees. 

In a nutshell, [Overfitting](https://www.geeksforgeeks.org/underfitting-and-overfitting-in-machine-learning/) is a problem where the evaluation of machine learning algorithms on training data is different from unseen data.

### Reasons for Overfitting:

1.   High variance and low bias.
2.  The model is too complex.
3.  The size of the training data.

### ****Techniques to Reduce Overfitting****

1.  Improving the quality of training data reduces overfitting by focusing on meaningful patterns, mitigate the risk of fitting the noise or irrelevant features.
2.  Increase the training data can improve the model’s ability to generalize to unseen data and reduce the likelihood of overfitting.
3.  Reduce model complexity.
4.  [Early stopping](https://www.geeksforgeeks.org/regularization-by-early-stopping/) during the training phase (have an eye over the loss over the training period as soon as loss begins to increase stop training).
5.  [Ridge Regularization](https://www.geeksforgeeks.org/lasso-vs-ridge-vs-elastic-net-ml/) and [Lasso Regularization](https://www.geeksforgeeks.org/implementation-of-lasso-regression-from-scratch-using-python/).
6.  Use [dropout](https://www.geeksforgeeks.org/dropout-in-neural-networks/) for [neural networks](https://www.geeksforgeeks.org/neural-networks-a-beginners-guide/) to tackle overfitting.

![Underfitting and Overfitting in Machine Learning-Geeksforgeeks](https://media.geeksforgeeks.org/wp-content/cdn-uploads/20190523171258/overfitting_2.png)

Underfitting and Overfitting

****Good Fit in a Statistical Model****
---------------------------------------

Ideally, the case when the model makes the predictions with 0 error, is said to have a good fit on the data. This situation is achievable at a spot between overfitting and underfitting. In order to understand it, we will have to look at the performance of our model with the passage of time, while it is learning from the training dataset.

With the passage of time, our model will keep on learning, and thus the error for the model on the training and testing data will keep on decreasing. If it will learn for too long, the model will become more prone to overfitting due to the presence of noise and less useful details. Hence the performance of our model will decrease. In order to get a good fit, we will stop at a point just before where the error starts increasing. At this point, the model is said to have good skills in training datasets as well as our unseen testing dataset.

  

