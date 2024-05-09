# Hyperparameters in Machine Learning
**_Hyperparameters in Machine learning are those parameters that are explicitly defined by the user to control the learning process._** These hyperparameters are used to improve the learning of the model, and their values are set before starting the learning process of the model.

![Hyperparameters in Machine Learning](https://static.javatpoint.com/tutorial/machine-learning/images/hyperparameters-in-machine-learning.png)

In this topic, we are going to discuss one of the most important concepts of machine learning, i.e., Hyperparameters, their examples, hyperparameter tuning, categories of hyperparameters, how hyperparameter is different from parameter in Machine Learning? But before starting, let's first understand the Hyperparameter.

What are hyperparameters?
-------------------------

In Machine Learning/Deep Learning, a model is represented by its parameters. In contrast, a training process involves selecting the best/optimal hyperparameters that are used by learning algorithms to provide the best result. So, what are these hyperparameters? The answer is, "**_Hyperparameters are defined as the parameters that are explicitly defined by the user to control the learning process."_**

Here the prefix "hyper" suggests that the parameters are top-level parameters that are used in controlling the learning process. The value of the Hyperparameter is selected and set by the machine learning engineer before the learning algorithm begins training the model. **Hence, these are external to the model, and their values cannot be changed during the training process**.

### Some examples of Hyperparameters in Machine Learning

*   The k in kNN or K-Nearest Neighbour algorithm
*   Learning rate for training a neural network
*   Train-test split ratio
*   Batch Size
*   Number of Epochs
*   Branches in Decision Tree
*   Number of clusters in Clustering Algorithm

Difference between Parameter and Hyperparameter?
------------------------------------------------

There is always a big confusion between Parameters and hyperparameters or model hyperparameters. So, in order to clear this confusion, let's understand the difference between both of them and how they are related to each other.

### Model Parameters:

Model parameters are configuration variables that are internal to the model, and a model learns them on its own. For example**, W Weights or Coefficients of independent variables in the Linear regression model**. or **Weights or Coefficients of independent variables in SVM, weight, and biases of a neural network, cluster centroid in clustering.** Some key points for model parameters are as follows:

*   They are used by the model for making predictions.
*   They are learned by the model from the data itself
*   These are usually not set manually.
*   These are the part of the model and key to a machine learning Algorithm.

### Model Hyperparameters:

Hyperparameters are those parameters that are explicitly defined by the user to control the learning process. Some key points for model parameters are as follows:

*   These are usually defined manually by the machine learning engineer.
*   One cannot know the exact best value for hyperparameters for the given problem. The best value can be determined either by the rule of thumb or by trial and error.
*   Some examples of Hyperparameters are **the learning rate for training a neural network, K in the KNN algorithm,**

Categories of Hyperparameters
-----------------------------

Broadly hyperparameters can be divided into two categories, which are given below:

1.  **Hyperparameter for Optimization**
2.  **Hyperparameter for Specific Models**

### Hyperparameter for Optimization

The process of selecting the best hyperparameters to use is known as hyperparameter tuning, and the tuning process is also known as hyperparameter optimization. Optimization parameters are used for optimizing the model.

![Hyperparameters in Machine Learning](https://static.javatpoint.com/tutorial/machine-learning/images/hyperparameters-in-machine-learning2.png)

Some of the popular optimization parameters are given below:

*   **Learning Rate:** The learning rate is the hyperparameter in optimization algorithms that controls how much the model needs to change in response to the estimated error for each time when the model's weights are updated. It is one of the crucial parameters while building a neural network, and also it determines the frequency of cross-checking with model parameters. Selecting the optimized learning rate is a challenging task because if the learning rate is very less, then it may slow down the training process. On the other hand, if the learning rate is too large, then it may not optimize the model properly.

#### Note: Learning rate is a crucial hyperparameter for optimizing the model, so if there is a requirement of tuning only a single hyperparameter, it is suggested to tune the learning rate.

*   **Batch Size:** To enhance the speed of the learning process, the training set is divided into different subsets, which are known as a batch. **Number of Epochs:** An epoch can be defined as the complete cycle for training the machine learning model. Epoch represents an iterative learning process. The number of epochs varies from model to model, and various models are created with more than one epoch. To determine the right number of epochs, a validation error is taken into account. The number of epochs is increased until there is a reduction in a validation error. If there is no improvement in reduction error for the consecutive epochs, then it indicates to stop increasing the number of epochs.

### Hyperparameter for Specific Models

Hyperparameters that are involved in the structure of the model are known as hyperparameters for specific models. These are given below:

*   **A number of Hidden Units:** Hidden units are part of neural networks, which refer to the components comprising the layers of processors between input and output units in a neural network.

It is important to specify the number of hidden units hyperparameter for the neural network. It should be between the size of the input layer and the size of the output layer. More specifically, the number of hidden units should be 2/3 of the size of the input layer, plus the size of the output layer.

For complex functions, it is necessary to specify the number of hidden units, but it should not overfit the model.

*   **Number of Layers:** A neural network is made up of vertically arranged components, which are called layers. There are mainly **input layers, hidden layers, and output layers**. A 3-layered neural network gives a better performance than a 2-layered network. For a Convolutional Neural network, a greater number of layers make a better model.

Conclusion
----------

Hyperparameters are the parameters that are explicitly defined to control the learning process before applying a machine-learning algorithm to a dataset. These are used to specify the learning capacity and complexity of the model. Some of the hyperparameters are used for the optimization of the models, such as Batch size, learning rate, etc., and some are specific to the models, such as Number of Hidden layers, etc.

* * *