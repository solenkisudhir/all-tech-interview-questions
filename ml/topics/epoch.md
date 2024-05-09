# Epoch in Machine Learning
In Machine Learning, whenever you want to train a model with some data, then **Epoch** refers to one complete pass of the training dataset through the algorithm. Moreover, it takes a few epochs while training a machine learning model, but, in this scenario, you will face an issue while feeding a bunch of training data in the model. This issue happens due to limitations of computer storage. To overcome this issue, we have to break the training data into small batches according to the computer memory or storage capacity. Then only we can train a machine learning model by feeding these batches without any hassle. This process is called batch in machine learning, and **_further, when all batches are fed exactly once to train the model, then this entire procedure is known as Epoch in Machine Learning_**. In this article, **''Epoch in Machine Learning''** we will briefly discuss the Epoch, batch, and sample, etc. So let's start with the definition of the Epoch in Machine Learning.

![Epoch in Machine Learning](https://static.javatpoint.com/tutorial/machine-learning/images/epoch-in-machine-learning.png)

What is Epoch in Machine Learning?
----------------------------------

Epochs are defined as the total number of iterations for training the machine learning model with all the training data in one cycle. In Epoch, all training data is used exactly once. Further, in other words, Epoch can also be understood as the total number of passes an algorithm has completed around the training dataset. A forward and a backward pass together counted as one pass in training.

Usually, when a machine learning model is trained, then it requires a little number of Epochs. An Epoch is often mixed up with iteration.

What is Iteration?
------------------

Iteration is defined as a total number of batches required to complete one epoch, where a number of batches are equal to the total number of iteration for one epoch.

Let's understand the iteration and epoch with an example, where we have 3000 training examples that we are going to use to train a machine learning model.

In the above scenario, we can break up the training dataset into sizeable batches. So let's suppose we have considered the batches of 500 examples in each batch, then it will take 6 iterations to complete 1 Epoch.

Mathematically, we can understand it as follows;

*   Total number of training examples = 3000;
*   Assume each batch size = 500;
*   Then the total number of Iterations = Total number of training examples/Individual batch size = 3000/500
*   Total number of iterations = 6
*   And **_1 Epoch = 6 Iterations_**

Now, understand the Batch size in brief.

What is Batch in Machine Learning?
----------------------------------

Before starting the introduction of Batch in machine learning, you must have one thing in your mind that the batch size and the batch are two separate entities in machine learning.

Batch size is defined as the total number of training examples that exist in a single batch. You can understand batch with the above-mentioned example also, where we have divided the entire training dataset/examples into different batches or sets or parts.

Let's understand the concept of mixing up an Epoch and iteration with the below example where we have considered 1000 datasets as shown in the below image.

![Epoch in Machine Learning](https://static.javatpoint.com/tutorial/machine-learning/images/epoch-in-machine-learning2.png)

In the above figure, we can understand this concept as follows:

*   If the Batch size is 1000, then an epoch will complete in one iteration.
*   If the Batch size is 500, then an epoch will complete in 2 iterations.

Similarly, if the batch size is too small or such as 100, then the epoch will be complete in 10 iterations. So, as a result, we can conclude that for each epoch, the required number of iterations times the batch size gives the number of data points. However, we can use multiple numbers epochs for training the machine learning model.

Key points about Epoch and Batch in Machine Learning:
-----------------------------------------------------

There are a few important points that everyone should keep in mind during training a machine learning model. These are as follows:

*   Epoch is a machine learning terminology that refers to the number of passes the training data goes through machine learning algorithm during the entire data points.
*   If there is a large amount of data available, then you can divide entire data sets into common groups or batches.
*   The process of running one batch through the learning model is known as iteration. In Machine Learning, one cycle in entire training data sets is called an Epoch. However, in ideal conditions, one cycle in entire training data sets is called an Epoch but training a model typically requires multiple numbers of Epochs.
*   Better generalization can be achieved with new inputs by using more epochs in the training of the machine learning model.
*   Given the complexity and variety of data in real-world applications, hundreds to thousands of epochs may be required to achieve reasonable test data correctness. Furthermore, the term epoch has several definitions depending on the topic at hand.

Why use more than one Epoch?
----------------------------

It may not look correct that passing the entire dataset through an ML algorithm or neural network is not enough, and we need to pass it multiple times to the same algorithm.

So it needs to be kept in mind that to optimize the learning, we use gradient descent, an iterative process. Hence, it is not enough to update the weights with a single pass or one epoch.

Moreover, one epoch may lead to overfitting in the model.

* * *