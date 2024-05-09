# Perceptron in Machine Learning
In Machine Learning and Artificial Intelligence, Perceptron is the most commonly used term for all folks. It is the primary step to learn Machine Learning and Deep Learning technologies, which consists of a set of weights, input values or scores, and a threshold. **_Perceptron is a building block of an Artificial Neural Network_**. Initially, in the mid of 19th century, **Mr. Frank Rosenblatt** invented the Perceptron for performing certain calculations to detect input data capabilities or business intelligence. Perceptron is a linear Machine Learning algorithm used for supervised learning for various binary classifiers. This algorithm enables neurons to learn elements and processes them one by one during preparation. In this tutorial, "Perceptron in Machine Learning," we will discuss in-depth knowledge of Perceptron and its basic functions in brief. Let's start with the basic introduction of Perceptron.

![Perceptron in Machine Learning](https://static.javatpoint.com/tutorial/machine-learning/images/perceptron-in-machine-learning.png)

What is the Perceptron model in Machine Learning?
-------------------------------------------------

Perceptron is Machine Learning algorithm for supervised learning of various binary classification tasks. Further, **_Perceptron is also understood as an Artificial Neuron or neural network unit that helps to detect certain input data computations in business intelligence_**.

Perceptron model is also treated as one of the best and simplest types of Artificial Neural networks. However, it is a supervised learning algorithm of binary classifiers. Hence, we can consider it as a single-layer neural network with four main parameters, i.e., **input values, weights and Bias, net sum, and an activation function.**

What is Binary classifier in Machine Learning?
----------------------------------------------

In Machine Learning, binary classifiers are defined as the function that helps in deciding whether input data can be represented as vectors of numbers and belongs to some specific class.

Binary classifiers can be considered as linear classifiers. In simple words, we can understand it as a **_classification algorithm that can predict linear predictor function in terms of weight and feature vectors._**

Basic Components of Perceptron
------------------------------

Mr. Frank Rosenblatt invented the perceptron model as a binary classifier which contains three main components. These are as follows:

![Perceptron in Machine Learning](https://static.javatpoint.com/tutorial/machine-learning/images/perceptron-in-machine-learning2.png)

*   **Input Nodes or Input Layer:**

This is the primary component of Perceptron which accepts the initial data into the system for further processing. Each input node contains a real numerical value.

*   **Wight and Bias:**

Weight parameter represents the strength of the connection between units. This is another most important parameter of Perceptron components. Weight is directly proportional to the strength of the associated input neuron in deciding the output. Further, Bias can be considered as the line of intercept in a linear equation.

*   **Activation Function:**

These are the final and important components that help to determine whether the neuron will fire or not. Activation Function can be considered primarily as a step function.

Types of Activation functions:

*   Sign function
*   Step function, and
*   Sigmoid function

![Perceptron in Machine Learning](https://static.javatpoint.com/tutorial/machine-learning/images/perceptron-in-machine-learning3.png)

The data scientist uses the activation function to take a subjective decision based on various problem statements and forms the desired outputs. Activation function may differ (e.g., Sign, Step, and Sigmoid) in perceptron models by checking whether the learning process is slow or has vanishing or exploding gradients.

How does Perceptron work?
-------------------------

In Machine Learning, Perceptron is considered as a single-layer neural network that consists of four main parameters named input values (Input nodes), weights and Bias, net sum, and an activation function. The perceptron model begins with the multiplication of all input values and their weights, then adds these values together to create the weighted sum. Then this weighted sum is applied to the activation function 'f' to obtain the desired output. This activation function is also known as the **step function** and is represented by **'f'**.

![Perceptron in Machine Learning](https://static.javatpoint.com/tutorial/machine-learning/images/perceptron-in-machine-learning4.png)

This step function or Activation function plays a vital role in ensuring that output is mapped between required values (0,1) or (-1,1). It is important to note that the weight of input is indicative of the strength of a node. Similarly, an input's bias value gives the ability to shift the activation function curve up or down.

Perceptron model works in two important steps as follows:

**Step-1**

In the first step first, multiply all input values with corresponding weight values and then add them to determine the weighted sum. Mathematically, we can calculate the weighted sum as follows:

∑wi\*xi = x1\*w1 + x2\*w2 +…wn\*xn

Add a special term called **bias 'b'** to this weighted sum to improve the model's performance.

**∑wi\*xi + b**

**Step-2**

In the second step, an activation function is applied with the above-mentioned weighted sum, which gives us output either in binary form or a continuous value as follows:

**Y = f(∑wi\*xi + b)**

Types of Perceptron Models
--------------------------

Based on the layers, Perceptron models are divided into two types. These are as follows:

1.  Single-layer Perceptron Model
2.  Multi-layer Perceptron model

### Single Layer Perceptron Model:

This is one of the easiest Artificial neural networks (ANN) types. A single-layered perceptron model consists feed-forward network and also includes a threshold transfer function inside the model. The main objective of the single-layer perceptron model is to analyze the linearly separable objects with binary outcomes.

In a single layer perceptron model, its algorithms do not contain recorded data, so it begins with inconstantly allocated input for weight parameters. Further, it sums up all inputs (weight). After adding all inputs, if the total sum of all inputs is more than a pre-determined value, the model gets activated and shows the output value as +1.

If the outcome is same as pre-determined or threshold value, then the performance of this model is stated as satisfied, and weight demand does not change. However, this model consists of a few discrepancies triggered when multiple weight inputs values are fed into the model. Hence, to find desired output and minimize errors, some changes should be necessary for the weights input.

_"Single-layer perceptron can learn only linearly separable patterns."_

### Multi-Layered Perceptron Model:

Like a single-layer perceptron model, a multi-layer perceptron model also has the same model structure but has a greater number of hidden layers.

The multi-layer perceptron model is also known as the Backpropagation algorithm, which executes in two stages as follows:

*   **Forward Stage:** Activation functions start from the input layer in the forward stage and terminate on the output layer.
*   **Backward Stage:** In the backward stage, weight and bias values are modified as per the model's requirement. In this stage, the error between actual output and demanded originated backward on the output layer and ended on the input layer.

Hence, a multi-layered perceptron model has considered as multiple artificial neural networks having various layers in which activation function does not remain linear, similar to a single layer perceptron model. Instead of linear, activation function can be executed as sigmoid, TanH, ReLU, etc., for deployment.

A multi-layer perceptron model has greater processing power and can process linear and non-linear patterns. Further, it can also implement logic gates such as AND, OR, XOR, NAND, NOT, XNOR, NOR.

**Advantages of Multi-Layer Perceptron:**

*   A multi-layered perceptron model can be used to solve complex non-linear problems.
*   It works well with both small and large input data.
*   It helps us to obtain quick predictions after the training.
*   It helps to obtain the same accuracy ratio with large as well as small data.

**Disadvantages of Multi-Layer Perceptron:**

*   In Multi-layer perceptron, computations are difficult and time-consuming.
*   In multi-layer Perceptron, it is difficult to predict how much the dependent variable affects each independent variable.
*   The model functioning depends on the quality of the training.

Perceptron Function
-------------------

Perceptron function ''f(x)'' can be achieved as output by multiplying the input 'x' with the learned weight coefficient 'w'.

Mathematically, we can express it as follows:

**f(x)=1; if w.x+b>0**

**otherwise, f(x)=0**

*   'w' represents real-valued weights vector
*   'b' represents the bias
*   'x' represents a vector of input x values.

Characteristics of Perceptron
-----------------------------

The perceptron model has the following characteristics.

1.  Perceptron is a machine learning algorithm for supervised learning of binary classifiers.
2.  In Perceptron, the weight coefficient is automatically learned.
3.  Initially, weights are multiplied with input features, and the decision is made whether the neuron is fired or not.
4.  The activation function applies a step rule to check whether the weight function is greater than zero.
5.  The linear decision boundary is drawn, enabling the distinction between the two linearly separable classes +1 and -1.
6.  If the added sum of all input values is more than the threshold value, it must have an output signal; otherwise, no output will be shown.

Limitations of Perceptron Model
-------------------------------

**A perceptron model has limitations as follows:**

*   The output of a perceptron can only be a binary number (0 or 1) due to the hard limit transfer function.
*   Perceptron can only be used to classify the linearly separable sets of input vectors. If input vectors are non-linear, it is not easy to classify them properly.

Future of Perceptron
--------------------

The future of the Perceptron model is much bright and significant as it helps to interpret data by building intuitive patterns and applying them in the future. Machine learning is a rapidly growing technology of Artificial Intelligence that is continuously evolving and in the developing phase; hence the future of perceptron technology will continue to support and facilitate analytical behavior in machines that will, in turn, add to the efficiency of computers.

The perceptron model is continuously becoming more advanced and working efficiently on complex problems with the help of artificial neurons.

Conclusion:
-----------

In this article, you have learned how Perceptron models are the simplest type of artificial neural network which carries input and their weights, the sum of all weighted input, and an activation function. Perceptron models are continuously contributing to Artificial Intelligence and Machine Learning, and these models are becoming more advanced. Perceptron enables the computer to work more efficiently on complex problems using various Machine Learning technologies. The Perceptrons are the fundamentals of artificial neural networks, and everyone should have in-depth knowledge of perceptron models to study deep neural networks.

* * *