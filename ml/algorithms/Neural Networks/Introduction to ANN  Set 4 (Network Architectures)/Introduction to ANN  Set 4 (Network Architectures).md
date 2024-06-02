# Introduction to ANN | Set 4 (Network Architectures)

**Prerequisites:** Introduction to ANN | [Set-1](https://www.geeksforgeeks.org/introduction-to-artificial-neutral-networks/), [Set-2](https://www.geeksforgeeks.org/introduction-artificial-neural-network-set-2/), [Set-3](https://www.geeksforgeeks.org/introduction-ann-artificial-neural-networks-set-3-hybrid-systems/) 

An [Artificial Neural Network (ANN)](https://www.geeksforgeeks.org/implementing-ann-training-process-in-python/) is an information processing paradigm that is inspired by the brain. ANNs, like people, learn by examples. An ANN is configured for a specific application, such as pattern recognition or data classification, through a learning process. Learning largely involves adjustments to the synaptic connections that exist between the neurons. 

Artificial Neural Networks (ANNs) are a type of machine learning model that are inspired by the structure and function of the human brain. They consist of layers of interconnected “neurons” that process and transmit information.

There are several different architectures for ANNs, each with their own strengths and weaknesses. Some of the most common architectures include:

Feedforward Neural Networks: This is the simplest type of ANN architecture, where the information flows in one direction from input to output. The layers are fully connected, meaning each neuron in a layer is connected to all the neurons in the next layer.

Recurrent Neural Networks (RNNs): These networks have a “memory” component, where information can flow in cycles through the network. This allows the network to process sequences of data, such as time series or speech.

Convolutional Neural Networks (CNNs): These networks are designed to process data with a grid-like topology, such as images. The layers consist of convolutional layers, which learn to detect specific features in the data, and pooling layers, which reduce the spatial dimensions of the data.

Autoencoders: These are neural networks that are used for unsupervised learning. They consist of an encoder that maps the input data to a lower-dimensional representation and a decoder that maps the representation back to the original data.

Generative Adversarial Networks (GANs): These are neural networks that are used for generative modeling. They consist of two parts: a generator that learns to generate new data samples, and a discriminator that learns to distinguish between real and generated data.

The model of an artificial neural network can be specified by three entities:   
 

*   **Interconnections**
*   [**Activation functions**](https://www.geeksforgeeks.org/activation-functions-neural-networks/)
*   **Learning rules**

### Interconnections:

Interconnection can be defined as the way processing elements (Neuron) in ANN are connected to each other. Hence, the arrangements of these processing elements and geometry of interconnections are very essential in ANN.   
These arrangements always have two layers that are common to all network architectures, the Input layer and output layer where the input layer buffers the input signal, and the output layer generates the output of the network. The third layer is the Hidden layer, in which neurons are neither kept in the input layer nor in the output layer. These neurons are hidden from the people who are interfacing with the system and act as a black box to them. By increasing the hidden layers with neurons, the system’s computational and processing power can be increased but the training phenomena of the system get more complex at the same time. 

There exist five basic types of neuron connection architecture : 

1.  Single-layer feed-forward network
2.  Multilayer feed-forward network
3.  Single node with its own feedback
4.  Single-layer recurrent network
5.  Multilayer recurrent network

**1.** **Single-layer feed-forward network** 

![](https://media.geeksforgeeks.org/wp-content/uploads/Untitled-Diagram-3-3.png)

In this type of network, we have only two layers input layer and the output layer but the input layer does not count because no computation is performed in this layer. The output layer is formed when different weights are applied to input nodes and the cumulative effect per node is taken. After this, the neurons collectively give the output layer to compute the output signals.

**2.** **Multilayer feed-forward network** 

![](https://media.geeksforgeeks.org/wp-content/uploads/w2.png)

This layer also has a hidden layer that is internal to the network and has no direct contact with the external layer. The existence of one or more hidden layers enables the network to be computationally stronger, a feed-forward network because of information flow through the input function, and the intermediate computations used to determine the output Z. There are no feedback connections in which outputs of the model are fed back into itself.

**3.** **Single node with its own feedback**   
 

![](https://media.geeksforgeeks.org/wp-content/uploads/w3.png)

Single Node with own Feedback  
 

When outputs can be directed back as inputs to the same layer or preceding layer nodes, then it results in feedback networks. Recurrent networks are feedback networks with closed loops. The above figure shows a single recurrent network having a single neuron with feedback to itself.

**4.** **Single-layer recurrent network** 

![](https://media.geeksforgeeks.org/wp-content/uploads/w4.png)

The above network is a single-layer network with a feedback connection in which the processing element’s output can be directed back to itself or to another processing element or both. A recurrent neural network is a class of artificial neural networks where connections between nodes form a directed graph along a sequence. This allows it to exhibit dynamic temporal behavior for a time sequence. Unlike feedforward neural networks, RNNs can use their internal state (memory) to process sequences of inputs.

**5.** **Multilayer recurrent network**   
 

![](https://media.geeksforgeeks.org/wp-content/uploads/w5.png)

In this type of network, processing element output can be directed to the processing element in the same layer and in the preceding layer forming a multilayer recurrent network. They perform the same task for every element of a sequence, with the output being dependent on the previous computations. Inputs are not needed at each time step. The main feature of a Recurrent Neural Network is its hidden state, which captures some information about a sequence.
