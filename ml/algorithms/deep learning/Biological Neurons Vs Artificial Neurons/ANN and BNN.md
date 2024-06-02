# Difference between ANN and BNN

Do you ever think of what it’s like to build anything like a brain, how these things work, or what they do? Let us look at how nodes communicate with neurons and what are some differences between artificial and biological neural networks.

**1\.** [**Artificial Neural Network**](https://www.geeksforgeeks.org/artificial-neural-networks-and-its-applications/)**:** Artificial Neural Network (ANN) is a type of neural network that is based on a Feed-Forward strategy. It is called this because they pass information through the nodes continuously till it reaches the output node. This is also known as the simplest type of neural network. **Some advantages of ANN :**

*   Ability to learn irrespective of the type of data (Linear or Non-Linear).
*   ANN is highly volatile and serves best in financial time series forecasting.

**Some disadvantages of ANN :**

*   The simplest architecture makes it difficult to explain the behavior of the network.
*   This network is dependent on hardware.

**2\. Biological Neural Network:** Biological Neural Network (BNN) is a structure that consists of Synapse, dendrites, cell body, and axon. In this neural network, the processing is carried out by neurons. Dendrites receive signals from other neurons, Soma sums all the incoming signals and axon transmits the signals to other cells. 

**Some advantages of BNN :**  

*   The synapses are the input processing element.
*   It is able to process highly complex parallel inputs.

**Some disadvantages of BNN :**

*   There is no controlling mechanism.
*   Speed of processing is slow being it is complex.

**Differences between ANN and BNN :**

Biological Neural Networks (BNNs) and Artificial Neural Networks (ANNs) are both composed of similar basic components, but there are some differences between them.

**Neurons:** In both BNNs and ANNs, neurons are the basic building blocks that process and transmit information. However, BNN neurons are more complex and diverse than ANNs. In BNNs, neurons have multiple **dendrites** that receive input from multiple sources, and the **axons** transmit signals to other neurons, while in ANNs, neurons are simplified and usually only have a single output.

**Synapses:** In both BNNs and ANNs, synapses are the points of connection between neurons, where information is transmitted. However, in ANNs, the connections between neurons are usually fixed, and the strength of the connections is determined by a set of weights, while in BNNs, the connections between neurons are more flexible, and the strength of the connections can be modified by a variety of factors, including learning and experience.

**Neural Pathways:** In both BNNs and ANNs, neural pathways are the connections between neurons that allow information to be transmitted throughout the network. However, in BNNs, neural pathways are highly complex and diverse, and the connections between neurons can be modified by experience and learning. In ANNs, neural pathways are usually simpler and predetermined by the architecture of the network.



* Parameters: Structure 
  * ANN: inputweightoutputhidden layer
  * BNN: dendritessynapseaxoncell body
* Parameters: Learning
  * ANN: very precise structures and formatted data
  * BNN: they can tolerate ambiguity
* Parameters: Processor
  * ANN: complexhigh speedone or a few
  * BNN: simplelow speedlarge number
* Parameters: Memory 
  * ANN: separate from a processorlocalizednon-content addressable
  * BNN: integrated into processor distributedcontent-addressable
* Parameters: Computing
  * ANN: centralizedsequentialstored programs
  * BNN: distributedparallelself-learning
* Parameters: Reliability
  * ANN: very vulnerable
  * BNN: robust
* Parameters: Expertise
  * ANN: numerical and symbolicmanipulations
  * BNN: perceptual problems
* Parameters: Operating Environment
  * ANN: well-definedwell-constrained
  * BNN: poorly definedun-constrained
* Parameters: Fault Tolerance
  * ANN: the potential of fault tolerance
  * BNN: performance degraded even on partial damage


Overall, while BNNs and ANNs share many basic components, there are significant differences in their complexity, flexibility, and adaptability. BNNs are highly complex and adaptable systems that can process information in parallel, and their plasticity allows them to learn and adapt over time. In contrast, ANNs are simpler systems that are designed to perform specific tasks, and their connections are usually fixed, with the network architecture determined by the designer.

  
