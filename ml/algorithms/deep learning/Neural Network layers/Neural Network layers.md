# List of Deep Learning Layers

****Deep learning (DL)**** is characterized by the use of ****neural networks**** with multiple layers to model and solve complex problems. Each layer in the neural network plays a unique role in the process of converting input data into meaningful and insightful outputs. The article explores the ****layers**** that are used to construct a neural network.

**Table of Content**

*   [Role of Deep Learning Layers](#role-of-deep-learning-layers)
*   [MATLAB Input Layer](#matlab-input-layer)
*   [MATLAB Fully Connected Layers](#matlab-fully-connected-layers)
*   [MATLAB Convolution Layers](#matlab-convolution-layers)
*   [MATLAB Recurrent Layers](#matlab-recurrent-layers)
*   [MATLAB Activation Layers](#matlab-activation-layers)
*   [MATLAB Pooling and Unpooling Layers](#matlab-pooling-and-unpooling-layers)
*   [MATLAB Normalization Layer and Dropout Layer](#matlab-normalization-layer-and-dropout-layer)
*   [MATLAB Output Layers](#matlab-output-layers)

Role of Deep Learning Layers
----------------------------

A layer in a [deep learning](https://www.geeksforgeeks.org/introduction-deep-learning/) model serves as a fundamental building block in the model’s architecture. The structure of the network is responsible for processing and transforming input data. The flow of information through these layers is sequential, with each layer taking input from the preceding layers and passing its transformed output to the subsequent layers. This cascading process continues through the network until the final layer produces the model’s ultimate output.

The input to a layer consists of features or representations derived from the data processed by earlier layers. Each layer performs a specific computation or [set](https://www.geeksforgeeks.org/set-in-cpp-stl/) of operations on this input, introducing non-linearity and abstraction to the information. The transformed output, often referred to as activations or feature maps, encapsulates higher-level representations that capture complex patterns and relationships within the data. The nature and function of each layer vary based on its type within the neural network architecture.

The nature and function of each layer vary based on its type within the neural network architecture. For instance:

1.  ****Dense (Fully Connected) Layer:**** Neurons in this layer are connected to every neuron in the previous layer, creating a dense network of connections. This layer is effective in capturing global patterns in the data.
2.  ****Convolutional Layer:**** Specialized for grid-like data, such as images, this layer employs [convolution operations](https://www.geeksforgeeks.org/introduction-convolution-neural-network/) to detect spatial patterns and features.
3.  ****Recurrent Layer:**** Suited for sequential data, recurrent layers utilize feedback loops to consider context from previous time steps, making them suitable for tasks like [natural language processing](https://www.geeksforgeeks.org/introduction-to-natural-language-processing/).
4.  ****Pooling Layer:**** Reduces spatial dimensions and focuses on retaining essential information, aiding in downsampling and [feature selection.](https://www.geeksforgeeks.org/feature-selection-techniques-in-machine-learning/)

MATLAB Input Layer
------------------



* Layer: inputLayer
  * Description of Layer: Input layer receives and process data in a specialized format, serving as the initial stage for information entry into a neural network.
* Layer: sequenceInputLayer
  * Description of Layer: Sequence input layer receives sequential data for a neural network and incorporates the normalization of the data during the input process.
* Layer: featureInputLayer
  * Description of Layer: Feature Input Layer processes feature data for a neural network and integrates data normalization. This layer is suitable when dealing with a dataset consisting of numerical scalar values that represent features, without spatial or temporal dimensions.
* Layer: imageInputLayer
  * Description of Layer: Image input layer processes 2 dimensional images in a neural network and uses data normalization using the input stage.
* Layer: image3dInputLayer
  * Description of Layer: 3-D image input layer receives 3-D image for a neural network.


MATLAB Fully Connected Layers
-----------------------------



* Layers: fullyConnectedLayer
  * Description of Layer: Fully connected layer performs matrix multiplication with a weight matrix and subsequently adding a bias vector.


MATLAB Convolution Layers
-------------------------



* Layer: convolution1dLayer
  * Description of Layer: One-dimensional convolutional layer employs sliding convolutional filters on 1-D input data.
* Layer: convolution2dLayer
  * Description of Layer: Two-dimensional convolutional layer employs sliding convolutional filters on 2-D input data.
* Layer: convolution3dLayer
  * Description of Layer: Three-dimensional convolutional layer employs sliding convolutional filters on 3-D input data.
* Layer: transposedConv2dLayer
  * Description of Layer: Transposed two-dimensional convolutional layer increases the resolution of two-dimensional feature maps through upsampling.
* Layer: transposedConv3dLayer
  * Description of Layer: Transposed three-dimensional convolutional layer increases the resolution of three-dimensional feature maps through upsampling.


MATLAB Recurrent Layers
-----------------------



* Layer: lstmLayer
  * Description of Layer: LSTM layer represents a type of recurrent neural network (RNN) layer specifically designed to capture and learn long-term dependencies among different time steps in time-series and sequential data.
* Layer: lstmProjectedLayer
  * Description of Layer: LSTM projected layer, within the realm of recurrent neural networks (RNNs), is adept at understanding and incorporating long-term dependencies among various time steps within time-series and sequential data. This is achieved through the utilization of learnable weights designed for projection.
* Layer: bilstmLayer
  * Description of Layer: Bidirectional LSTM (BiLSTM) layer, belonging to the family of recurrent neural networks (RNNs), is proficient in capturing long-term dependencies in both forward and backward directions among different time steps within time-series or sequential data. This bidirectional learning is valuable when the RNN needs to gather insights from the entire time series at each individual time step.
* Layer: gruLayer
  * Description of Layer: Gated Recurrent Unit (GRU) layer serves as a type of recurrent neural network (RNN) layer designed to capture dependencies among different time steps within time-series and sequential data.
* Layer: gruProjectedLayer
  * Description of Layer: A GRU projected layer, within the context of recurrent neural networks (RNNs), is specialized in understanding and incorporating dependencies among various time steps within time-series and sequential data. This is accomplished through the utilization of learnable weights designed for projection.


****MATLAB Activation Layers****
--------------------------------



* Layer: reluLayer
  * Description of Layer: ReLU conducts a threshold operation on each element of the input, setting any value that is less zero to zero.
* Layer: leakyReluLayer
  * Description of Layer: Leaky ReLU applies a threshold operation, where any input value that is less than zero is multiplied by a constant scalar.
* Layer: clippedReluLayer
  * Description of Layer: Clipped ReLU layer executes a threshold operation, setting any input value below zero to zero and capping any value surpassing the defined ceiling to that specific ceiling value.
* Layer: eluLayer
  * Description of Layer: Exponential Linear Unit (ELU) activation layer executes the identity operation for positive inputs and applies an exponential nonlinearity for negative inputs.
* Layer: geluLayer
  * Description of Layer: Gaussian Error Linear Unit (GELU) layer adjusts the input by considering its probability within a Gaussian distribution.
* Layer: tanhLayer
  * Description of Layer: Hyperbolic tangent (tanh) activation layer utilizes the tanh function to transform the inputs of the layer.
* Layer: swishLayer
  * Description of Layer: Swish activation layer employs the swish function to process the inputs of the layer.


MATLAB Pooling and Unpooling Layers
-----------------------------------



* Layer: averagePooling1dLayer
  * Description of Layer: One dimensional average pooling layer accomplishes downsampling by segmenting the input into 1-D pooling regions and subsequently calculating the average within each region.
* Layer: averagePooling2dLayer 
  * Description of Layer: Two dimensional average pooling layer conducts downsampling by partitioning the input into rectangular pooling regions and subsequently determining the average value within each region.
* Layer: averagePooling3dLayer
  * Description of Layer: Three dimensional average pooling layer achieves downsampling by partitioning the three-dimensional input into cuboidal pooling regions and then calculating the average values within each of these regions.
* Layer: globalAveragePooling1dLayer
  * Description of Layer: 1-D global average pooling layer achieves downsampling by generating the average output across the time or spatial dimensions of the input.
* Layer: globalAveragePooling2dLayer 
  * Description of Layer: 2-D global average pooling layer accomplishes downsampling by determining the mean value across the height and width dimensions of the input.
* Layer: globalAveragePooling3dLayer 
  * Description of Layer: 3-D global average pooling layer achieves downsampling by calculating the mean across the height, width, and depth dimensions of the input.
* Layer: maxPooling1dLayer
  * Description of Layer: 1-D global max pooling layer achieves downsampling by producing the maximum value across the time or spatial dimensions of the input.
* Layer: maxUnpooling2dLayer
  * Description of Layer: 2-D max unpooling layer reverses the pooling operation on the output of a 2-D max pooling layer.


MATLAB Normalization Layer and Dropout Layer
--------------------------------------------



* Layer:  batchNormalizationLayer
  * Description of Layer: Batch normalization layer normalizes a mini-batch of data independently across all observations for each channel. To enhance the training speed of a convolutional neural network and mitigate sensitivity to network initialization, incorporate batch normalization layers between convolutional layers and non-linearities, such as ReLU layers.
* Layer: groupNormalizationLayer
  * Description of Layer: Group normalization layer normalizes a mini-batch of data independently across distinct subsets of channels for each observation. To expedite the training of a convolutional neural network and minimize sensitivity to network initialization, integrate group normalization layers between convolutional layers and non-linearities, such as ReLU layers.
* Layer: layerNormalizationLayer
  * Description of Layer: Layer normalization layer normalizes a mini-batch of data independently across all channels for each observation. To accelerate the training of recurrent and multilayer perceptron neural networks and diminish sensitivity to network initialization, incorporate layer normalization layers after the learnable layers, such as LSTM and fully connected layers.
* Layer: dropoutLayer
  * Description of Layer: Dropout layer randomly zeros out input elements based on a specified probability.


MATLAB Output Layers
--------------------



* Layer: softmaxLayer
  * Description of Layer: Softmax layer employs the softmax function on the input.
* Layer: sigmoidLayer
  * Description of Layer: Sigmoid layer utilizes a sigmoid function on the input, ensuring that the output is constrained within the range (0,1).
* Layer: classificationLayer
  * Description of Layer: Classification layer calculates the cross-entropy loss for tasks involving classification and weighted classification, specifically for scenarios with mutually exclusive classes.
* Layer: regressionLayer
  * Description of Layer: Regression layer calculates the loss using the half-mean-squared-error for tasks related to regression.

