# Types Of Activation Function in ANN 

The biological neural network has been modeled in the form of Artificial Neural Networks with artificial neurons simulating the function of a biological neuron. The artificial neuron is depicted in the below picture:

![](https://media.geeksforgeeks.org/wp-content/uploads/20210103184807/ann-660x332.jpg)

**Structure of an Artificial Neuron**

Each neuron consists of three major components: 

1.  A set of **‘i’ synapses having weight wi.** A signal xi forms the input to the i-th synapse having weight wi. The value of any weight may be positive or negative. A positive weight has an extraordinary effect, while a negative weight has an inhibitory effect on the output of the summation junction.
2.  A **summation junction** for the input signals is weighted by the respective synaptic weight. Because it is a linear combiner or adder of the weighted input signals, the output of the summation junction can be expressed as follows: ![y_{sum}=\sum_{i=1}^{n}w_ix_i](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-ac6bbf1ad4c4533635e432f607115054_l3.png "Rendered by QuickLaTeX.com")
3.  A threshold **activation function** (or simply **the activation function,** also known as **squashing function**) results in an output signal only when an input signal exceeding a specific threshold value comes as an input. It is similar in behaviour to the biological neuron which transmits the signal only when the total input signal meets the firing threshold.

**Types of Activation Function :**

There are different types of activation functions. The most commonly used activation function are listed below:

**A. Identity Function:** Identity function is used as an activation function for the input layer. It is a linear function having the form

![y_{out}=f(x)=x, \forall x](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-0b5fc8360c215531031dfc2233279723_l3.png "Rendered by QuickLaTeX.com")

As obvious, the output remains the same as the input.

**B. Threshold/step Function:** It is a commonly used activation function. As depicted in the diagram, it gives **1 as output** of the input is either 0 or positive. If the input is negative, it gives **0 as output**. Expressing it mathematically, 

![y_{out}=f(y_{sum})=\bigg\{\begin{matrix} 1, x \geq 0 \\ 0, x < 0 \end{matrix}](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-197f63e418f08e5cc4c8201010229809_l3.png "Rendered by QuickLaTeX.com")

![](https://media.geeksforgeeks.org/wp-content/uploads/20210103201237/step-300x242.jpg)

The threshold function is almost like the step function, with the only difference being a fact that ![\theta   ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-8959979047d94b75a5d60220e676ee54_l3.png "Rendered by QuickLaTeX.com") is used as a threshold value instead of . Expressing mathematically,

![y_{out}=f(y_{sum})=\bigg\{\begin{matrix} 1, x \geq \theta \\ 0, x < \theta \end{matrix}](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-d27d3b73f4f258cf7df151a482137645_l3.png "Rendered by QuickLaTeX.com")

![](https://media.geeksforgeeks.org/wp-content/uploads/20210103201821/threshold-300x232.jpg)

**C. ReLU (Rectified Linear Unit) Function:** It is the most popularly used activation function in the areas of convolutional neural networks and deep learning. It is of the form:

![f(x)=\bigg\{\begin{matrix} x, x \geq 0\\ 0, x < 0 \end{matrix}](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-ffeb5652372e8493381cbbc95241f1ac_l3.png "Rendered by QuickLaTeX.com")

![](https://media.geeksforgeeks.org/wp-content/uploads/20210103202342/ReLU-300x225.jpg)

This means that f(x) is zero when x is less than zero and f(x) is equal to x when x is above or equal to zero. **This function is differentiable**, except at a single point x = 0. In that sense, the derivative of a ReLU is actually a sub-derivative.

**D. Sigmoid Function:** It is by far the most commonly used activation function in neural networks. The need for sigmoid function stems from the fact that many learning algorithms require the activation function to be differentiable and hence continuous. There are two types of sigmoid function: 

**1\. Binary Sigmoid Function**

![](https://media.geeksforgeeks.org/wp-content/uploads/20210103204033/sf1-300x232.jpg)

A binary sigmoid function is of the form: ![y_{out}=f(x)=\frac{1}{1+e^{-kx}}](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-b54d4ac899d76782cc4a89a0474efe39_l3.png "Rendered by QuickLaTeX.com")

, where **k = steepness or slope parameter,** By varying the value of k, sigmoid function with different slopes can be obtained. It has a range of (0,1).  The slope of origin is **k/4.** As the value of k becomes very large, the sigmoid function becomes a threshold function. 

**2\. Bipolar Sigmoid Function**

![](https://media.geeksforgeeks.org/wp-content/uploads/20210103204519/sf2-300x230.jpg)

A bipolar sigmoid function is of the form ![y_{out}=f(x)=\frac{1-e^{-kx}}{1+e^{-kx}}](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-2490034e4f483490808d272ce4169c32_l3.png "Rendered by QuickLaTeX.com")

The range of values of sigmoid functions can be varied depending on the application. However, the range of (-1,+1) is most commonly adopted.

**E. Hyperbolic Tangent Function:** It is bipolar in nature. It is a widely adopted activation function for a special type of neural network known as **Backpropagation Network.** The hyperbolic tangent function is of the form

![y_{out}=f(x)\frac{e^x-e^-x}{e^x+e^-x}   ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-ddc1389478e8135aa4331a2ad8d1ba9d_l3.png "Rendered by QuickLaTeX.com") 

This function is similar to the bipolar sigmoid function. 
