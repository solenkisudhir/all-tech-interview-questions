# Long Short Term Memory Networks Explanation

**Prerequisites: Recurrent Neural Networks**

To solve the problem of Vanishing and Exploding Gradients in a Deep Recurrent Neural Network, many variations were developed. One of the most famous of them is the **Long Short Term Memory Network**(LSTM). In concept, an LSTM recurrent unit tries to “remember” all the past knowledge that the network is seen so far and to “forget” irrelevant data. This is done by introducing different activation function layers called “gates” for different purposes. Each LSTM recurrent unit also maintains a vector called the **Internal Cell State** which conceptually describes the information that was chosen to be retained by the previous LSTM recurrent unit.

LSTM networks are the most commonly used variation of Recurrent Neural Networks (RNNs). The critical component of the LSTM is the memory cell and the gates (including the forget gate but also the input gate), inner contents of the memory cell are modulated by the input gates and forget gates. Assuming that both of the segue he are closed, the contents of the memory cell will remain unmodified between one time-step and the next gradients gating structure allows information to be retained across many time-steps, and consequently also allows group that to flow across many time-steps. This allows the LSTM model to overcome the vanishing gradient properly occurs with most Recurrent Neural Network models.

 A Long Short Term Memory Network consists of four different gates for different purposes as described below:- 

1.  **Forget Gate(f):** At forget gate the input is combined with the previous output to generate a fraction between 0 and 1, that determines how much of the previous state need to be preserved (or in other words, how much of the state should be forgotten). This output is then multiplied with the previous state. Note: An activation output of 1.0 means “remember everything” and activation output of 0.0 means “forget everything.” From a different perspective, a better name for the forget gate might be the “remember gate”
2.  **Input Gate(i):** Input gate operates on the same signals as the forget gate, but here the objective is to decide which new information is going to enter the state of LSTM. The output of the input gate (again a fraction between 0 and 1) is multiplied with the output of tan h block that produces the new values that must be added to previous state. This gated vector is then added to previous state to generate current state
3.  **Input Modulation Gate(g):** It is often considered as a sub-part of the input gate and much literature on LSTM’s does not even mention it and assume it is inside the Input gate. It is used to modulate the information that the Input gate will write onto the Internal State Cell by adding non-linearity to the information and making the information **Zero-mean**. This is done to reduce the learning time as Zero-mean input has faster convergence. Although this gate’s actions are less important than the others and are often treated as a finesse-providing concept, it is good practice to include this gate in the structure of the LSTM unit.
4.  **Output Gate(o):** At output gate, the input and previous state are gated as before to generate another scaling fraction that is combined with the output of tanh block that brings the current state. This output is then given out. The output and state are fed back into the LSTM block.

The basic workflow of a Long Short Term Memory Network is similar to the workflow of a Recurrent Neural Network with the only difference being that the Internal Cell State is also passed forward along with the Hidden State. 

![](https://media.geeksforgeeks.org/wp-content/uploads/20190702161054/unrolled2.png)

**Working of an LSTM recurrent unit:**  

1.  Take input the current input, the previous hidden state, and the previous internal cell state.
2.  Calculate the values of the four different gates by following the below steps:-
    *   For each gate, calculate the parameterized vectors for the current input and the previous hidden state by element-wise multiplication with the concerned vector with the respective weights for each gate.
    *   Apply the respective activation function for each gate element-wise on the parameterized vectors. Below given is the list of the gates with the activation function to be applied for the gate.
3.  Calculate the current internal cell state by first calculating the element-wise multiplication vector of the input gate and the input modulation gate, then calculate the element-wise multiplication vector of the forget gate and the previous internal cell state and then add the two vectors.   
    ![c_{t} = i\odot g + f\odot c_{t-1}](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-98047cd905c37471c660bc681ba0cb4a_l3.png "Rendered by QuickLaTeX.com")
4.  Calculate the current hidden state by first taking the element-wise hyperbolic tangent of the current internal cell state vector and then performing element-wise multiplication with the output gate.

The above-stated working is illustrated as below:-  

![](https://media.geeksforgeeks.org/wp-content/uploads/20190702161123/working3.png)

Note that the blue circles denote element-wise multiplication. The weight matrix W contains different weights for the current input vector and the previous hidden state for each gate. 

Just like Recurrent Neural Networks, an LSTM network also generates an output at each time step and this output is used to train the network using gradient descent. 

![](https://media.geeksforgeeks.org/wp-content/uploads/20190702161217/yt2.png)

The only main difference between the Back-Propagation algorithms of Recurrent Neural Networks and Long Short Term Memory Networks is related to the mathematics of the algorithm. 

Let ![\overline{y}_{t}     ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-3c4718722dd3e81b15a1f1f92afccbd7_l3.png "Rendered by QuickLaTeX.com")be the predicted output at each time step and ![y_{t}     ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-3592a53d1c72803f61c760eebd434ed7_l3.png "Rendered by QuickLaTeX.com")be the actual output at each time step. Then the error at each time step is given by:- 

![E_{t} = -y_{t}log(\overline{y}_{t})     ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-e6cf536bbecb7e2efff745a460664b3a_l3.png "Rendered by QuickLaTeX.com")

The total error is thus given by the summation of errors at all time steps. 

![E = \sum _{t} E_{t}     ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-b2d46541914d12d35293a0d2278da7f2_l3.png "Rendered by QuickLaTeX.com")  
![\Rightarrow E = \sum _{t} -y_{t}log(\overline{y}_{t})     ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-9e0adfb29a28ffbb8e8e8e444ecda114_l3.png "Rendered by QuickLaTeX.com")

Similarly, the value ![\frac{\partial E}{\partial W}     ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-8fdfb69f0343924e0b58b69cda2bcf8e_l3.png "Rendered by QuickLaTeX.com")can be calculated as the summation of the gradients at each time step. 

![\frac{\partial E}{\partial W} = \sum _{t} \frac{\partial E_{t}}{\partial W}     ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-c0e64b7f460b94519310fcb103f39a57_l3.png "Rendered by QuickLaTeX.com")

Using the chain rule and using the fact that ![\overline{y}_{t}     ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-3c4718722dd3e81b15a1f1f92afccbd7_l3.png "Rendered by QuickLaTeX.com")is a function of ![h_{t}     ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-f562bc7b877c039d26ff10a5aa1f2831_l3.png "Rendered by QuickLaTeX.com")and which indeed is a function of ![c_{t}     ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-83eb0483bfc219383ca1be7db3a813a3_l3.png "Rendered by QuickLaTeX.com"), the following expression arises:- 

![\frac{\partial E_{t}}{\partial W} = \frac{\partial E_{t}}{\partial \overline{y}_{t}}\frac{\partial \overline{y}_{t}}{\partial h_{t}}\frac{\partial h_{t}}{\partial c_{t}}\frac{\partial c_{t}}{\partial c_{t-1}}\frac{\partial c_{t-1}}{\partial c_{t-2}}.......\frac{\partial c_{0}}{\partial W}     ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-f409e50fc00a9d709686badc9a6072df_l3.png "Rendered by QuickLaTeX.com")

Thus the total error gradient is given by the following:- 

![\frac{\partial E}{\partial W} = \sum _{t} \frac{\partial E_{t}}{\partial \overline{y}_{t}}\frac{\partial \overline{y}_{t}}{\partial h_{t}}\frac{\partial h_{t}}{\partial c_{t}}\frac{\partial c_{t}}{\partial c_{t-1}}\frac{\partial c_{t-1}}{\partial c_{t-2}}.......\frac{\partial c_{0}}{\partial W}     ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-2b1f4d2ca03cff6d9a81aaed3ff9e6c4_l3.png "Rendered by QuickLaTeX.com")

Note that the gradient equation involves a chain of ![\partial c_{t}     ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-818bd9eae8f61cb001aab3eaff61c09a_l3.png "Rendered by QuickLaTeX.com")for an LSTM Back-Propagation while the gradient equation involves a chain of ![\partial h_{t}     ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-58a05930299afed7a06c806d159510bf_l3.png "Rendered by QuickLaTeX.com")for a basic Recurrent Neural Network. 

**How does LSTM solve the problem of vanishing and exploding gradients?**

Recall the expression for ![c_{t}     ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-83eb0483bfc219383ca1be7db3a813a3_l3.png "Rendered by QuickLaTeX.com"). 

![c_{t} = i\odot g + f\odot c_{t-1}     ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-0bae30fadc8dd976087fe812763a738c_l3.png "Rendered by QuickLaTeX.com")

The value of the gradients is controlled by the chain of derivatives starting from ![\frac{\partial c_{t}}{\partial c_{t-1}}     ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-e3d5cee87f7124d76f2633ba67ca32a9_l3.png "Rendered by QuickLaTeX.com"). Expanding this value using the expression for ![c_{t}     ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-83eb0483bfc219383ca1be7db3a813a3_l3.png "Rendered by QuickLaTeX.com"):- 

![\frac{\partial c_{t}}{\partial c_{t-1}} = \frac{\partial c_{t}}{\partial f}\frac{\partial f}{\partial h_{t-1}}\frac{\partial h_{t-1}}{\partial c_{t-1}} + \frac{\partial c_{t}}{\partial i}\frac{\partial i}{\partial h_{t-1}}\frac{\partial h_{t-1}}{\partial c_{t-1}} + \frac{\partial c_{t}}{\partial g}\frac{\partial g}{\partial h_{t-1}}\frac{\partial h_{t-1}}{\partial c_{t-1}} + \frac{\partial c_{t}}{\partial c_{t-1}}     ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-886c109f36364df05f8794dffb234f2d_l3.png "Rendered by QuickLaTeX.com")

For a basic RNN, the term ![\frac{\partial h_{t}}{\partial h_{t-1}}     ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-95f515311206ea9a890d4f81729b1366_l3.png "Rendered by QuickLaTeX.com")after a certain time starts to take values either greater than 1 or less than 1 but always in the same range. This is the root cause of the vanishing and exploding gradients problem. In an LSTM, the term ![\frac{\partial c_{t}}{\partial c_{t-1}}     ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-e3d5cee87f7124d76f2633ba67ca32a9_l3.png "Rendered by QuickLaTeX.com")does not have a fixed pattern and can take any positive value at any time step. Thus, it is not guaranteed that for an infinite number of time steps, the term will converge to 0 or diverge completely. If the gradient starts converging towards zero, then the weights of the gates can be adjusted accordingly to bring it closer to 1. Since during the training phase, the network adjusts these weights only, it thus learns when to let the gradient converge to zero and when to preserve it.
