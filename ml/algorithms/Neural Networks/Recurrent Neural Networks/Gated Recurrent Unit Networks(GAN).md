# Gated Recurrent Unit Networks

Gated Recurrent Unit (GRU) is a type of recurrent neural network (RNN) that was introduced by Cho et al. in 2014 as a simpler alternative to Long Short-Term Memory (LSTM) networks. Like LSTM, GRU can process sequential data such as text, speech, and time-series data.

The basic idea behind GRU is to use gating mechanisms to selectively update the hidden state of the network at each time step. The gating mechanisms are used to control the flow of information in and out of the network. The GRU has two gating mechanisms, called the reset gate and the update gate.

The reset gate determines how much of the previous hidden state should be forgotten, while the update gate determines how much of the new input should be used to update the hidden state. The output of the GRU is calculated based on the updated hidden state.

The equations used to calculate the reset gate, update gate, and hidden state of a GRU are as follows:

> Reset gate: **r\_t =** **sigmoid(W\_r \* \[h\_{t-1}, x\_t\])**  
> Update gate: **z\_t** **\=** **sigmoid(W\_z \* \[h\_{t-1}, x\_t\])**  
> Candidate hidden state: **h\_t’** **\= tanh(W\_h \* \[r\_t \* h\_{t-1}, x\_t\])**  
> Hidden state: **h\_t = (1 – z\_t) \* h\_{t-1} + z\_t \* h\_t’**  
> where W\_r, W\_z, and W\_h are learnable weight matrices, x\_t is the input at time step t, h\_{t-1} is the previous hidden state, and h\_t is the current hidden state.

In summary, GRU networks are a type of RNN that use gating mechanisms to selectively update the hidden state at each time step, allowing them to effectively model sequential data. They have been shown to be effective in various natural language processing tasks, such as language modeling, machine translation, and speech recognition

**Prerequisites: Recurrent Neural Networks, Long Short Term Memory Networks**

To solve the Vanishing-Exploding gradients problem often encountered during the operation of a basic Recurrent Neural Network, many variations were developed. One of the most famous variations is the **Long Short Term Memory Network(LSTM)**. One of the lesser-known but equally effective variations is the **Gated Recurrent Unit Network(GRU)**. 

Unlike LSTM, it consists of only three gates and does not maintain an Internal Cell State. The information which is stored in the Internal Cell State in an LSTM recurrent unit is incorporated into the hidden state of the Gated Recurrent Unit. This collective information is passed onto the next Gated Recurrent Unit. The different gates of a GRU are as described below:- 

1.  **Update Gate(z):** It determines how much of the past knowledge needs to be passed along into the future. It is analogous to the Output Gate in an LSTM recurrent unit.
2.  **Reset Gate(r):** It determines how much of the past knowledge to forget. It is analogous to the combination of the Input Gate and the Forget Gate in an LSTM recurrent unit.
3.  **Current Memory Gate(**![  \overline{h}_{t}    ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-1b04e2d405797b880ca65eb547f5bf24_l3.png "Rendered by QuickLaTeX.com")**):** It is often overlooked during a typical discussion on Gated Recurrent Unit Network. It is incorporated into the Reset Gate just like the Input Modulation Gate is a sub-part of the Input Gate and is used to introduce some non-linearity into the input and to also make the input Zero-mean. Another reason to make it a sub-part of the Reset gate is to reduce the effect that previous information has on the current information that is being passed into the future.

The basic work-flow of a Gated Recurrent Unit Network is similar to that of a basic Recurrent Neural Network when illustrated, the main difference between the two is in the internal working within each recurrent unit as Gated Recurrent Unit networks consist of gates which modulate the current input and the previous hidden state. 

![](https://media.geeksforgeeks.org/wp-content/uploads/20190703110443/unrolled3.png)

**Working of a Gated Recurrent Unit:** 

*   Take input the current input and the previous hidden state as vectors. 
*   Calculate the values of the three different gates by following the steps given below:-   
    1.  For each gate, calculate the parameterized current input and previously hidden state vectors by performing element-wise multiplication (Hadamard Product) between the concerned vector and the respective weights for each gate.
    2.  Apply the respective activation function for each gate element-wise on the parameterized vectors. Below given is the list of the gates with the activation function to be applied for the gate.

```
Update Gate : Sigmoid Function
Reset Gate  : Sigmoid Function
```


*   The process of calculating the Current Memory Gate is a little different. First, the Hadamard product of the Reset Gate and the previously hidden state vector is calculated. Then this vector is parameterized and then added to the parameterized current input vector. 
    
    ![\overline{h}_{t} = tanh(W\odot x_{t}+W\odot (r_{t}\odot h_{t-1}))](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-a26c7eea5a299205a23cc3a28cdea84a_l3.png "Rendered by QuickLaTeX.com")
    
*   To calculate the current hidden state, first, a vector of ones and the same dimensions as that of the input is defined. This vector will be called ones and mathematically be denoted by 1. First, calculate the Hadamard Product of the update gate and the previously hidden state vector. Then generate a new vector by subtracting the update gate from ones and then calculate the Hadamard Product of the newly generated vector with the current memory gate. Finally, add the two vectors to get the currently hidden state vector.  
    ![h_{t} = z_{t}\odot h_{t-1} + (1-z_{t})\odot \overline{h}_{t}       ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-7dbf9ffe84c916e0627516c5625c9f91_l3.png "Rendered by QuickLaTeX.com")
    
    The above-stated working is stated as below:- 
    

![](https://media.geeksforgeeks.org/wp-content/uploads/20190703131515/working5.png)

Note that the blue circles denote element-wise multiplication. The positive sign in the circle denotes vector addition while the negative sign denotes vector subtraction(vector addition with negative value). The weight matrix W contains different weights for the current input vector and the previous hidden state for each gate. 

Just like Recurrent Neural Networks, a GRU network also generates an output at each time step and this output is used to train the network using gradient descent.  

![](https://media.geeksforgeeks.org/wp-content/uploads/20190703135758/yt3.png)

Note that just like the workflow, the training process for a GRU network is also diagrammatically similar to that of a basic Recurrent Neural Network and differs only in the internal working of each recurrent unit. 

The Back-Propagation Through Time Algorithm for a Gated Recurrent Unit Network is similar to that of a Long Short Term Memory Network and differs only in the differential chain formation. 

Let ![\overline{y}_{t}       ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-2d9b0090c89ddbd4d083bd7c2f92867d_l3.png "Rendered by QuickLaTeX.com")be the predicted output at each time step and ![y_{t}       ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-570302a186165a2b8597b1027b1c171e_l3.png "Rendered by QuickLaTeX.com")be the actual output at each time step. Then the error at each time step is given by:- 

![E_{t} = -y_{t}log(\overline{y}_{t})       ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-4d0e5ff99e872c1284dd34af8d2e1f92_l3.png "Rendered by QuickLaTeX.com")

The total error is thus given by the summation of errors at all time steps. 

![E = \sum _{t} E_{t}       ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-8983880ee16cc4d2fcfea2c478a4e91f_l3.png "Rendered by QuickLaTeX.com")  
![\Rightarrow E = \sum _{t} -y_{t}log(\overline{y}_{t})       ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-1cef88e521aaa6e0d996b40bcb707bb5_l3.png "Rendered by QuickLaTeX.com")

Similarly, the value ![\frac{\partial E}{\partial W}       ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-e76cb76c042d277e91e67d97bd9e62e0_l3.png "Rendered by QuickLaTeX.com")can be calculated as the summation of the gradients at each time step. 

![\frac{\partial E}{\partial W} = \sum _{t} \frac{\partial E_{t}}{\partial W}       ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-09337962778e2f09d958fa0ea9287b5c_l3.png "Rendered by QuickLaTeX.com")

Using the chain rule and using the fact that ![\overline{y}_{t}       ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-2d9b0090c89ddbd4d083bd7c2f92867d_l3.png "Rendered by QuickLaTeX.com")is a function of ![h_{t}       ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-f0a0b104662098326604ce66483d543a_l3.png "Rendered by QuickLaTeX.com")and which indeed is a function of ![\overline{h}_{t}       ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-6743d5f13028adbe18113dd5893970ec_l3.png "Rendered by QuickLaTeX.com"), the following expression arises:- 

![\frac{\partial E_{t}}{\partial W} = \frac{\partial E_{t}}{\partial \overline{y}_{t}}\frac{\partial \overline{y}_{t}}{\partial h_{t}}\frac{\partial h_{t}}{\partial h_{t-1}}\frac{\partial h_{t-1}}{\partial h_{t-2}}......\frac{\partial h_{0}}{\partial W}       ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-675bd06908be679604003337a6cb9d34_l3.png "Rendered by QuickLaTeX.com")

Thus the total error gradient is given by the following:- 

![\frac{\partial E}{\partial W} = \sum _{t}\frac{\partial E_{t}}{\partial \overline{y}_{t}}\frac{\partial \overline{y}_{t}}{\partial h_{t}}\frac{\partial h_{t}}{\partial h_{t-1}}\frac{\partial h_{t-1}}{\partial h_{t-2}}......\frac{\partial h_{0}}{\partial W}       ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-c91606f3088b5e1a75ef6af98083954d_l3.png "Rendered by QuickLaTeX.com")

Note that the gradient equation involves a chain of ![\partial {h}_{t}       ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-03c927ae997b6c277090d8a633ce7ca6_l3.png "Rendered by QuickLaTeX.com")which looks similar to that of a basic Recurrent Neural Network but this equation works differently because of the internal workings of the derivatives of ![h_{t}       ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-f0a0b104662098326604ce66483d543a_l3.png "Rendered by QuickLaTeX.com"). 

**How do Gated Recurrent Units solve the problem of vanishing gradients?**

The value of the gradients is controlled by the chain of derivatives starting from ![\frac{\partial h_{t}}{\partial h_{t-1}}       ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-d0175360140d986f972434b95f0f7226_l3.png "Rendered by QuickLaTeX.com"). Recall the expression for ![h_{t}       ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-f0a0b104662098326604ce66483d543a_l3.png "Rendered by QuickLaTeX.com"):- 

![h_{t} = z_{t}\odot h_{t-1} + (1-z_{t})\odot \overline{h}_{t}       ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-7dbf9ffe84c916e0627516c5625c9f91_l3.png "Rendered by QuickLaTeX.com")

Using the above expression, the value for ![\frac{\partial {h}_{t}}{\partial {h}_{t-1}}       ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-636ed321132d2113d759b7fb447c44c5_l3.png "Rendered by QuickLaTeX.com")is:- 

![\frac{\partial h_{t}}{\partial h_{t-1}} = z + (1-z)\frac{\partial \overline{h}_{t}}{\partial h_{t-1}}       ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-7139087207ea7420530d2b180fabd663_l3.png "Rendered by QuickLaTeX.com")

Recall the expression for ![\overline{h}_{t}       ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-6743d5f13028adbe18113dd5893970ec_l3.png "Rendered by QuickLaTeX.com"):- 

![\overline{h}_{t} = tanh(W\odot x_{t}+W\odot (r_{t}\odot h_{t-1}))       ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-9ff9f92d4168e621c947c105e4a65758_l3.png "Rendered by QuickLaTeX.com")

Using the above expression to calculate the value of ![\frac{\partial \overline{h_{t}}}{\partial h_{t-1}}       ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-a760646eb68e31fa804b76428fc949a8_l3.png "Rendered by QuickLaTeX.com"):- 

![\frac{\partial \overline{h_{t}}}{\partial h_{t-1}} = \frac{\partial (tanh(W\odot x_{t}+W\odot (r_{t}\odot h_{t-1})))}{\partial h_{t-1}} \Rightarrow \frac{\partial \overline{h_{t}}}{\partial h_{t-1}} = (1-\overline{h}_{t}^{2})(W\odot r)       ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-7de4493cab2cbd136769f21973f61a37_l3.png "Rendered by QuickLaTeX.com")

Since both the update and reset gate use the sigmoid function as their activation function, both can take values either 0 or 1. 

**Case 1(z = 1):**

In this case, irrespective of the value of ![r       ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-8dfec8c41edba991b0d16621f0bc89fb_l3.png "Rendered by QuickLaTeX.com"), the term ![\frac{\partial \overline{h_{t}}}{\partial h_{t-1}}       ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-a760646eb68e31fa804b76428fc949a8_l3.png "Rendered by QuickLaTeX.com")is equal to z which in turn is equal to 1. 

**Case 2A(z=0 and r=0):**

In this case, the term ![\frac{\partial \overline{h_{t}}}{\partial h_{t-1}}       ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-a760646eb68e31fa804b76428fc949a8_l3.png "Rendered by QuickLaTeX.com")is equal to 0. 

**Case 2B(z=0 and r=1):**

In this case, the term ![\frac{\partial \overline{h_{t}}}{\partial h_{t-1}}       ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-a760646eb68e31fa804b76428fc949a8_l3.png "Rendered by QuickLaTeX.com")is equal to ![(1-\overline{h}_{t}^{2})(W)       ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-9b87486ab405d5554e2f3b2ab336eede_l3.png "Rendered by QuickLaTeX.com"). This value is controlled by the weight matrix which is trainable and thus the network learns to adjust the weights in such a way that the term ![\frac{\partial \overline{h_{t}}}{\partial h_{t-1}}       ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-a760646eb68e31fa804b76428fc949a8_l3.png "Rendered by QuickLaTeX.com")comes closer to 1. 

Thus the Back-Propagation Through Time algorithm adjusts the respective weights in such a manner that the value of the chain of derivatives is as close to 1 as possible.

