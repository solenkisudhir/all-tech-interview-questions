# ML | Why Logistic Regression in Classification ? 

Using Linear Regression, all predictions >= 0.5 can be considered as 1 and rest all < 0.5 can be considered as 0. But then the question arises why classification can’t be performed using it? **Problem –** Suppose we are classifying a mail as spam or not spam and our output is **y**, it can be 0(spam) or 1(not spam). In case of Linear Regression, hθ(x) can be > 1 or < 0. Although our prediction should be in between 0 and 1, the model will predict value out of the range i.e. maybe > 1 or < 0. So, that’s why for a Classification task, Logistic/Sigmoid Regression plays its role. 

Logistic regression is a statistical method commonly used in machine learning for binary classification problems, where the goal is to predict one of two possible outcomes, such as true/false or yes/no. Here are some reasons why logistic regression is widely used in classification tasks:

**Simple and interpretable**: Logistic regression is a relatively simple algorithm that is easy to understand and interpret. It can provide insights into the relationship between the independent variables and the probability of a particular outcome.

**Linear decision boundary:** Logistic regression can be used to model linear decision boundaries, which makes it useful for separating data points that belong to different classes.

**Efficient training:** Logistic regression can be trained quickly, even with large datasets, and is less computationally expensive than more complex models like neural networks.

**Robust to noise**: Logistic regression can handle noise in the input data and is less prone to overfitting compared to other machine learning algorithms.

**Works well with small datasets:** Logistic regression can perform well even when there is limited data available, making it a useful algorithm when dealing with small datasets.

   Overall, logistic regression is a popular and effective method for binary classification problems. However, it may not be suitable for more complex classification problems where there are multiple classes or nonlinear relationships between the input variables and the outcome.

![](https://media.geeksforgeeks.org/wp-content/uploads/20190502133352/Logistic_Regression.jpg)

![h_{\Theta} (x) = g (\Theta ^{T}x) z = \Theta ^{T}x g(z) = \frac{1}{1+e^{-z}} ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-75961e1beb9eae8c51b5221774bb39ce_l3.png "Rendered by QuickLaTeX.com")

Here, we plug **θTx** into logistic function where θ are the weights/parameters and **x** is the input and **hθ(x)** is the hypothesis function. **g()** is the sigmoid function.

![h_{\Theta} (x) = P( y =1 | x ; \Theta ) ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-2aa2d21399a52af17ec023fea24d6f8c_l3.png "Rendered by QuickLaTeX.com")

It means that y = 1 probability when x is parameterized to **θ** To get the discrete values 0 or 1 for classification, discrete boundaries are defined. The hypothesis function cab be translated as

![h_{\Theta} (x) \geq 0.5 \rightarrow y = 1 h_{\Theta} (x) < 0.5 \rightarrow y = 0 ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-1be99caf24afd06a4b60517fbc9342b1_l3.png "Rendered by QuickLaTeX.com")

![{g(z) \geq 0.5} \\ {\Rightarrow \Theta ^{T}x \geq 0.5} \\ {\Rightarrow z \geq 0.5 } ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-1d6473fcce63559b8265263aebc9d2bd_l3.png "Rendered by QuickLaTeX.com")

Decision Boundary is the line that distinguishes the area where y=0 and where y=1. These decision boundaries result from the hypothesis function under consideration. **Understanding Decision Boundary with an example –** Let our hypothesis function be

![h_{\Theta}(x)= g[\Theta_{0}+ \Theta_1x_1+\Theta_2x_2+ \Theta_3x_1^2 + \Theta_4x_2^2 ] ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-e1e3ae2d7c3940b94fda030e5b11222d_l3.png "Rendered by QuickLaTeX.com")

Then the decision boundary looks like ![](https://media.geeksforgeeks.org/wp-content/uploads/20190503112448/Logistics_Regression2-3.jpg) Let out weights or parameters be –

![\Theta=\begin{bmatrix} -1\\ 0\\ 0\\ 1\\ 1 \end{bmatrix} ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-5884f304827b5ebe61d387bd567a4103_l3.png "Rendered by QuickLaTeX.com")

So, it predicts y = 1 if

![-1 + x_{1}^2 + x_{2}^2 \geqslant 0 ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-8a4e17d6ccf473c7602a979ce392a64a_l3.png "Rendered by QuickLaTeX.com")

![\Rightarrow x_{1}^2 + x_{2}^2 \geqslant 1 ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-b07dfcc421f4a400aa6a537962dfd502_l3.png "Rendered by QuickLaTeX.com")

And that is the equation of a circle with radius = 1 and origin as the center. This is the Decision Boundary for our defined hypothesis.
