# ML | Normal Equation in Linear Regression

We know the Linear Regression model is a parameterized model which means that the model’s behavior and predictions are determined by a set of parameters or coefficients in the model. However, we use different methods for finding these parameters which give the lowest error on our dataset.  In this article, we will read one such article which is the normal equation. 

Normal Equation 
----------------

**Normal Equation** is an analytical approach to [Linear Regression](https://www.geeksforgeeks.org/ml-linear-regression/) with a [Least Square](https://www.geeksforgeeks.org/least-square-regression-line/) Cost Function. We can use the normal equation to directly compute the parameters of a model that minimizes the [Sum of the squared difference](https://www.geeksforgeeks.org/sum-of-squares-of-differences-between-all-pairs-of-an-array/) between the actual term and the predicted term. This method is quite useful when the dataset is small. However, with a large dataset, it may not be able to give us the best parameter of the model. 

The normal Equation is as follows:

![Normal equation formula ](https://media.geeksforgeeks.org/wp-content/uploads/Untitled-drawing-1-10.png)

Normal equation formula 

> In the above equation, 
> 
> **θ:** hypothesis parameters that define it the best.   
> **X:** Input feature value of each instance.   
> **Y:** Output value of each instance. 

### Maths Behind the Equation:

Given the hypothesis function 

![h(\theta) = \theta_0 x_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + \ldots + \theta_n x_n](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-b6587e88fdd9de4602b60ff7041f08f3_l3.png "Rendered by QuickLaTeX.com")

where,   
**n:** the no. of features in the data set.   
**x0:** 1 (for vector multiplication)   
Notice that this is a dot product between θ and x values. So for the convenience to solve we can write it as:

![h(\theta) = \theta^T X](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-673b1ae12db1446d3658140f3f498043_l3.png "Rendered by QuickLaTeX.com")

The motive in Linear Regression is to minimize the cost function: 

![J(\Theta) = \frac{1}{2m} \sum_{i = 1}^{m} \frac{1}{2} [h_{\Theta}(x^{(i)}) - y^{(i)}]^{2} ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-964ab765d209f44bdaf2dd3c4a8ed6a7_l3.png "Rendered by QuickLaTeX.com")

  
where,   
**xi:** the input value of iih training example.   
**m:** no. of training instances   
**n:** no. of data-set features   
**yi:** the expected result of ith instance 

Let us represent the cost function in a vector form.

![](https://media.geeksforgeeks.org/wp-content/uploads/3-43.jpg)

We have ignored 1/2m here as it will not make any difference in the working. It was used for mathematical convenience while calculating gradient descent. But it is no more needed here. 

![](https://media.geeksforgeeks.org/wp-content/uploads/5-20.jpg)

![](https://media.geeksforgeeks.org/wp-content/uploads/6-12.jpg)

**xij:** the value of jih feature in iih training example. 

This can further be reduced to 

![X\theta - y       ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-727f499e84fff463f3da409dd1c2d5b3_l3.png "Rendered by QuickLaTeX.com")

But each residual value is squared. We cannot simply square the above expression. As the square of a vector/[matrix](https://www.geeksforgeeks.org/matrix/) is not equal to the square of each of its values. So to get the squared value, multiply the vector/matrix with its transpose. So, the final equation derived is 

![(X\theta - y)^T(X\theta - y)](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-f3be30f62f46a281d466c481fca26b3d_l3.png "Rendered by QuickLaTeX.com")

Therefore, the cost function is 

![Cost = (X\theta - y)^T(X\theta - y)](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-adfa900cf0aee4368af1050685b4b265_l3.png "Rendered by QuickLaTeX.com")

### Calculating the value of θ using the partial derivative of the Normal Equation

We will take a partial derivative of the cost function with respect to the parameter theta. Note that in [partial derivative](https://www.geeksforgeeks.org/program-derivative-polynomial/) we treat all variables except the parameter theta as constant. 

![\frac{\partial J}{\partial \theta} = \frac{\partial}{\partial \theta} \left( (X\theta - y)^T(X\theta - y) \right)](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-48265e521b597311bda7ed87c40ce641_l3.png "Rendered by QuickLaTeX.com")

![\frac{\partial J}{\partial \theta} = 2X^T(X\theta - y)](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-227a8741df456be8ad297558b777effd_l3.png "Rendered by QuickLaTeX.com")

We know to find the optimum value of any partial derivative equation we have to equate it to 0. 

![Cost(\theta) = 0](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-4ff66171827abba5359c8e704c765a67_l3.png "Rendered by QuickLaTeX.com")

 ![2X^TX\theta - X^Ty = 0](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-132a5a569cdbc7f69b4833fb775cfeff_l3.png "Rendered by QuickLaTeX.com")

![2X^TX\theta = X^Ty    ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-a4d7172a0f019b318505adb1a178052d_l3.png "Rendered by QuickLaTeX.com") 

Finally, we can solve for θ by multiplying both sides of the equation by the inverse of (2XᵀX):

![\theta = (X^\intercal X)^{-1}X^\intercal y   ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-e80a469cc75446a519eb2581ce252faf_l3.png "Rendered by QuickLaTeX.com") 

![](https://media.geeksforgeeks.org/wp-content/uploads/Untitled-drawing-1-10.png)

We can implement this normal equation using [Python](https://www.geeksforgeeks.org/python-programming-language/) programming language. 

Python implementation of Normal Equation 
-----------------------------------------

We will create a synthetic dataset using [sklearn](https://www.geeksforgeeks.org/python-create-test-datasets-using-sklearn/) having only one feature. Also, we will use numpy for mathematical computation like for getting the matrix to transform and inverse of the dataset.  Also, we will use try and except block in our function so that in case if our input data matrix is singular our function will not be throwing an error.

Python3
-------

`import` `numpy as np`

`from` `sklearn.datasets` `import` `make_regression`

`X, y` `=` `make_regression(n_samples``=``100``, n_features``=``1``,`

                       `n_informative``=``1``, noise``=``10``, random_state``=``10``)`

`def` `linear_regression_normal_equation(X, y):`

    `X_transpose` `=` `np.transpose(X)`

    `X_transpose_X` `=` `np.dot(X_transpose, X)`

    `X_transpose_y` `=` `np.dot(X_transpose, y)`

    `try``:`

        `theta` `=` `np.linalg.solve(X_transpose_X, X_transpose_y)`

        `return` `theta`

    `except` `np.linalg.LinAlgError:`

        `return` `None`

`X_with_intercept` `=` `np.c_[np.ones((X.shape[``0``],` `1``)), X]`

`theta` `=` `linear_regression_normal_equation(X_with_intercept, y)`

`if` `theta` `is` `not` `None``:`

    `print``(theta)`

`else``:`

    `print``(``"Unable to compute theta. The matrix X_transpose_X is singular."``)`

**Output:**

```
[ 0.52804151 30.65896337]
```


### To Predict on New Test Data Instances

Since we have trained our model and have found parameters that give us the lowest error. We can use this parameter to predict on new unseen test data. 

Python3
-------

`def` `predict(X, theta):`

    `predictions` `=` `np.dot(X, theta)`

    `return` `predictions`

`X_test` `=` `np.array([[``1``], [``4``]])`

`X_test_with_intercept` `=` `np.c_[np.ones((X_test.shape[``0``],` `1``)), X_test]`

`predictions` `=` `predict(X_test_with_intercept, theta)`

`print``(``"Predictions:"``, predictions)`

**Output:**

```
Predictions: [ 31.18700488  123.16389501]
```
