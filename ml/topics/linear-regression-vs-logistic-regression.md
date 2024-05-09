# Linear Regression vs Logistic Regression

Linear Regression and Logistic Regression are the two famous Machine Learning Algorithms which come under supervised learning technique. Since both the algorithms are of supervised in nature hence these algorithms use labeled dataset to make the predictions. But the main difference between them is how they are being used. The Linear Regression is used for solving Regression problems whereas Logistic Regression is used for solving the Classification problems. The description of both the algorithms is given below along with difference table.

![inear Regression vs Logistic Regression](https://static.javatpoint.com/tutorial/machine-learning/images/linear-regression-vs-logistic-regression.png)

Linear Regression:
------------------

*   Linear Regression is one of the most simple Machine learning algorithm that comes under Supervised Learning technique and used for solving regression problems.
*   It is used for predicting the continuous dependent variable with the help of independent variables.
*   The goal of the Linear regression is to find the best fit line that can accurately predict the output for the continuous dependent variable.
*   If single independent variable is used for prediction then it is called Simple Linear Regression and if there are more than two independent variables then such regression is called as Multiple Linear Regression.
*   By finding the best fit line, algorithm establish the relationship between dependent variable and independent variable. And the relationship should be of linear nature.
*   The output for Linear regression should only be the continuous values such as price, age, salary, etc. The relationship between the dependent variable and independent variable can be shown in below image:

![inear Regression vs Logistic Regression](https://static.javatpoint.com/tutorial/machine-learning/images/linear-regression-vs-logistic-regression2.png)

In above image the dependent variable is on Y-axis (salary) and independent variable is on x-axis(experience). The regression line can be written as:

```
y= a0+a1x+ ε

```


Where, a0 and a1 are the coefficients and ε is the error term.

Logistic Regression:
--------------------

*   Logistic regression is one of the most popular Machine learning algorithm that comes under Supervised Learning techniques.
*   It can be used for Classification as well as for Regression problems, but mainly used for Classification problems.
*   Logistic regression is used to predict the categorical dependent variable with the help of independent variables.
*   The output of Logistic Regression problem can be only between the 0 and 1.
*   Logistic regression can be used where the probabilities between two classes is required. Such as whether it will rain today or not, either 0 or 1, true or false etc.
*   Logistic regression is based on the concept of Maximum Likelihood estimation. According to this estimation, the observed data should be most probable.
*   In logistic regression, we pass the weighted sum of inputs through an activation function that can map values in between 0 and 1. Such activation function is known as **sigmoid function** and the curve obtained is called as sigmoid curve or S-curve. Consider the below image:

![inear Regression vs Logistic Regression](https://static.javatpoint.com/tutorial/machine-learning/images/linear-regression-vs-logistic-regression3.png)

*   The equation for logistic regression is:

![inear Regression vs Logistic Regression](https://static.javatpoint.com/tutorial/machine-learning/images/linear-regression-vs-logistic-regression4.png)

Difference between Linear Regression and Logistic Regression:



* Linear Regression: Linear regression is used to predict the continuous dependent variable using a given set of independent variables.
  * Logistic Regression: Logistic Regression is used to predict the categorical dependent variable using a given set of independent variables.
* Linear Regression: Linear Regression is used for solving Regression problem.
  * Logistic Regression: Logistic regression is used for solving Classification problems.
* Linear Regression: In Linear regression, we predict the value of continuous variables.
  * Logistic Regression: In logistic Regression, we predict the values of categorical variables.
* Linear Regression: In linear regression, we find the best fit line, by which we can easily predict the output.
  * Logistic Regression: In Logistic Regression, we find the S-curve by which we can classify the samples.
* Linear Regression: Least square estimation method is used for estimation of accuracy.
  * Logistic Regression: Maximum likelihood estimation method is used for estimation of accuracy.
* Linear Regression: The output for Linear Regression must be a continuous value, such as price, age, etc.
  * Logistic Regression: The output of Logistic Regression must be a Categorical value such as 0 or 1, Yes or No, etc.
* Linear Regression: In Linear regression, it is required that relationship between dependent variable and independent variable must be linear.
  * Logistic Regression: In Logistic regression, it is not required to have the linear relationship between the dependent and independent variable.
* Linear Regression: In linear regression, there may be collinearity between the independent variables.
  * Logistic Regression: In logistic regression, there should not be collinearity between the independent variable.


* * *

