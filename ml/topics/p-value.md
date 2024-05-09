# What is P-Value
In Statistical hypothesis testing, the P-value or sometimes called probability value, is used to observe the test results or more extreme results by assuming that the null hypothesis (H0) is true. In data science, there are lots of concepts that are borrowed from different disciplines, and the p-value is one of them. The concept of p-value comes from statistics and widely used in [machine learning](https://www.javatpoint.com/machine-learning) and [data science](https://www.javatpoint.com/data-science).

*   P-value is also used as an alternative to determine the point of rejection in order to provide the smallest significance level at which the null hypothesis is least or rejected.
*   It is expressed as the level of significance that lies between 0 and 1, and _if there is smaller p-value, then there would be strong evidence to reject the null hypothesis._ If the value of p-value is very small, then it means the observed output is feasible but doesn't lie under the null hypothesis conditions (H0).
*   The p-value of 0.05 is known as the level of significance (**α**). Usually, it is considered using two suggestions, which are given below:
    *   **If p-value>0.05:** The large p-value shows that the null hypothesis needs to be accepted.
    *   **If p-value<0.05:** The small p-value shows that the null hypothesis needs to be rejected, and the result is declared as statically significant.

In Statistics, our main goal is to determine the statistical significance of our result, and this statistical significance is made on below three concepts:

*   Hypothesis Testing
*   Normal Distribution
*   Statistical Significance

Let's understand each of them.

### Hypothesis Testing

**Hypothesis testing** can be defined between two terms; **Null hypothesis** and **Alternative Hypothesis**. It is used to check the validity of the null hypothesis or claim made using the sample data. Here, the **null hypothesis (H0)** is defined as the hypothesis with no statistical significance between two variables, while an **alternative hypothesis** is defined as the hypothesis with a statistical significance between the two variables. No significant relationship between the two variables tells that one variable will not affect the other variable. Thus, the Null hypothesis tells that what you are going to prove doesn't actually happen. If the independent variable doesn't affect the dependent variable, then it shows the alternative hypothesis condition.

In a simple way, we can say that _in hypothesis testing, first, we make a claim that is assumed as a null hypothesis using the sample data. If this claim is found invalid, then the alternative hypothesis is selected._ This assumption or claim is validated using the p-value to see if it is statistically significant or not using the evidence. If the evidence supports the alternative hypothesis, then the null hypothesis is rejected.

**Steps for Hypothesis testing**

Below are the steps to perform an experiment for hypothesis testing:

1.  Claim or state a Null hypothesis for the experiment.
2.  State the alternative hypothesis, which is opposite to the null hypothesis.
3.  Set the value of alpha to be used in the experiment.
4.  Determine the z-score using the normal distribution.
5.  Compare the P-value to validate the statistical significance.

### Normal Distribution

The normal distribution, which is also known as Gaussian distribution, is the Probability distribution function. It is symmetric about the mean, and use to see the distribution of data using a graph plot. It shows that data near the mean is more frequent to occur as compared to data which is far from the mean, and it looks like a **bell-shaped curve**. The two main terms of the normal distribution are mean(**μ**) and standard deviation(σ). For a normal distribution, the mean is zero, and the standard deviation is 1.

In hypothesis testing, we need to calculate z-score. **Z-score** is the number of standard deviations from the mean of data-point.

![P-Value](https://static.javatpoint.com/tutorial/machine-learning/images/machine-learning-p-value.png)

Here, the z-score inform us that where the data lies compared to the average population.

### Statistical significance:

To determine the statistical significance of the hypothesis test is the goal of calculating the p-value. To do this, first, we need to set a threshold, which is said to be alpha. We should always set the value of alpha before the experiment, and it is set to be either 0.05 or 0.01(depending on the type of problem).

The result is concluded as a significant result if the observed p-value is lower than alpha.

Errors in P-value
-----------------

Two types of errors are defined for the p-value; these errors are given below:

1.  Type I error
2.  Type II error

### Type I Error:

It is defined as the incorrect or false rejection of the Null hypothesis. For this error, the maximum probability is alpha, and it is set in advance. The error is not affected by the sample size of the dataset. The type I error increases as we increase the number of tests or endpoints.

### Type II error

Type II error is defined as the wrong acceptance of the Null hypothesis. The probability of type II error is beta, and the beta depends upon the sample size and value of alpha. The beta cannot be determined as the function of the true population effect. The value of beta is inversely proportional to the sample size, and it means beta decreases as the sample size increases.

The value of beta also decreases when we increase the number of tests or endpoints.

We can understand the relationship between hypothesis testing and decision on the basis of the below table:


|           |Decision        |                |
|-----------|----------------|----------------|
|Truth      |Accept H0       |Reject H0       |
|H0 is true |Correct decision|Type I error    |
|H0 is false|Type II error   |Correct Decision|


### Importance of P-value

The importance of p-value can be understood in two aspects:

*   **Statistics Aspect:** In statistics, the concept of the p-value is important for hypothesis testing and statistical methods such as Regression.
*   **Data Science Aspect:** In data science also, it is one of the important aspect Here the smaller p-value shows that there is an association between the predictor and response. It is advised while working with the machine learning problem in data science, the p-value should be taken carefully.

* * *