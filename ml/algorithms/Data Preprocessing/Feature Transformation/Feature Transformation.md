# Feature Transformation Techniques in Machine Learning 

Most machine learning algorithms are statistics dependent, meaning that all of the algorithms are indirectly using a statistical approach to solve the complex problems in the data. In statistics, the normal distribution of the data is one that a statistician desires to be. A [normal distribution](https://www.geeksforgeeks.org/mathematics-probability-distributions-set-3-normal-distribution/) of the data helps statisticians to solve the complex patterns of the data and gain valuable insights from the same. But for the algorithm scenario, a normal distribution of the data can not be desired every time with every type of dataset, which means the data which is not normally distributed needs preprocessing and cleaning before applying the [machine learning](https://www.geeksforgeeks.org/machine-learning/) algorithm to it.

In this article, we will be discussing the feature transformation techniques in machine learning which are used to transform the data from one form to another form, keeping the essence of the data. In simple words, the transformers are the type of functions that are applied to data that is not normally distributed, and once applied there is a high of getting normally distributed data.

There are 3 types of Feature transformation techniques:

1.  Function Transformers
2.  Power Transformers
3.  Quantile Transformers

Function Transformers
---------------------

Function transformers are the type of feature transformation technique that uses a particular function to transform the data to the normal distribution. Here the particular function is applied to the data observations.

There is not any thumb rule for the selection of function transformers, the function can be designed by anyone good at domain knowledge of the data, but mostly there are 5 types of function transformers that are used and which also solve the issue of normal distribution almost every time.

1.  Log Transform
2.  Square Transform
3.  Square Root Transform
4.  Reciprocal Transform
5.  Custom Transform

Let us try to discuss the core intuition of every transformation one by one.

### Log Transform

Log transform is one of the simplest transformations on the data in which the log is applied to every single distribution of the data and the result from the log is considered the final day to feed the machine learning algorithms.

Through experiments, it is proven that [log transforms](https://www.geeksforgeeks.org/numpy-log-python/) performs so well on the right-skewed data. It transforms the right-skewed data into normally distributed data so well.

Python3
-------

`from` `sklearn.preprocessing` `import` `FunctionTransformer`

`transform` `=` `FunctionTransformer(func``=``np.log1p)`

`transformed_data` `=` `transform.fit_transform(data)`

### Square Transform

Square transform is the type of transformer in which the square of the data is considered instead of the normal data. In simple words, in this transformed the data is applied with the [square function](https://www.geeksforgeeks.org/numpy-square-python/), where the square of every single observation will be considered as the final transformed data.

Python3
-------

`import` `numpy as np`

`tranformed_data` `=` `np.square(data)`

### Square Root Transform

In this transform, the [square root](https://www.geeksforgeeks.org/floor-square-root-without-using-sqrt-function-recursive/) of the data is calculated. This transform performs so well on the left-skewed data and efficiently transformed the left-skewed data into normally distributed data.

Python3
-------

`import` `numpy as np` 

`tranformed_data` `=` `np.sqrt(data)`

### Reciprocal Transform

In this transformation, the reciprocal of every observation is considered. This transform is useful in some of the datasets as the reciprocal of the observations works well to achieve normal distributions.

Python3
-------

`import` `numpy as np`

`tranformed_data` `=` `np.reciprocal(data)`

### Custom Transforms

In every dataset, the log and square root transforms can not be used, as every data can have different patterns and complexity. Based on the domain knowledge of the data, custom transformations can be applied to transform the data into a normal distribution. The custom transforms here can be any function or parameter like sin, cos, tan, cube, etc.

Python3
-------

`importy numpy as np`

`sin_tranformed_data` `=` `np.sin(data)`

`cos_tranformed_data` `=` `np.cos(data)`

`tan_tranformed_data` `=` `np.tan(data)`

Power Transformers 
-------------------

Power Transformation techniques are the type of feature transformation technique where the power is applied to the data observations for transforming the data.

There are two types of Power Transformation techniques:

1.  [Box-Cox Transform](https://www.geeksforgeeks.org/box-cox-transformation-using-python/)
2.  Yeo-Johnson Transform

### Box-Cox Transform

This transform technique is mainly used for transforming the data observations by applying power to them. The power of the data observations is denoted by Lambda(λ). There are mainly two conditions associated with the power in this transform, which is lambda equals zero and not equal to zero. The mathematical formulation of this transform is as follows:

![X_{i}^{\lambda}= \left\{\begin{matrix} \ln{X_i} & ;\;\;\;\;\mathrm{for } \;\; \lambda = 0 \\  \frac{X_{i}^{\lambda}-1}{\lambda} &  ;\;\;\;\;\mathrm{for } \;\; \lambda\neq 0\\  \end{matrix}\right.](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-c10fbcdf5d9f8ab2176cbb23a687e550_l3.png "Rendered by QuickLaTeX.com")

Here the lambda is the power applied to every data observation. Based upon the iteration technique every single value of the lambda is examined and the best fit value of the lambda is then applied to the data to transform it.

Here the transformed value of every data observation will lie between 5 to -5. One major disadvantage associated with this transformation technique is that this technique can only be applied to positive observations. it is not applicable for negative and zero values of the data observations.

Python3
-------

`from` `sklearn.preprocessing` `import` `PowerTransformer`

`boxcox` `=` `PowerTransformer(method``=``'box-cox'``)`

`data_transformed` `=` `boxcox.fit_transform(data)`

### Yeo Johnson Transform

This transformation technique is also a power transform technique, where the power of the data observations is applied to transform the data. This is an advanced form of a box cox transformations technique where it can be applied to even zero and negative values of data observations also.

The mathematical formulations of this transformations technique are as follows:

![X_{i}=  \left\{\begin{matrix} \frac{\left ( y+1 \right )^\lambda-1}{\lambda} & ;\mathrm{for}\;y\geq 0\;\mathrm{and}\;\lambda\neq 0 \\ \log\left ( y+1 \right ) & ;\mathrm{for}\;y\geq 0\;\mathrm{and}\;\lambda = 0 \\ \frac{\left (1-y \right )^{2-\lambda}-1}{2-\lambda} & ;\mathrm{for}\;y<0\;\mathrm{and}\;\lambda \neq 2  \\ -\log\left ( 1-y \right ) & ;\mathrm{for}\;y< 0\;\mathrm{and}\;\lambda = 2  \\ \end{matrix}\right. ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-8895dd00812fdf384b4ce2e645008e72_l3.png "Rendered by QuickLaTeX.com")

In this transformation technique, y represents the appropriate value of Xi. In scikit learn the default parameter is set to Yeo Johnson in the Power Transformer class.

Python3
-------

`from` `sklearn.preprocessing` `import` `PowerTransformer`

`boxcox` `=` `PowerTransformer()`

`data_transformed` `=` `boxcox.fit_transform(data)`

Quantile Transformers
---------------------

Quantile transformation techniques are the type of feature transformation technique that can be applied to NY numerical data observations. This transformation technique can be implemented using sklearn.

In this transformation technique, the input data can be fed to this transformer where this transformer makes the distribution of the output data normal to fed to the further machine learning algorithm.

Here there is a paramere called _output\_distribution_, which value can be set to _uniform_ or _normal_.

Python3
-------

`from` `sklearn.preprocessing` `import` `QuantileTransformer`

`quantile_trans` `=` `QuantileTransformer(output_distribution``=``'normal'``)`

`data_transformed` `=` `quantile.fit_transform(data)`

Key Takeaways
-------------

*   The featured transformation techniques are used to transform the data to normal distribution for better performance of the algorithm.
*   The Log transforms perform so well on the right-skewed data. Whereas the square root transformers perform so well on left-skewed data.
*   Based on the domain knowledge of the problem statement and the data, the custom data transformations technique can be also applied efficiently.
*   Box-Cox transformations can be applied to only positive data observations which return the transformed values between -5 to 5.
*   Yeo Johnson’s transformations technique can be applied to zero and negative values as well.

Conclusion
----------

In this article, we discussed some of the famous and most used data transformation techniques that are used to transform the data from any other distribution to normal distribution. this will help one to apply data preprocessing and cleaning techniques n the complex data easily and will help one to answer some of the interview questions related to it very efficiently.
