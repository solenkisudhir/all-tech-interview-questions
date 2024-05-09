# Least Angle Regression
LARS and forward stepwise regression are comparable. It locates the characteristic most associated with the target at each stage. When many functions have the same correlation, the way behaves in an equiangular path to the functions rather than continuing alongside the same feature.

### The Advantages of LARS are:

*   It is numerically efficient when the number of features is much more than the number of samples.
*   It has the same order of complexity as regular least squares and computes just as quickly as forward selection.
*   It generates a full piecewise linear solution route to fine-tune the model, such as cross-validation.
*   The coefficients of two features nearly equally linked with the target should rise roughly at the same rate. As a result, the algorithm is more stable and acts as intuition would predict.
*   It may be altered to get answers for different estimators, such as the Lasso.

### The following are some of the LARS method's drawbacks:

*   LARS is more susceptible to the effects of noise because its foundation is an iterative refitting of the residuals.
*   Standardized coefficients are displayed as a function of shrinkage proportion.
*   In contrast to classification, which predicts categorical or discrete values, regression is a supervised machine-learning task that may predict continuous values (real numbers).
*   For high-dimensional records with a variety of properties, the Least Angle Regression (LARS) set of rules is used. Forward stepwise regression and least angle regression are fairly comparable. LARS determines the property most strongly connected with the goal value at each stage since it is used with data that includes several attributes.
*   The connection between two attributes may apply to more than one of them. In this case, LARS takes the attributes into account and moves in a direction that is perpendicular to the attributes. This approach is known as least angle regression for just this reason. LARS moves in the path calculated to be best without overfitting the model.

### Algorithm:

*   In this case, LARS takes the attributes into account and moves in a direction that is perpendicular to the attributes.
*   This approach is known as least angle regression for just this reason. LARS moves in the path calculated to be best without overfitting the model.

#### NOTE: The discrepancy between the actual and predicted values is known as the residual. Variable in this context suggests an attribute.

The regression line should be moved at the least angle between the two variables when there is a correlation between two variables.

### Least Angle Regression functions as follows mathematically:

*   All coefficients are set to zero ('B').
*   It is discovered that the predictor, xj, is most linked with y.
*   When you identify another predictor, xk, with an equal or greater correlation to y than xj, you should cease increasing the coefficient Bj in that direction.
*   (Bj, Bk) should be extended in an equiangular direction to both xj and xk.
*   Repeat this process until the model includes every predictor.

### Implementation of Least Angle Regression in Python3:

For this example, we'll utilize the Boston housing dataset, which contains information on the median price of homes in the greater Boston, Massachusetts, area. Here are more details regarding this dataset.

The maximum r2 value is 1.0. If the predictor consistently predicts a constant value, regardless of the values of the attributes, it can also be negative and equal to 0.

**Advantages:**

*   Though computationally slower than an immediate option, it may occasionally be more correct.
*   When the number of features exceeds the number of data instances, it is numerically highly efficient.
*   It is simple to adapt it to provide answers for different estimators.

**Disadvantages:**

*   Since Least Angle Regression is so sensitive to noise, its output occasionally can be unreliable.

Finding the variable best linked with the answer is the first stage in LAR. Instead of fully fitting this variable, LAR continuously advances the coefficient of this variable towards its least squares value, reducing the absolute magnitude of its association with the evolving residual. The process is stopped when another variable "catches up" regarding correlation with the residual. The second variable then becomes a member of the active set, and their coefficients are shifted close to one another to maintain their tied and waning correlations.

Once the full least-squares fit is achieved, the process is repeated until all variables are included in the model. The LAR algorithm uses the same order of computation as a single least squares fit with the p predictors, making it incredibly efficient. To obtain the complete least squares estimates, least angle regression always requires p steps. Even though they are sometimes relatively comparable, the lasso path can contain more steps than p.

### More specifically, LARS functions as follows:

*   For the sake of simplicity, let's assume that our explanatory variables have been standardized to have a zero mean and a unit variance and that our response variable has a zero mean.
*   Start your model with no variables.
*   The variable having the strongest correlation to the residual is $ x\_1 $. (Note that the variable with the highest correlation to the residual is also the one with the lowest angle; hence, the name.)
*   Continue moving in this direction until another variable, $ x\_2 $, correlates equally to this one.
*   Start moving at this point so that the residual maintains an equal correlation with variables $ x\_1 $ and $ x\_2 $ (i.e., the residual forms equal angles with both variables), and continue moving until a variable $ x\_3 $ achieves an equal correlation with our residual.
*   And so on until we decide that our model is large enough.

It's simple to adapt it to provide solutions for different estimators.

*   All coefficients bj are set to zero at the beginning.
*   Choose the factor xj that is best associated with y.
*   The coefficient bj should be increased in the direction of the y-axis's association with it. Along the route, take residuals r=y-yhat. Stop when another predictor, xk, has the same degree of correlation with r as xj.

Surprisingly, with a single tweak, this approach offers all of the lasso solutions over the whole path as s is modified from 0 to infinity.

In statistics and machine learning, the popular and effective linear regression technique, Least Angle Regression (LARS), is used for feature selection and modelling. This approach aims to minimize overfitting by selecting the optimum subset of predictor variables to account for the variation in the response variable. LARS is particularly helpful when working with high-dimensional datasets where the number of predictor variables is significantly more than the number of observations. It relies on the concepts of forward stepwise regression.

### Key Steps in LARS:

1.  **Initialization:** LARS determines the predictor with the highest correlation with the response variable after setting all coefficients to zero at the start.
2.  **Adding Predictors:** LARS gradually incorporates this predictor into the model as it approaches its ordinary least squares (OLS) coefficient.
3.  **Ongoing Monitoring:** LARS determines the predictor at each step that is "most correlated" with the residual (the discrepancy between the actual response and the forecast made by the present model). The predictors' coefficients are then moved in the direction with the least angle (thus the name) until another predictor exhibits an equivalent correlation with the residual.
4.  **Active Set:** Predictors with non-zero coefficients in the present model comprise LARS' active set of predictors. Predictors are included in the active set when they exhibit an equal correlation with the residual.
5.  **Shrinkage:** LARS additionally includes shrinkage, which regulates how quickly coefficients are introduced to the model. This parameter makes sure the approach doesn't overfit and stays stable.
6.  **Stopping Criteria:** LARS keeps on until it either reaches a predefined number of predictors or incorporates every predictor. The user can specify the stopping condition.

### Benefits of LARS:

1.  **Effective:** LARS handles high-dimensional datasets effectively, making it appropriate for contemporary data analysis when the number of predictors can be significantly greater than the number of observations.
2.  **Regularisation:** By managing the rate at which predictors are added, LARS automatically incorporates a sort of regularisation. By doing this, overfitting is avoided.
3.  **Interpretability:** LARS offers a transparent procedure for including predictors in the model, making it simpler to grasp the significance of each variable.
4.  **LARS:** Least Angle Regression handles multicollinearity elegantly because it doesn't introduce predictors strongly correlated with those already present in the model.

### Applications of LARS:

1.  **Data Mining:** Data mining jobs employ LARS for feature selection to find the most pertinent variables for predictive modelling.
2.  **Machine Learning:** LARS can be utilized for regression problems in machine learning, particularly when working with high-dimensional data.
3.  **Variable Selection:** In disciplines including genetics, economics, and environmental science, researchers utilize LARS to choose pertinent variables from enormous datasets.

LARS has undergone several iterations to accommodate various requirements and limitations. LARS-lasso, LARS-EN (Elastic Net), and various variations fall under this category.

Conclusion:
-----------

In conclusion, Least Angle Regression (LARS) is a useful and effective method for feature selection and linear regression. It was created to deal with the difficulties presented by high-dimensional datasets, where the number of predictor variables vastly outnumbers the number of observations. In addition to handling multicollinearity, offering a clear and understandable approach for choosing predictors, and incorporating a type of regularisation to prevent overfitting, LARS has several important features.

Numerous domains, including data mining, machine learning, genetics, economics, and environmental research, have discovered extensive uses for LARS. Due to its versatility and efficiency, it is a crucial tool for data analysts, academics, and machine learning practitioners.

* * *