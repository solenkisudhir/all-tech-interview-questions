# ML | Dummy variable trap in Regression Models 

Before learning about the dummy variable trap, let’s first understand what actually dummy variable is. 

**Dummy Variable in Regression Models:**   
In statistics, especially in regression models, we deal with various kinds of data. The data may be quantitative (numerical) or qualitative (categorical). The numerical data can be easily handled in regression models but we can’t use categorical data directly, it needs to be transformed in some way. 

For transforming categorical attributes to numerical attributes, we can use the label encoding procedure (label encoding assigns a unique integer to each category of data). But this procedure is not alone that suitable, hence, _**One hot encoding**_ is used in regression models following label encoding. This enables us to create new attributes according to the number of classes present in the categorical attribute i.e if there are _n_ number of categories in categorical attribute, _n_ new attributes will be created. These attributes created are called _**Dummy Variables**_. Hence, dummy variables are “proxy” variables for categorical data in regression models.   
These dummy variables will be created with one-hot _encoding_ and each attribute will have a value of either 0 or 1, representing the presence or absence of that attribute. 

**Dummy Variable Trap:**   
The Dummy variable trap is a scenario where there are attributes that are highly correlated (Multicollinear) and one variable predicts the value of others. When we use _one-hot encoding_ for handling the categorical data, then one dummy variable (attribute) can be predicted with the help of other dummy variables. Hence, one dummy variable is highly correlated with other dummy variables. Using all dummy variables for regression models leads to a _**dummy variable trap**_. So, the regression models should be designed to exclude one dummy variable. 

**For Example –**   
Let’s consider the case of gender having two values _male_ (0 or 1) and _female_ (1 or 0). Including both the dummy variable can cause redundancy because if a person is not male in such case that person is a female, hence, we don’t need to use both the variables in regression models. This will protect us from the dummy variable trap.  
 

  

