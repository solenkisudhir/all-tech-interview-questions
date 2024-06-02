# Chi-Square Test for Feature Selection - Mathematical Explanation 

One of the primary tasks involved in any supervised Machine Learning venture is to select the best features from the given dataset to obtain the best results. One way to select these features is the Chi-Square Test. Mathematically, a Chi-Square test is done on two distributions two determine the level of similarity of their respective variances. In its **null hypothesis**, it assumes that the given distributions are independent. This test thus can be used to determine the best features for a given dataset by determining the features on which the output class label is most dependent. For each feature in the dataset, the ![\chi ^{2}  ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-b3ddf55d91abd66ec2516c77d21401de_l3.png "Rendered by QuickLaTeX.com")is calculated and then ordered in descending order according to the ![\chi ^{2}  ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-b3ddf55d91abd66ec2516c77d21401de_l3.png "Rendered by QuickLaTeX.com")value. The higher the value of ![\chi ^{2}  ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-b3ddf55d91abd66ec2516c77d21401de_l3.png "Rendered by QuickLaTeX.com"), the more dependent the output label is on the feature and higher the importance the feature has on determining the output. Let the feature in question have m attribute values and the output have k class labels. Then the value of ![\chi ^{2}  ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-b3ddf55d91abd66ec2516c77d21401de_l3.png "Rendered by QuickLaTeX.com")is given by the following expression:

```
\chi ^{2} = \sum _{i=1}^{m} \sum _{j=1}^{k}\frac{(O_{ij}-E_{ij})^{2}}{E_{ij}}
```


where ![O_{ij}  ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-b07b7934a1df6d7c452d163e0ff709f5_l3.png "Rendered by QuickLaTeX.com")– Observed frequency ![E_{ij}  ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-aa401cf61e076753060dd58aea6475f5_l3.png "Rendered by QuickLaTeX.com")– Expected frequency For each feature, a contingency table is created with m rows and k columns. Each cell (i,j) denotes the number of rows having attribute feature as i and class label as k. Thus each cell in this table denotes the observed frequency. To calculate the expected frequency for each cell, first, the proportion of the feature value in the total dataset is calculated and then it is multiplied by the total number of the current class label. 

**Solved Example:** Consider the following table:

![](https://media.geeksforgeeks.org/wp-content/uploads/20190717115442/data5.png) 

Here the output variable is the column named “PlayTennis” which determines whether tennis was played on the given day given the weather conditions. The contingency table for the feature “Outlook” is constructed as below:- ![](https://media.geeksforgeeks.org/wp-content/uploads/20190717120148/outlook.png) 

> **Note**: Expected value for each cell is given inside the parenthesis. The expected value for the cell (Sunny,Yes) is calculated as ![\frac{5}{14}\times 9 = 3.21  ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-ae40c20501780867a203cb50af39bdfb_l3.png "Rendered by QuickLaTeX.com")and similarly for others. The ![\chi ^{2}_{outlook}  ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-f73fd7de18d5b50a3c50b2d152fc8074_l3.png "Rendered by QuickLaTeX.com")value is calculated as below:- ![\chi ^{2}_{outlook} = \frac{(2-3.21)^{2}}{3.21}+\frac{(3-1.79)^{2}}{1.79}+\frac{(4-2.57)^{2}}{2.57}+\frac{(0-1.43)^{2}}{1.43}+\frac{(3-3.21)^{2}}{3.21}+\frac{(2-1.79)^{2}}{1.79}  ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-1957eca483b2180e894cc3f11dbb709b_l3.png "Rendered by QuickLaTeX.com")\[Tex\]\\Rightarrow \\chi ^{2}\_{outlook} = 3.129  \[/Tex\]

The contingency table for the feature “Wind” is constructed as below:

![](https://media.geeksforgeeks.org/wp-content/uploads/20190717120150/wind1.png) 

The ![\chi ^{2}_{wind}  ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-c1f07634e83fed82f7fab7011b41241e_l3.png "Rendered by QuickLaTeX.com")value is calculated as below:- ![\chi ^{2}_{wind} = \frac{(3-3.86)^{2}}{3.86}+\frac{(3-1.14)^{2}}{1.14}+\frac{(6-5.14)^{2}}{5.14}+\frac{(2-2.86)^{2}}{2.86}  ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-041ea90077a1f1b457076188ee35dafa_l3.png "Rendered by QuickLaTeX.com")\[Tex\]\\Rightarrow \\chi ^{2}\_{wind} = 3.629  \[/Tex\]On comparing the two scores, we can conclude that the feature “Wind” is more important to determine the output than the feature “Outlook”. [This article](https://www.geeksforgeeks.org/ml-chi-square-test-for-feature-selection/) demonstrates how to do feature selection using Chi-Square Test.

The chi-square test is a statistical method that can be used for feature selection in machine learning. It is used to determine whether there is a significant association between two categorical variables. In the context of feature selection, the chi-square test can be used to identify the features that are most strongly associated with the target variable.

Mathematically, the chi-square test involves calculating the chi-square statistic, which is a measure of the difference between the observed frequency of each category and the expected frequency under the null hypothesis of no association between the variables.

### The chi-square statistic is calculated as follows:

```
χ² = Σ((O - E)² / E)
```


where:

χ² is the chi-square statistic  
O is the observed frequency of each category  
E is the expected frequency of each category, which is calculated under the assumption of no association between the variables  
The expected frequency for each category is calculated as follows:

E = (row total x column total) / grand total

where:

row total is the total number of observations in the row  
column total is the total number of observations in the column  
grand total is the total number of observations in the entire dataset  
Once the chi-square statistic has been calculated for each feature, the p-value can be calculated using the chi-square distribution with (number of categories – 1) degrees of freedom. The p-value represents the probability of observing a chi-square statistic as extreme as the one calculated, assuming that there is no association between the variables.

Features with low p-values are considered to be more strongly associated with the target variable and are selected for further analysis or modeling.

In summary, the chi-square test is a statistical method that can be used for feature selection by measuring the association between categorical variables. The test involves calculating the chi-square statistic and p-value and selecting features with low p-values as being more strongly associated with the target variable.

### Advantages of using the chi-square test for feature selection include:

1.  Simple and easy to use: The chi-square test is a simple and widely-used statistical method that can be easily applied for feature selection in machine learning.
2.  Computationally efficient: The chi-square test is computationally efficient and can be applied to large datasets with many features.
