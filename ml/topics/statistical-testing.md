# Commonly Used Statistical Tests in Data Science | by Nathan Rosidi | Medium
[

![Nathan Rosidi](https://miro.medium.com/v2/resize:fill:88:88/1*n5OvyvEWJV2w0LP_4vOaPg.png)



](https://nathanrosidi.medium.com/?source=post_page-----93787568eb36--------------------------------)

_A Comprehensive Guide to Essential Statistical Tests and Their Applications_

Photo by [Campaign Creators](https://unsplash.com/@campaign_creators?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com/?utm_source=medium&utm_medium=referral)

In the digital era, making informed choices requires the nuanced skill of analyzing statistics and trends.

This article describes the statistical tests widely employed in data science to help make these informed judgments.

Each statistical test, such as the T-test and Chi-square test, is explained, calculated, and implemented in Python, and project recommendations are included. Let us start with the T-test.

If you want to know more about statistical tests check this one [basic types of statistical tests in data science](https://www.stratascratch.com/blog/basic-types-of-statistical-tests-in-data-science/?utm_source=blog&utm_medium=click&utm_campaign=medium+common+statistical+tests).

Image by author

T-Test
------

A t-test is applied to ascertain whether the average difference among two groups differs significantly from each other. This emanates from the t-distribution applied in making statistical decisions.

There are mainly three types of T-tests.

*   **Independent samples T-test:** The T-test looks at averages in two groups that aren’t connected.
*   **Paired sample T-test**: Compares means from the same group at different times.
*   **One-sample T-test**: Compares the mean of a particular group to a known mean.

Calculation
-----------

Simply put, the T-test estimates the difference between two groups by dividing it by the data’s variability. Let’s see the formula.

The sample means are x1 and x2, the variance is s2, and the sample sizes are n1 and n2.

Simple Python Implementation
----------------------------

Let’s see a simple implementation of it.

```
import numpy as np
from scipy import stats
# Sample data: Group A and Group B
group_a = np.random.normal(5.0, 1.5, 30)
group_b = np.random.normal(6.0, 1.5, 30)
# Performing an Independent T-test
t_stat, p_val = stats.ttest_ind(group_a, group_b)
print(f"T-Statistic: {t_stat}, P-Value: {p_val}")
```


The Python code’s output for the T-test is:

*   **T-Statistic** -3.06
*   **P-Value** 0.003

This P-value (smaller than 0.05), shows a statistically significant difference in means between the two groups. The negative T-statistic shows that group A has a lower mean than group B.

Further Project Suggestion
--------------------------

**Effectiveness of Sleep Aids**: Compare the average sleep duration of subjects taking a new herbal sleep aid versus a placebo.

**Educational Methods**: Evaluate students’ test scores using traditional methods against those taught via e-learning platforms.

**Fitness Program Results**: Assess the impact of two different 8-week fitness programs on weight loss among similar demographic groups.

**Productivity Software:** Compare the average task completion time for two groups using different productivity apps.

**Food Preference Study**: Measure and compare the perceived taste rating of a new beverage with a standard competitor’s product across a sample of consumers.

Chi-Square Test
---------------

The Chi-Square Test determines whether there is a strong association between the two data types or not. There are two types of Chi-square tests.

*   **Chi-Square Test of Independence**: The aim is to find out whether two category variables are independent or not.
*   **Chi-Square Goodness of Fit Test**: In this one, the aim is to find out whether a sample distribution matches a population distribution or not.

Calculation
-----------

The formula for the Chi-Square statistic is:

‘Oi’ is the number we see and ‘Ei’ is the number we expect.

Simply, it involves calculating a value that summarizes the difference between observed and expected frequencies. The larger this value, the more likely the observed differences are not due to random chance.

Simple Python Implementation
----------------------------

Let’s see a simple implementation of it.

```
from scipy.stats import chi2_contingency
import numpy as np
# Example data: Gender vs. Movie Preference
data = np.array([[30, 10], [5, 25]])
chi2, p, dof, expected = chi2_contingency(data)
print(f"Chi2 Statistic: {chi2}, P-value: {p}")
```


The Python code’s output for the Chi-Square Test is

*   Chi-Square Statistic: 21.06
*   P-Value: 0.00000446

The Chi-Square statistic is 21.06 with a P-value of approximately 0.00000446. This very low P-value suggests a significant association between gender and movie preference at a 5% significance level.

Further Project Suggestion
--------------------------

**Election Prediction:** Examine the relationship between voter age groups and their preferences for particular political topics.

**Marketing Campaign:** Determine if there is a difference in responses to two distinct marketing campaigns across geographies.

**Education Level and Technologies Use:** Explore the link between educational level and adopting new technologies in a community.

**illness Outbreak:** Explore the relationship between illness spread and population density in the most affected areas.

**Customer Satisfaction:** Find out the relationship between customer satisfaction and the time of day they get service in retail.

ANOVA (Analysis of Variance)
----------------------------

ANOVA is used to assess averages between three or more groups.. It helps determine if at least one group’s mean is statistically different.

*   **One-Way ANOVA:** Compares means across one independent variable with three or more levels (groups).
*   **Two-Way ANOVA**: Compares means considering two independent variables.
*   **Repeated Measures ANOVA:** Used when the same subjects are used in all groups.

Calculation
-----------

The formula for ANOVA is:

In simpler terms, ANOVA calculates an F-statistic, a ratio of the variance between groups to the variance within groups. A higher F-value indicates a more significant difference between the group means.

Simple Python Implementation
----------------------------

Let’s see a simple implementation of it.

```
from scipy import stats
import numpy as np
# Sample data: Three different groups
group1 = np.random.normal(5.0, 1.5, 30)
group2 = np.random.normal(6.0, 1.5, 30)
group3 = np.random.normal(7.0, 1.5, 30)
# Performing One-Way ANOVA
f_stat, p_val = stats.f_oneway(group1, group2, group3)
print(f"F-Statistic: {f_stat}, P-Value: {p_val}")
```


The Python code’s output for the ANOVA test is:

*   F-Statistic: 15.86
*   P-Value: 0.00000134

The F-statistic is 15.86 with a P-value of approximately 0.00000134. This extremely low P-value indicates a significant difference between the means of at least one of the groups compared to the others at a 5% significance level.

Further Project Suggestion
--------------------------

**Agricultural Crop Yields:** Compare the average yields of various wheat types across numerous areas to determine which are the most productive.

**Employees Productivity:** Compare staff productivity across firm departments to evaluate whether there is a substantial variation.

**Therapeutic strategies:** Evaluate the effectiveness of various therapy techniques to reduce anxiety levels, for instance.

**Gaming Platforms:** Determine whether meaningful differences in average frame rates exist over many gaming systems running the same video game.

**Dietary Effects on Health:** Investigate the influence of different diets, like vegan, and vegetarian on specific health indicators in participants.

Pearson Correlation
-------------------

Pearson Correlation evaluates the straight-line connection between two ongoing variables. It produces a value between -1 and 1, indicating the strength and direction of the association.

The Pearson Correlation is a specific type of correlation, mainly differing from others like Spearman’s correlation, used for non-linear relationships.

Calculation
-----------

The formula for the Pearson Correlation coefficient is:

Simply put, it calculates how much one variable changes with another.

A value close to 1 indicates a strong positive correlation, and close to -1 indicates a strong negative correlation.

Simple Python Implementation
----------------------------

Let’s see a simple implementation of it.

```
import numpy as np
from scipy.stats import pearsonr
# Sample data
x = np.array([10, 20, 30, 40, 50])
y = np.array([15, 25, 35, 45, 55])
# Calculating Pearson Correlation
corr, _ = pearsonr(x, y)
print(f"Pearson Correlation Coefficient: {corr}")
```


The Python code’s output for the Pearson Correlation test is:

*   Pearson Correlation Coefficient: 1.0

This shows a perfect positive relationship between these variables. If you are doing an ML project with this correlation, you have to suspect overfitting.

Further Project Suggestion
--------------------------

**Economic Indicators:** Investigate the link between consumer confidence and retail sales volume.

**Healthcare Analysis**: In this one, you can explore the link between the number of hours spent physically active and blood pressure levels.

**Educational Achievement:** Would not be super nice, if you had a chance to examine the link between the amount of time you spent on your homework and your success.

**Technology Use:** Maybe it is time to lower your screen time, to find out if you can investigate the relationship between time spent on social media and perceived stress or happiness.

**Real Estate Pricing:** Research the link between social media use and estimated stress or happiness levels.

Mann-Whitney U Test
-------------------

The Mann-Whitney U Test is a test that evaluates differences between two independent groups if the data you have does not fit a normal distribution.

It is a substitute for the T-test when data doesn’t adhere to the normality assumption.

Calculation
-----------

The Mann-Whitney U statistic is calculated based on the ranks of the data in the combined dataset.

Where

*   U is the Mann-Whitney U statistic.
*   R1 and R2 are the sum of ranks for the first and second groups, respectively.
*   n1 and n2 are the sample sizes of the two groups

Simple Python Implementation
----------------------------

Let’s see a simple implementation of it.

```
from scipy.stats import mannwhitneyu
import numpy as np
# Sample data: Two groups
group1 = np.random.normal(5.0, 1.5, 30)
group2 = np.random.normal(6.0, 1.5, 30)
# Performing Mann-Whitney U Test
u_stat, p_val = mannwhitneyu(group1, group2)
print(f"U Statistic: {u_stat}, P-Value: {p_val}")
```


The Python code’s output for the Mann-Whitney U Test is:

*   U Statistic: 305.0
*   P-Value: 0.032

This P-value is below the typical alpha level of 0.05, indicating that there is a statistically significant difference in the median ranks of the two groups at the 5% significance level. The Mann-Whitney U Test result suggests that the distributions of the two groups are not equal.

Further Project Suggestion
--------------------------

**Medication Response:** Compare the change in symptom severity before and after taking two distinct drugs in non-normally distributed patient data.

**Job Satisfaction:** It can be a good time to switch departments. To decide where to go, you can Compare the levels of job satisfaction among employees in your company’s high- and low-stress departments.

**Teaching Materials:** Determine the impact of two teaching materials on student involvement in a classroom where data is not typically distributed.

**E-commerce Delivery timeframes:** Compare the delivery timeframes of two courier services for e-commerce packages.

**Exercise Impact on Mood:** Investigate the impact of two distinct forms of short-term exercise on mood improvement in individuals, with a focus on nonparametric data.

Conclusion
----------

We've looked at everything from the T-test to the Mann-Whitney U Test, including Python implementations and real-world project ideas.

Remember that being a skilled data scientist requires practice. Exploring these assessments through hands-on projects strengthens your comprehension and sharpens your analytical abilities.

To do that, visit our platform and do data projects like [Student Performance Analysis](https://platform.stratascratch.com/data-projects/student-performance-analysis?utm_source=blog&utm_medium=click&utm_campaign=medium+common+statistical+tests). Here, you’ll have a chance to do Chi-square tests.



=================================================================================
# Statistics for Machine Learning: A Complete Guide | Simplilearn
Statistics is a core component of data analytics and machine learning. It helps you analyze and [visualize data](https://www.simplilearn.com/data-visualization-article "visualize data") to find unseen patterns. If you are interested in machine learning and want to grow your career in it, then learning statistics along with programming should be the first step. In this article, you will learn all the concepts in statistics for machine learning.

What Is Statistics?
-------------------

Statistics is a branch of mathematics that deals with collecting, analyzing, interpreting, and visualizing empirical data. Descriptive statistics and inferential statistics are the two major areas of statistics. Descriptive statistics are for describing the properties of sample and population data (what has happened). Inferential statistics use those properties to test hypotheses, reach conclusions, and make predictions (what can you expect).

> Looking forward to becoming a Machine Learning Engineer? Check out Simplilearn's [AIML Course](https://www.simplilearn.com/pgp-ai-machine-learning-certification-training-course "AIML Course"), [Machine Learning Course](https://www.simplilearn.com/iitk-professional-certificate-course-ai-machine-learning "Machine Learning Course") and get certified today.

Use of Statistics in Machine Learning
-------------------------------------

![StatisticsUses](https://www.simplilearn.com/ice9/assets/form_opacity.png)

*   Asking questions about the data
*   Cleaning and preprocessing the data
*   Selecting the right features
*   Model evaluation
*   Model prediction

With this basic understanding, itâ€™s time to dive deep into learning all the crucial concepts related to statistics for machine learning.

Population and Sample
---------------------

### Population:

In statistics, the population comprises all observations (data points) about the subject under study.

An example of a population is studying the voters in an election. In the 2019 Lok Sabha elections, nearly 900 million voters were eligible to vote in 543 constituencies.

### Sample:

In statistics, a sample is a subset of the population. It is a small portion of the total observed population.

An example of a sample is analyzing the first-time voters for an opinion poll.

Measures of Central Tendency
----------------------------

Measures of central tendency are the measures that are used to describe the distribution of data using a single value. Mean, Median and Mode are the three measures of central tendency.

### Mean:

The arithmetic mean is the average of all the data points.

If there are n number of observations and xi is the ith observation, then mean is:

![Mean](https://www.simplilearn.com/ice9/assets/form_opacity.png)

Consider the data frame below that has the names of seven employees and their salaries.

![EmployeeDataset](https://www.simplilearn.com/ice9/assets/form_opacity.png)

To find the mean or the average salary of the employees, you can use the mean() functions in Python.

![MeanSalary.](https://www.simplilearn.com/ice9/assets/form_opacity.png)

### Median:

Median is the middle value that divides the data into two equal parts once it sorts the data in ascending order.

If the total number of data points (n) is odd, the median is the value at position (n+1)/2.

When the total number of observations (n) is even, the median is the average value of observations at n/2 and (n+2)/2 positions.

The median() function in Python can help you find the median value of a column. From the above data frame, you can find the median salary as:

![MedianSalary](https://www.simplilearn.com/ice9/assets/form_opacity.png)

### Mode:

The mode is the observation (value) that occurs most frequently in the data set. There can be over one mode in a dataset.

Given below are the heights of students (in cm) in a class:

155, 157, 160, 159, 162, 160, 161, 165, 160, 158

Mode = 160 cm.

The mode salary from the data frame can be calculated as:

![ModeSalary](https://www.simplilearn.com/ice9/assets/form_opacity.png)

Variance and Standard Deviation
-------------------------------

Variance is used to measure the variability in the data from the mean.Â 

![VarianceFormula](https://www.simplilearn.com/ice9/assets/form_opacity.png)

Consider the below dataset.

![EmployeeDataframe](https://www.simplilearn.com/ice9/assets/form_opacity.png)

To calculate the variance of the Grade, use the following:

![VarianceGrade](https://www.simplilearn.com/ice9/assets/form_opacity.png)

Standard deviation in statistics is the square root of the variance. Variance and standard deviation represent the measures of fit, meaning how well the mean represents the data.

![StandardDeviationFormula](https://www.simplilearn.com/ice9/assets/form_opacity.png)

You can find the standard deviation using the std() function in Python.

![stdGrade](https://www.simplilearn.com/ice9/assets/form_opacity.png)

Range and Interquartile Range
-----------------------------

### Range:

The Range in statistics is the difference between the maximum and the minimum value of the dataset.

![Range](https://www.simplilearn.com/ice9/assets/form_opacity.png)

### Interquartile Range (IQR) :

The IQR is a measure of the distance between the 1st quartile (Q1) and 3rd quartile (Q3).

![IQR](https://www.simplilearn.com/ice9/assets/form_opacity.png)

Skewness and Kurtosis
---------------------

### Skewness:

Skewness measures the shape of the distribution. A distribution is symmetrical when the proportion of data at an equal distance from the mean (or median) is equal. If the values extend to the right, it is right-skewed, and if the values extend left, it is left-skewed.

![Skewness](https://www.simplilearn.com/ice9/assets/form_opacity.png)

### Kurtosis:

Kurtosis in statistics is used to check whether the tails of a given distribution have extreme values. It also represents the shape of a probability distribution.

![Skewness-Kurtosis](https://www.simplilearn.com/ice9/assets/form_opacity.png)

![SalarySkewness](https://www.simplilearn.com/ice9/assets/form_opacity.png)

![HoursSkewness](https://www.simplilearn.com/ice9/assets/form_opacity.png)

![GradeSkewness](https://www.simplilearn.com/ice9/assets/form_opacity.png)

Now, itâ€™s time to discuss a very popular distribution in statistics for machine learning, i.e., Gaussian Distribution.

Gaussian Distribution
---------------------

In statistics and probability, Gaussian (normal) distribution is a popular continuous probability distribution for any random variable. It is characterized by 2 parameters (mean Î¼ and standard deviation Ïƒ). Many natural phenomena follow a normal distribution, such as the heights of people and IQ scores.

![GaussianDistribution](https://www.simplilearn.com/ice9/assets/form_opacity.png)

Properties of Gaussian Distribution:

*   The mean, median, and mode are the same
*   It has a symmetrical bell shape
*   68% data lies within 1 standard deviation of the mean
*   95% data lie within 2 standard deviations of the mean
*   99.7% of the data lie within 3 standard deviations of the mean

![GaussianCode.](https://www.simplilearn.com/ice9/assets/form_opacity.png)

![GaussianPlot](https://www.simplilearn.com/ice9/assets/form_opacity.png)

Central Limit Theorem
---------------------

According to the central limit theorem, given a population with mean as Î¼ and standard deviation as Ïƒ, if you take large random samples from the population, then the distribution of the sample means will be roughly normally distributed, irrespective of the original population distribution.

Rule of Thumb: For the central limit theorem to hold true, the sample size should be greater than or equal to 30.

![Clt](https://www.simplilearn.com/ice9/assets/form_opacity.png)

Now, you will learn a very critical concept in statistics for machine learning, i.e., Hypothesis testing.Â Â 

Hypothesis Testing
------------------

Hypothesis testing is a statistical analysis to make decisions using experimental data. It allows you to statistically back up some findings you have made in looking at the data. In hypothesis testing, you make a claim and the claim is usually about population parameters such as mean, median, standard deviation, etc.

*   The assumption made for a statistical test is called the null hypothesis (H0).
*   The Alternative hypothesis (H1) contradicts the null hypothesis stating that the assumptions do not hold true at some level of significance.

Hypothesis testing lets you decide to either reject or retain a null hypothesis.

Example: H0: The average BMI of boys and girls in a class is the same

Â  Â  H1: The average BMI of boys and girls in a class is not the same

To determine whether a finding is statistically significant, you need to interpret the p-value. It is common to compare the p-value to a threshold value called the significance level.

It often sets the level of significance to 5% or 0.05.

If the p-value > 0.05 - Accept the null hypothesis.

If the p-value < 0.05 - Reject the null hypothesis.

Some popular hypothesis tests are:

*   Chi-square test
*   T-test
*   Z-test
*   Analysis of Variance (ANOVA)

Conclusion
----------

Statistics is a core component of machine learning. It helps you draw meaningful conclusions by analyzing raw data. In this article on Statistics for Machine Learning, you covered all the critical concepts that are widely used to make sense of data.Â 

If you are looking to learn further about machine learning with the aim of becoming an expert machine learning engineer, Simplilearnâ€™s Machine Learning program in partnership with IIT Kanpur University is the ideal way to go about it. Ranked #1 AI and Machine Learning course by TechGig, this unique AI and Machine Learning Program offers an extremely comprehensive and applied learning curriculum covering the most in-demand tools, skills, and techniques used in machine learning today. You get to perfect your skills with a capstone project in 3 domains, and 25+ projects that use real industry data sets from companies such as Twitter, Amazon, Mercedes etc.

Do you have any questions regarding this article on Statistics for Machine Learning? If you have, then please put them in the comments section. Weâ€™ll help you solve your queries. To learn more about the crucial statistical techniques, click on the following link: [Mathematics for Machine Learning](https://www.youtube.com/watch?v=iyxqcS1u5go&t=1646s "Mathematics for Machine Learning").
