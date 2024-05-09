# Machine Learning with Anomaly Detection
**Anomaly detection is a process of finding those rare items, data points, events, or observations that make suspicions by being different from the rest data points or observations.** Anomaly detection is also known as **outlier detection**.

![Machine Learning with Anomaly Detection](https://static.javatpoint.com/tutorial/machine-learning/images/machine-learning-with-anomaly-detection.png)

Generally, anomalous data is related to some kind of problems such as bank fraud, medical problems, malfunctioning equipment, etc.

Finding an anomaly is the ability to define what is normal? For example, in the below image, the yellow vehicle is an anomaly among all red vehicles.

![Machine Learning with Anomaly Detection](https://static.javatpoint.com/tutorial/machine-learning/images/machine-learning-with-anomaly-detection2.png)

Types of Anomaly Detection
--------------------------

**1\. Point Anomaly**

A tuple within the dataset can be said as a Point anomaly if it is far away from the rest of the data.

**Example**: An example of a point anomaly is a sudden transaction of a huge amount from a credit card.

**2\. Contextual Anomaly**

Contextual anomaly is also known as conditional outliers. If a particular observation is different from other data points, then it is known as a contextual Anomaly. In such types of anomalies, an anomaly in one context may not be an anomaly in another context.

**3\. Collective Anomaly**

Collective anomalies occur when a data point within a set is anomalous for the whole dataset, and such values are known as collective outliers. In such anomalies, specific or individual values are not anomalous as a whole or contextually.

Categories of Anomaly detection techniques
------------------------------------------

Anomaly detection techniques are broadly categorized into three types:

1.  **Supervised Anomaly detection**
2.  **Unsupervised Anomaly detection**

### Supervised Anomaly Detection

Supervised Anomaly detection needs the labeled training data, which contains both normal and anomalous data for creating a predictive model. Some of the common supervised methods are neural networks, support vector machines, k-nearest neighbors, Bayesian networks, decision trees, etc.

K-nearest neighbor is one of the popular nonparametric techniques, which find the approximate distance between different points on the input vector. This is one of the best anomaly detection methods. Another popular model is the Bayesian network, which is used for anomaly detection when combined with statistical schemes. This model encodes a probabilistic relationship among variable interests.

Supervised anomaly detection techniques have different advantages, such as the capability of encoding interdependencies between variables and of predicting events; it also provides the ability to incorporate both prior knowledge and data.

### Unsupervised Anomaly Detection

Unsupervised Anomaly detection does not require labeled training data. These techniques are based on two assumptions, which are,

*   Most of the network connections are from normal traffic, and only a small amount of data is abnormal.
*   Malicious traffic is statistically different from normal traffic.

On the basis of these assumptions, data clusters of similar data points that occur frequently are assumed to be normal traffic, and those data groups that are infrequent are considered abnormal or malicious.

Some of the common unsupervised anomaly detection algorithms are self-organizing maps (SOM), K-means, C-means, expectation-maximization meta-algorithm (EM), adaptive resonance theory (ART), and one-class support vector machines. SOM, or Self-organizing map, is a popular technique that aims to reduce the dimension of data visualization.

Anomaly detection can effectively help in catching the fraud, discovering strange activity in large and complex Big Data sets. This can prove to be useful in areas such as banking security, natural sciences, medicine, and marketing, which are prone to malicious activities. With machine learning, an organization can intensify search and increase the effectiveness of its digital business initiatives.

Need of Anomaly Detection
-------------------------

### 1\. Anomaly detection for application performance

Application performance of any company can either generate or reduce workforce productivity and revenue. General or traditional approaches for monitoring the application performance allow to react to issues, but still business used to suffer, and hence it affects the user. But with the help of anomaly detection using machine learning, it is easy to identify and resolve the application performance issues before they affect the business as well as users.

Anomaly detection using machine learning algorithms can simply correlate data with corresponding application performance metrics and find out the complete knowledge of the issue. There are different industries that also employ anomaly detection techniques for their businesses, such as **Telco, Adtech**, etc.

### 2\. Anomaly detection for product quality

It is not enough for product managers to trust another department for taking care of required monitoring and alerts. It is always required for product managers to be able to trust that product will work smoothly. It is because the product always needs changes, from each version release to new feature upgradation, and generates anomalies. If you don't properly monitor these anomalies, it may cause millions of revenues lost and can also affect the brand reputation.

### 3\. Anomaly detection for user experience

If you release a faulty version, you may experience a DDoS attack, risk of usage lapses across customer experiences. So, it is required to react to such issues before they impact user experience to reduce the chances of revenue loss.

Proactively streamlining and improving user experiences will help improve customer satisfaction in a variety of industries, including Gaming, online business, etc.

Conclusion
----------

In this topic, we have provided a detailed description of anomaly detection and its use cases in business. Anomaly detection is very helpful in different business applications such as Credit Card Fraud detection systems, Intrusion detection, etc.

* * *

Next Topic[#](#)