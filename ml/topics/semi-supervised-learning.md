# Introduction to Semi-Supervised Learning
**_Semi-Supervised learning is a type of Machine Learning algorithm that represents the intermediate ground between Supervised and Unsupervised learning algorithms. It uses the combination of labeled and unlabeled datasets during the training period._**

![Introduction to Semi-Supervised Learning](https://static.javatpoint.com/tutorial/machine-learning/images/semi-supervised-learning.png)

Before understanding the Semi-Supervised learning, you should know the main categories of [Machine Learning](https://www.javatpoint.com/machine-learning) algorithms. Machine Learning consists of three main categories: **[Supervised Learning](https://www.javatpoint.com/supervised-machine-learning), [Unsupervised Learning](https://www.javatpoint.com/unsupervised-machine-learning), and [Reinforcement Learning](https://www.javatpoint.com/reinforcement-learning).** Further, the basic difference between Supervised and unsupervised learning is that _supervised learning datasets consist of an output label training data associated with each tuple,_ and _unsupervised datasets do not consist the same. **Semi-supervised learning is an important category that lies between the Supervised and Unsupervised machine learning.**_ Although Semi-supervised learning is the middle ground between supervised and unsupervised learning and operates on the data that consists of a few labels, it mostly consists of unlabeled data. As labels are costly, but for the corporate purpose, it may have few labels.

The basic disadvantage of supervised learning is that it requires hand-labeling by ML specialists or data scientists, and it also requires a high cost to process. Further unsupervised learning also has a limited spectrum for its applications. **To overcome these drawbacks of supervised learning and unsupervised learning algorithms, the concept of Semi-supervised learning is introduced**. In this algorithm, training data is a combination of both labeled and unlabeled data. However, labeled data exists with a very small amount while it consists of a huge amount of unlabeled data. Initially, similar data is clustered along with an unsupervised learning algorithm, and further, it helps to label the unlabeled data into labeled data. It is why label data is a comparatively, more expensive acquisition than unlabeled data.

We can imagine these algorithms with an example. Supervised learning is where a student is under the supervision of an instructor at home and college. Further, if that student is self-analyzing the same concept without any help from the instructor, it comes under unsupervised learning. Under semi-supervised learning, the student has to revise itself after analyzing the same concept under the guidance of an instructor at college.

Assumptions followed by Semi-Supervised Learning
------------------------------------------------

To work with the unlabeled dataset, there must be a relationship between the objects. To understand this, semi-supervised learning uses any of the following assumptions:

*   **Continuity Assumption:**  
    As per the continuity assumption, the objects near each other tend to share the same group or label. This assumption is also used in supervised learning, and the datasets are separated by the decision boundaries. But in semi-supervised, the decision boundaries are added with the smoothness assumption in low-density boundaries.
*   **Cluster assumptions-** In this assumption, data are divided into different discrete clusters. Further, the points in the same cluster share the output label.
*   **Manifold assumptions-** This assumption helps to use distances and densities, and this data lie on a manifold of fewer dimensions than input space.
*   The dimensional data are created by a process that has less degree of freedom and may be hard to model directly. **(This assumption becomes practical if high).**

Working of Semi-Supervised Learning
-----------------------------------

Semi-supervised learning uses pseudo labeling to train the model with less labeled training data than supervised learning. The process can combine various neural network models and training ways. The whole working of semi-supervised learning is explained in the below points:

*   Firstly, it trains the model with less amount of training data similar to the supervised learning models. The training continues until the model gives accurate results.
*   The algorithms use the unlabeled dataset with pseudo labels in the next step, and now the result may not be accurate.
*   Now, the labels from labeled training data and pseudo labels data are linked together.
*   The input data in labeled training data and unlabeled training data are also linked.
*   In the end, again train the model with the new combined input as did in the first step. It will reduce errors and improve the accuracy of the model.

### Difference between Semi-supervised and Reinforcement Learning.

Reinforcement learning is different from semi-supervised learning, as it works with rewards and feedback. **_Reinforcement learning aims to maximize the rewards by their hit and trial actions, whereas in semi-supervised learning, we train the model with a less labeled dataset._**

Real-world applications of Semi-supervised Learning-
----------------------------------------------------

Semi-supervised learning models are becoming more popular in the industries. Some of the main applications are as follows.

*   **Speech Analysis-** It is the most classic example of semi-supervised learning applications. Since, labeling the audio data is the most impassable task that requires many human resources, this problem can be naturally overcome with the help of applying SSL in a Semi-supervised learning model.
*   **Web content classification-** However, this is very critical and impossible to label each page on the internet because it needs mode human intervention. Still, this problem can be reduced through Semi-Supervised learning algorithms.  
    Further, Google also uses semi-supervised learning algorithms to rank a webpage for a given query.
*   **Protein sequence classification-** DNA strands are larger, they require active human intervention. So, the rise of the Semi-supervised model has been proximate in this field.
*   **Text document classifier-** As we know, it would be very unfeasible to find a large amount of labeled text data, so semi-supervised learning is an ideal model to overcome this.

* * *