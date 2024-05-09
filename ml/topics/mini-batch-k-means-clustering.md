# Mini Batch K-means clustering algorithm
**K-means** is among the most well-known clustering algorithms due to its speed performance. With the increase in the volume of data being analysed and the computational time of K-means grows due to its limitation of storing the entire dataset to be stored in memory. This is why a variety of approaches have been proposed to decrease the temporal and spatial costs of the method. Another method that can be used is the **Mini Batch K-means algorithm**.

The main idea of Mini Batch K-means algorithm is to utilize small random samples of fixed in size data, which allows them to be saved in memory. Every time a new random sample of the dataset is taken and used to update clusters; the process is repeated until convergence. Each mini-batch updates the clusters with an approximate combination of the prototypes and the data results, using the learning rate, which reduces with the number of iterations. This rate of learning is the reverse of the number of data assigned to the cluster as it goes through the process. When the number of repetitions increases and the impact of adding new data decreases, convergence is observed when no changes to the clusters happen in consecutive iterations. The research suggests that the algorithm could result in significant savings in computational time but at the cost of a decrease in the quality of the clusters, but not an extensive analysis of the method has yet been conducted to determine how the specific characteristics of the data like the size of the clusters, or its size, impact the quality of the partition.

![Mini Batch K-means clustering algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/mini-batch-k-means-clustering-algorithm.png)

Each batch of data is assigned to clusters based on the prior locations of the cluster's centroids. The algorithm uses small, random portions of the data each time. Then, it updates the positions of the cluster's centroids based on the updated points from the batch. The update is one of gradient descent updates that is much quicker than a standard batch K-Means update.

### Algorithm:

The following is an algorithm used for the Mini K-means batch.

![Mini Batch K-means clustering algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/mini-batch-k-means-clustering-algorithm2.png)

**Explanation:**

Mini Batch K-means is a method that updates clusters using tiny random samples rather than the complete dataset. The algorithm works in the following way:

1.  Initialize cluster centroids randomly.
2.  Take a random sample (mini-batch) from the dataset with a predetermined size.
3.  Using the previous positions of the centroids, assign data points in the mini-batch to the closest ones.
4.  A quicker gradient descent-like update is used to update the centroids' locations depending on the points in the mini-batch.
5.  Repeat steps 2 to 4 until convergence is achieved.

### Implementation:

Python implementation of the above algorithm using scikit-learn library:

**Code:**

A mini-batch K-means is quicker but produces somewhat different outcomes than usual batch K-means.

In this case, we group a set of data first using K-means, then using mini-batch K-means. We then display the results. We also plot points with different labels in these two methods.

![Mini Batch K-means clustering algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/mini-batch-k-means-clustering-algorithm3.png)

As the number of clusters and the amount of data grows, computation time also rises. The time savings in computation is evident only at times when the number of clusters is massive. The impact of the size of the batch on the time to compute is more apparent when the number of clusters is higher. It is possible to conclude that the increase in the number of clusters reduces the degree of similarity between the Mini-Batch K-Means algorithm with the solution K-means. At the same time, the agreement between partitions decreases when the number of clusters increases. However, the objective function does not diminish in the same way. This means that all final partitions will differ; however, they are closer in quality.

Comparison with K-means:
------------------------

*   Because it uses less data samples, Mini Batch K-means is quicker than regular K-means.
*   Although the results of the clustering may be slightly different from those of K-means, the trade-off is between effectiveness and clustering quality.
*   Massive cluster arrangements benefit the most from the time savings, which increase with the size of the clusters.
*   With larger clusters, the effect of batch size on computing time and clustering quality is increasingly pronounced.
*   Despite a possible modest decline in clustering quality when compared to K-means, overall agreement with the objective function is still rather good.

Real-world Use Cases:
---------------------

Mini Batch K-means is used in a variety of fields, such as:

*   **Image segmentation:** pixel clustering for object detection and identification in photos.
*   **Document clustering:** for the purpose of organising and analysing massive text corpora, grouping comparable documents.
*   **Customer segmentation:** dividing up clients into various groups to provide targeted advertising and individualised care.

Given its ability to effectively manage big data, it is especially well suited for scenarios requiring large-scale datasets. The algorithm's quickness and scalability enable rapid responses to changing data streams for real-time clustering jobs.

Tips for Parameter Tuning:
--------------------------

Consider adjusting parameters like the number of clusters, learning rate, and maximum number of iterations to maximise Mini Batch K-means performance. Try out several parameters to determine which one works best for your particular dataset and needs.

Conclusion:
-----------

Mini Batch K-means clustering algorithm offers a pragmatic answer for the computational difficulties presented by customary K-means for huge datasets. While it might bring about marginally unique clustering results, its proficiency makes it significant for large information applications. By understanding the compromises among productivity and clustering quality, specialists can bridle the force of Mini Batch K-means to perform adaptable and powerful information clustering.

* * *