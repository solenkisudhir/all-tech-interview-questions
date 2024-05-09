# Epoch: What is an Epoch, timestamp, time
An epoch in machine learning means a complete pass of the training dataset through the algorithm. The number of epochs is an important hyper-parameter for the algorithm. It specifies the number of epochs or complete passes of the entire training dataset that the algorithm undergoes in the training or learning process.

With each epoch, the dataset's internal model parameters are updated. Therefore, the epoch of 1 batch is called the batch gradient descent learning algorithm. Usually, the batch size of an epoch is 1 or more and is always an integer value in the epoch number.

It can also be seen as a 'for-loop with a specified number of epochs in which each loop path traverses the entire training dataset. A for-loop is a nested for-loop that allows the loop to iterate over a specified sample number in a batch when the "batch size" number is specified as one.

Typical values of the number of epochs when the training algorithm can run in thousands of epochs and the process is set to continue until the model error is sufficiently low. Usually, tutorials and examples use numbers like 10, 500, 100, 1000, or even bigger.

Line plots can be created for the training process, in which the x-axis is the epoch in machine learning and the y-axis is the skill or model error. This type of line plot is called the learning curve of an algorithm and helps diagnose problems such as learning the training set down, up, or down as appropriate.

Difference between Epoch and Batch
----------------------------------

The model updates when a specific number of samples are processed, known as the batch size of the samples. The number of complete passes of the training dataset is also important and is called the epoch in the machine learning number in the training dataset. The batch size is typically equal to 1 and can be equal to or less than the sample count of the training dataset. The epoch in a neural network or epoch number is usually an integer value between 1 and infinity. Thus one can run the algorithm for any period. To prevent the algorithm from running, one can use a fixed epoch number and factor in the model error rate of change over time.

In machine learning algorithms, both batch size and epoch are hyper-parameters containing integers as values to be used by the training model. A learning process does not find these values because they are not intrinsic parameters of the model and must be specified for the process when training the algorithm on the training dataset. These numbers are also not fixed values and, depending on the algorithm, it may be necessary to try different integer values before finding the most appropriate value for the procedure.

Example
-------

Consider this example from an era in machine learning. Suppose one uses a dataset with 200 samples (where samples mean data rows) with 1,000 epochs and 5 batch sizes to define epoch-making. The dataset then contains 5 samples in each of the 40 batches, with the model weights being updated when every batch of 5 samples has passed. Also, in this case, machine learning consists of 40 batches in one epoch, which means that the model will be updated 40 times.

Furthermore, since the epoch count is 1,000, the entire dataset passes by the model, and the model itself passes through 1.000 runs. When the model has 40 batches or updates, it means that there are 40,000 batches in the training dataset used in the process of training the algorithm on this dataset!

Conclusion
----------

In exploring the differences in stochastic gradient descent in an era in machine learning and batches, anyone can say that the gradient descent stochastic algorithm uses a dataset for training with its learning algorithm that iterates while updating the model.

The batch size is a gradient descent hyperparameter that measures the number of training samples to be trained before updating the model's internal parameters to work through the batch. Again, the epoch number is a gradient descent hyperparameter that defines the number of complete passes while passing through the dataset under training.

* * *