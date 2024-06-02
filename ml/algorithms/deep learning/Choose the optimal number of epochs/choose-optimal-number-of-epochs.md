# Choose Optimal Number of Epochs to Train a Neural Network in Keras

One of the critical issues while training a neural network on the sample data is ****Overfitting****. When the number of epochs used to train a neural network model is more than necessary, the training model learns patterns that are specific to sample data to a great extent. This makes the model incapable to perform well on a new dataset. This model gives high accuracy on the training set (sample data) but fails to achieve good accuracy on the test set. In other words, the model loses generalization capacity by overfitting the training data. To mitigate overfitting and increase the generalization capacity of the neural network, the model should be trained for an optimal number of epochs. A part of the training data is dedicated to the validation of the model, to check the performance of the model after each epoch of training. Loss and accuracy on the training set as well as on the validation set are monitored to look over the epoch number after which the model starts overfitting.

keras.callbacks.callbacks.EarlyStopping()
-----------------------------------------

Either loss/accuracy values can be monitored by the [Early stopping](https://www.geeksforgeeks.org/regularization-by-early-stopping/) call back function. If the loss is being monitored, training comes to a halt when there is an increment observed in loss values. Or, If accuracy is being monitored, training comes to a halt when there is a decrement observed in accuracy values.

> ****Syntax:****
> 
> __keras.callbacks.EarlyStopping(monitor=’val\_loss’, min\_delta=0, patience=0, verbose=0, mode=’auto’, baseline=None, restore\_best\_weights=False)__
> 
> ****where****,
> 
> *   ****monitor:**** The value to be monitored by the function should be assigned. It can be validation loss or validation accuracy.
> *   ****mode:**** It is the mode in which change in the quantity monitored should be observed. This can be ‘min’ or ‘max’ or ‘auto’. When the monitored value is loss, its value is ‘min’. When the monitored value is accuracy, its value is ‘max’. When the mode is set is ‘auto’, the function automatically monitors with the suitable mode.
> *   ****min\_delta:**** The minimum value should be set for the change to be considered i.e., Change in the value being monitored should be higher than ‘min\_delta’ value.
> *   ****patience:**** Patience is the number of epochs for the training to be continued after the first halt. The model waits for patience number of epochs for any improvement in the model.
> *   ****verbose:**** Verbose is an integer value-0, 1 or 2. This value is to select the way in which the progress is displayed while training.
>     *   Verbose = 0: Silent mode-Nothing is displayed in this mode.
>     *   Verbose = 1: A bar depicting the progress of training is displayed.
>     *   Verbose = 2: In this mode, one line per epoch, showing the progress of training per epoch is displayed.
> *   ****restore\_best\_weights:**** This is a boolean value. True value restores the weights which are optimal.

### Importing Libraries and Dataset

[****Python****](https://www.geeksforgeeks.org/python-programming-language/) libraries make it very easy for us to handle the data and perform typical and complex tasks with a single line of code.

*   [****Pandas****](https://www.geeksforgeeks.org/python-pandas-dataframe/) – This library helps to load the data frame in a 2D array format and has multiple functions to perform analysis tasks in one go.
*   [****Numpy****](https://www.geeksforgeeks.org/python-numpy/) – Numpy arrays are very fast and can perform large computations in a very short time.
*   [****Matplotlib****](https://www.geeksforgeeks.org/matplotlib-tutorial/) [](https://www.geeksforgeeks.org/introduction-to-seaborn-python/)– This library is used to draw visualizations.
*   Sklearn – This module contains multiple libraries having pre-implemented functions to perform tasks from data preprocessing to model development and evaluation.
*   [****OpenCV****](https://www.geeksforgeeks.org/opencv-python-tutorial/) – This is an open-source library mainly focused on image processing and handling.
*   [****TensorFlow****](https://www.geeksforgeeks.org/introduction-to-tensorflow/) – This is an open-source library that is used for Machine Learning and Artificial intelligence and provides a range of functions to achieve complex functionalities with single lines of code.

Python3`   ```
import keras
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist

# Loading data
(train_images, train_labels),\
    (test_images, test_labels) = mnist.load_data()

# Reshaping data-Adding number of
# channels as 1 (Grayscale images)
train_images = train_images.reshape((train_images.shape[0],
                                     train_images.shape[1],
                                     train_images.shape[2], 1))

test_images = test_images.reshape((test_images.shape[0],
                                   test_images.shape[1],
                                   test_images.shape[2], 1))

# Scaling down pixel values
train_images = train_images.astype('float32')/255
test_images = test_images.astype('float32')/255

# Encoding labels to a binary class matrix
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

```
     `

From this step onward we will use the TensorFlow library to build our [****CNN****](https://www.geeksforgeeks.org/convolutional-neural-network-cnn-in-machine-learning/) model. Keras framework of the tensor flow library contains all the functionalities that one may need to define the architecture of a Convolutional Neural Network and train it on the data.

### Model Architecture

We will implement a [****Sequential model****](https://www.geeksforgeeks.org/how-to-create-models-in-keras/) which will contain the following parts:

*   Three Convolutional Layers followed by [****MaxPooling****](https://www.geeksforgeeks.org/cnn-introduction-to-pooling-layer/) Layers.
*   The Flatten layer flattens the output of the convolutional layer.
*   Then we will have two fully connected layers followed by the output of the flattened layer.
*   The final layer is the output layer which outputs soft probabilities for the three classes. 

Python3`   ```
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu",
                        input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

model.summary()

```
     `

****Output:****

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 32)        320       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 13, 13, 32)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 11, 11, 64)        18496     
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 5, 5, 64)         0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 1600)              0         
                                                                 
 dense (Dense)               (None, 64)                102464    
                                                                 
 dense_1 (Dense)             (None, 10)                650       
                                                                 
=================================================================
Total params: 121,930
Trainable params: 121,930
Non-trainable params: 0
_________________________________________________________________
```


### Model Compilation

While compiling a model we provide these three essential parameters:

*   [****optimizer****](https://www.geeksforgeeks.org/intuition-of-adam-optimizer/) – This is the method that helps to optimize the cost function by using gradient descent.
*   [****loss****](https://www.geeksforgeeks.org/ml-common-loss-functions/) – The loss function by which we monitor whether the model is improving with training or not.
*   [****metrics****](https://www.geeksforgeeks.org/metrics-for-machine-learning-model/) – This helps to evaluate the model by predicting the training and the validation data.

Python3`   ```
model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=['accuracy'])

```
     `

### Data Preprocessing

While training a machine learning model it is considered a good practice to split the data into training and the validation part this helps us visualize the performance of the model epoch by epoch as the training process moves forward.

Python3`   ```
val_images = train_images[:10000]
partial_images = train_images[10000:]
val_labels = y_train[:10000]
partial_labels = y_train[10000:]

```
     `

### ****Early Stopping Callback****

If model performance is not improving then training will be stopped by [****EarlyStopping****](https://www.geeksforgeeks.org/tensorflow-js-tf-callbacks-earlystopping-function/). We can also define some custom callbacks to stop training in between if the desired results have been obtained early.

Python3`   ```
from keras import callbacks
earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                        mode="min",
                                        patience=5,
                                        restore_best_weights=True)

history = model.fit(partial_images, partial_labels,
                    batch_size=128,
                    epochs=25,
                    validation_data=(val_images, val_labels),
                    callbacks=[earlystopping])

```
     `

****Output:****

```
Epoch 10/25
391/391 [==============================] - 13s 33ms/step - loss: 0.0082 - accuracy: 0.9976 
- val_loss: 0.0464 - val_accuracy: 0.9893
Epoch 11/25
391/391 [==============================] - 12s 31ms/step - loss: 0.0064 - accuracy: 0.9981 
- val_loss: 0.0487 - val_accuracy: 0.9905
Epoch 12/25
391/391 [==============================] - 14s 35ms/step - loss: 0.0062 - accuracy: 0.9982 
- val_loss: 0.0454 - val_accuracy: 0.9885
Epoch 13/25
391/391 [==============================] - 13s 32ms/step - loss: 0.0046 - accuracy: 0.9986 
- val_loss: 0.0502 - val_accuracy: 0.9894
Epoch 14/25
391/391 [==============================] - 15s 38ms/step - loss: 0.0039 - accuracy: 0.9987 
- val_loss: 0.0511 - val_accuracy: 0.9904
```


****Note:**** Training stopped at the 14th epoch i.e., the model will start overfitting from the 15th epoch. As the number of epochs increases beyond 14, training set loss decreases and becomes nearly zero. Whereas, validation loss increases depicting the overfitting of the model on training data.

Let’s visualize the training and validation accuracy with each epoch.

Python3`   ```
import pandas as pd
import matplotlib.pyplot as plt
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
plt.show()

```
     `

****Output:****

![Comparison between Accuracy and Validation Accuracy Epoch-By-Epoch](https://media.geeksforgeeks.org/wp-content/uploads/20230512164542/error.png)

Comparison between Accuracy and Validation Accuracy Epoch-By-Epoch

