# Deep Learning Algorithms - The Complete Guide | AI Summer
Deep Learning is eating the world.

The hype began around 2012 when a Neural Network achieved super human performance on Image Recognition tasks and only a few people could predict what was about to happen.

During the past decade, more and more algorithms are coming to life. More and more companies are starting to add them in their daily business.

Here, I tried to cover all the most important Deep Learning algorithms and architectures concieved over the years for use in a variety of applications such as Computer Vision and Natural Language Processing.

Some of them are used more frequently than others and each one has its own streghth and weeknesses.

**My main goal is to give you a general idea of the field and help you understand what algorithm should you use in each specific case**. Because I know it seems chaotic for someone who wants to start from scratch.

But after reading the guide, I am confident that you will be able to recognize what is what and you will be ready to begin using them right away.

So if you are looking for a truly complete guide on Deep Learning , let's get started.

Contents
--------

Deep Learning is gaining [crazy amounts of popularity](https://theaisummer.com/Deep_learning/) in the scientific and corporate communities. Since 2012, the year when a Convolutional Neural Network achieved unprecedent accuracy on an image recognition competition ( ImageNet Large Scale Visual Recognition Challenge), [more and more research papers come out every year](https://www.technologyreview.com/s/612768/we-analyzed-16625-papers-to-figure-out-where-ai-is-headed-next/) and more and more companies started to incorporate Neural Networks into their businesses. It is estimated that Deep Learning is right now a 2.5 Billion Market and expected to become [18.16 Billion by 2023](https://www.marketsandmarkets.com/Market-Reports/deep-learning-market-107369271.html).

But what is Deep learning?
--------------------------

According to [Wikipedia](https://en.wikipedia.org/wiki/Deep_learning): “Deep learning (also known as deep structured learning or differential programming) is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised”.

In my mind, **Deep Learning is a collection of algorithms inspired by the workings of the human brain in processing data and creating patterns for use in decision making, which are expanding and improving on the idea of a single model architecture called Artificial Neural Network**.

Neural Networks
---------------

Just like the human brain, [Neural Networks](http://karpathy.github.io/neuralnets/) consist of Neurons. Each Neuron receives signals as an input, multiplies them by weights, sums them up and applies a non-linear function. These neurons are stacked next to each other and organized in layers.

But what do we accomplish by doing that?

 [![neuron](https://theaisummer.com/static/25ac0e7374f862841fe554de4b7a6450/4b190/neuron.jpg "neuron")](https://theaisummer.com/static/25ac0e7374f862841fe554de4b7a6450/4b190/neuron.jpg)_[Datacamp](https://www.datacamp.com/community/tutorials/deep-learning-python)_ Algorithm It turns out that Neural Networks are excellent **function approximators.**

We can assume that every behavior and every system can ultimately be represented as a mathematical function (sometimes an incredible complex one). If we somehow manage to find that function, we can essentially understand everything about the system. But finding the function can be extremely hard. So, we need to estimate it. Enter Neural Networks.

### Backpropagation

Neural Networks are able to learn the desired function using big amounts of data and an iterative algorithm called [backpropagation](https://brilliant.org/wiki/backpropagation/). We feed the network with data, it produces an output, we compare that output with a desired one (using a loss function) and we readjust the weights based on the difference.

And repeat. And repeat. The adjustment of weights is performed using a non-linear optimization technique called [stochastic gradient descent](https://ruder.io/optimizing-gradient-descent/).

After a while, the network will become really good at producing the output. Hence, the training is over. Hence, we manage to approximate our function. And if we pass an input with an unknown output to the network, it will give us an answer based on the approximated function.

Let’s use an example to make this clearer. Let’s say that for some reason we want to identify images with a tree. We feed the network with any kind of images and it produces an output. Since we know if the image has actually a tree or not, we can compare the output with our truth and adjust the network.

As we pass more and more images, the network will make fewer and fewer mistakes. Now we can feed it with an unknown image, and it will tell us if the image contains a tree. Pretty cool, right?

Over the years researchers came up with amazing improvements on the original idea. Each new architecture was targeted on a specific problem and one achieved better accuracy and speed. We can classify all those new models in specific categories:

Feedforward Neural Networks (FNN)
---------------------------------

Feedforward Neural Networks are usually [fully connected](https://theaisummer.com/Neural_Network_from_scratch/), which means that every neuron in a layer is connected with all the other neurons in the next layers. The described structure is called Multilayer Perceptron and originated back in 1958. Single-layer perceptron can only learn linearly separable patterns, but a multilayer perceptron is able to learn non-linear relationships between the data.

 [![neural-network](https://theaisummer.com/static/f867ca10ec93233d261116352e6bec56/c1b63/neural-network.png "neural-network")](https://theaisummer.com/static/f867ca10ec93233d261116352e6bec56/c9c44/neural-network.png)_[http://www.sci.utah.edu/](http://www.sci.utah.edu/~beiwang/teaching/cs6965-fall-2019/Lecture15-DeepLearningVis.pdf)_

They are exceptionally well on tasks like classification and regression. Contrary to other machine learning algorithms, they don’t converge so easily. The more data they have, the higher their accuracy.

Convolutional Neural Networks (CNN)
-----------------------------------

Convolutional Neural Networks employ a function called [convolution](https://theaisummer.com/Neural_Network_from_scratch_part2/) The concept behind them is that instead of connecting each neuron with all the next ones, we connect it with only a handful of them (the receptive field).

In a way, they try to regularize feedforward networks to avoid overfitting (when the model learns only pre-seen data and can’t generalize), which makes them very good in identifying spatial relationships between the data.

 [![convolutional-neural-network](https://theaisummer.com/static/2fcd242de84286b4c4be7e764b334630/167b5/convolutional-neural-network.png "convolutional-neural-network")](https://theaisummer.com/static/2fcd242de84286b4c4be7e764b334630/167b5/convolutional-neural-network.png)_[Face Recognition Based on Convolutional Neural Network](https://www.researchgate.net/figure/A-traditional-Convolutional-Neural-Networks-design_fig1_322303457)_

That’s why their primary use case is Computer Vision and applications such as image classification, video recognition, medical image analysis and [self-driving cars](https://theaisummer.com/Self_driving_cars/) where they achieve literally superhuman performance.

They are also ideal to combine with other types of models such as Recurrent Networks and Autoencoders. One such example is [Sign Language Recognition](https://theaisummer.com/Sign-Language-Recognition-with-PyTorch/).

Recurrent Neural Networks (RNN)
-------------------------------

Recurrent networks are perfect for time-related data and they are used in time series forecasting. They use some form of feedback, where they return the output back to the input. You can think of it as a loop from the output to the input in order to pass information back to the network. Therefore, they are capable to remember past data and use that information in its prediction.

To achieve better performance researchers have modified the original neuron into more complex structures such as [GRU units](https://www.coursera.org/lecture/nlp-sequence-models/gated-recurrent-unit-gru-agZiL) and [LSTM Units](https://theaisummer.com/Bitcon_prediction_LSTM/). LSTM units have been used extensively in natural language processing in tasks such as language translation, speech generation, text to speech synthesis.

 [![lstm_cell](https://theaisummer.com/static/5564cf7e0b865e2481dd5306d46cb402/faddd/lstm_cll.jpg "lstm_cell")](https://theaisummer.com/static/5564cf7e0b865e2481dd5306d46cb402/faddd/lstm_cll.jpg)_[STFCN: Spatio-Temporal FCN for Semantic Video Segmentation](https://www.researchgate.net/figure/An-example-of-a-basic-LSTM-cell-left-and-a-basic-RNN-cell-right-Figure-follows-a_fig2_306377072)_

Recursive Neural Network
------------------------

Recursive Neural Networks are another form of recurrent networks with the difference that they are structured in a tree-like form. As a result, they can model hierarchical structures in the training dataset.

They are traditionally used in NLP in applications such as Audio to text transcription and sentiment analysis because of their ties to binary trees, contexts, and natural-language-based parsers. However, they tend to be much slower than Recurrent Networks

AutoEncoders
------------

[Autoencoders](https://theaisummer.com/Autoencoder/) are mostly used as an unsupervised algorithm and its main use-case is dimensionality reduction and compression. Their trick is that they try to make the output equal to the input. In other works, they are trying to reconstruct the data.

They consist of an encoder and a decoder. The encoder receives the input and it encodes it in a latent space of a lower dimension. The decoder takes that vector and decodes it back to the original input.

 [![autoencoder](https://theaisummer.com/static/d57a2ae3fb095d80d99d4cca1635362c/e5166/autoencoder.jpg "autoencoder")](https://theaisummer.com/static/d57a2ae3fb095d80d99d4cca1635362c/3d2b2/autoencoder.jpg)_[Autoencoder Neural Networks for Outlier Correction in ECG- Based Biometric Identification](https://www.semanticscholar.org/paper/Autoencoder-Neural-Networks-for-Outlier-Correction-Karpinski-Khoma/ba8cbe2b4b1c03b00d888d95e23fc1de024b2797)_

That way we can extract from the middle of the network a representation of the input with fewer dimensions. Genius, right?

Of course, we can use this idea to reproduce the same but a bit different or even better data (training data augmentation, data denoising, etc)

Deep Belief Networks and Restricted Boltzmann Machines
------------------------------------------------------

[Restricted Boltzmann Machines](https://towardsdatascience.com/restricted-boltzmann-machines-simplified-eab1e5878976) are stochastic neural networks with generative capabilities as they are able to learn a probability distribution over their inputs. Unlike other networks, they consist of only input and hidden layers( no outputs).

In the forward part of the training the take the input and produce a representation of it. In the backward pass they reconstruct the original input from the representation. (Exactly like autoencoders but in a single network).

 [![restricted-boltzmann-machine](https://theaisummer.com/static/0c873b5f7d8d3d46590def6f6d2972da/5aae9/restricted-boltzmann-machine.png "restricted-boltzmann-machine")](https://theaisummer.com/static/0c873b5f7d8d3d46590def6f6d2972da/5aae9/restricted-boltzmann-machine.png)_[Explainable Restricted Boltzmann Machine for Collaborative Filtering](https://medium.com/@tanaykarmarkar/explainable-restricted-boltzmann-machine-for-collaborative-filtering-6f011035352d)_

Multiple RBM can be stacked to form a [Deep Belief Network](http://deeplearning.net/tutorial/DBN.html). They look exactly like Fully Connected layers, but they differ in how they are trained. That’s because they train layers in pairs, following the training process of RBMs (described before)

However, DBNs and RBMs have kind of abandoned by the scientific community in favor of Variational Autoencoders and GANs

Generative Adversarial Networks (GAN)
-------------------------------------

[GANs](https://theaisummer.com/Generative_Artificial_Intelligence/) were introduced in 2016 by Ian Goodfellow and they are based on a simple but elegant idea: You want to generate data, let’s say images. What do you do?

You build two models. You train the first one to generate fake data (generator) and the second one to distinguish real from fakes ones(discriminator). And you put them to compete against each other.

The generator becomes better and better at image generation, as its ultimate goal is to fool the discriminator. The discriminator becomes better and better at distinguish fake from real images, as its goal is to not be fooled. The result is that we now have incredibly realistic fake data from the discriminator.

 [![generative-adversarial-networks](https://theaisummer.com/static/9dc7ade8581705de13978e14f1ff7f33/748b0/generative-adversarial-networks.png "generative-adversarial-networks")](https://theaisummer.com/static/9dc7ade8581705de13978e14f1ff7f33/748b0/generative-adversarial-networks.png)_[O'Reilly](https://www.oreilly.com/content/generative-adversarial-networks-for-beginners/)_

Applications of Generative Adversarial Networks include video games, astronomical images, interior design, fashion. Basically, if you have images in your fields, you can potentially use GANs. Oooh, do you remember Deep Fakes? Yeah, that was all made by GANs.

Transformers
------------

[Transformers](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html) are also very new and they are mostly used in language applications as they are starting to make recurrent networks obsolete. They based on a concept called attention, which is used to force the network to focus on a particular data point.

Instead of having overly complex LSTM units, you use Attention mechanisms to weigh different parts of the input based on their significance. [The attention mechanism](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html) is nothing more than another layer with weights and its sole purpose is to adjust the weights in a way that prioritizes segments of inputs while deprioritizing others.

Transformers, in fact, consist of a number of stacked encoders (form the encoder layer), a number of stacked decoders (the decoder layer) and a bunch of attention layers (self- attentions and encoder-decoder attentions)

 [![transformers](https://theaisummer.com/static/962f8150540dda5a82b13eb77800191d/eb390/transformers.png "transformers")](https://theaisummer.com/static/962f8150540dda5a82b13eb77800191d/eb390/transformers.png)_[http://jalammar.github.io/illustrated-transformer/](http://jalammar.github.io/illustrated-transformer/)_

Transformers are designed to handle ordered sequences of data, such as natural language, for various tasks such as machine translation and text summarization. Nowadays BERT and GPT-2 are the two most prominent pretrained natural language systems, used in a variety of NLP tasks, and they are both based on Transformers.

Graph Neural Networks
---------------------

Unstructured data are not a great fit for Deep Learning in general. And there are many real-world applications where data are unstructured and organized in a graph format. Think social networks, chemical compounds, knowledge graphs, spatial data.

[Graph Neural Networks](https://theaisummer.com/Graph_Neural_Networks/) purpose is to model Graph data, meaning that they identify the relationships between the nodes in a graph and produce a numeric representation of it. Just like an embedding. So, they can later be used in any other machine learning model for all sorts of tasks like clustering, classification, etc.

Deep Learning in Natural Language Processing (NLP)
--------------------------------------------------

### Word Embeddings

Word Embeddings are the representations of words into numeric vectors in a way that capture the semantic and syntactic similarity between them. This is necessary because neural networks can only learn from numeric data so we had to find a way to encode words and text into numbers.

*   [Word2Vec](https://pathmind.com/wiki/word2vec) is the most popular technique and it tries to learn the embeddings by predicting a word based on its context (CBOW**)** or by predicting the surrounding words based on the word (Skip-Gram). Word2Vec is nothing more than a simple neural network with 2 layers that has words as inputs and outputs. Words are fed to the Neural Network in the form of one-hot encoding.
    
    In the case of CBOW, the inputs are the adjacent words and the output is the desired word. In the case of Skip-Gram, it’s the other way around.
    
*   [Glove](https://medium.com/@jonathan_hui/nlp-word-embedding-glove-5e7f523999f6) is another model that extends the idea of Word2Vec by combining it with matrix factorization techniques such as Latent Semantic Analysis, which are proven to be really good as global text statistics but unable to capture local context. So the union of those two gives us the best of both worlds.
    
*   [FastText](https://research.fb.com/blog/2016/08/fasttext/) by Facebook utilizes a different approach by making use of character-level representation instead of words.
    
*   **Contextual Word Embeddings** replace Word2Vec with Recurrent Neural Networks to predict, given a current word in a sentence, the next word. That way we can capture long term dependencies between words and each vector contains both the information on the current word and on the past ones. The most famous version is called [ELMo](https://allennlp.org/elmo) and it consists of a two-layer bi-directional LSTM network.
    
*   [Attention Mechanisms](https://blog.floydhub.com/attention-mechanism/) and Transformers are making RNN’s obsolete (as mentioned before), by weighting the most related words and forgetting the unimportant ones.
    

### Sequence Modeling

Sequence models are an integral part of Natural Language Processing as it appears on lots of common applications such as [Machine Translation](https://www.tensorflow.org/tutorials/text/nmt_with_attention), Speech Recognition, Autocompletion and Sentiment Classification. Sequence models are able to process a sequence of inputs or events such as a document of words.

For example, imagine that you want to translate a sentence from English to French.

To do that you need a [Sequence to Sequence model (seq2sec)](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html). Seq2sec models include an encoder and a decoder. The encoder takes the sequence(sentence in English) and produces an output, a representation of the input in a latent space. This representation is fed to the decoder, which gives us the new sequence (sentence in France).

The most common architectures for encoder and decoder are Recurrent Neural Networks (mostly LSTMs) because they are great in capturing long term dependencies and Transformers that tend to be faster and easier to parallelize. Sometimes they are also combined with Convolutional Networks for better accuracy.

[BERT](https://github.com/google-research/bert) and [GPT-2](https://openai.com/blog/better-language-models/) are considered the two best language models and they are in fact Transformer based Sequence models .

Deep Learning in Computer Vision
--------------------------------

 [![cv_tasks](https://theaisummer.com/static/fd0197a92a28cba1dcc37865514f5b30/c08c5/cv_tasks.jpg "cv_tasks")](https://theaisummer.com/static/fd0197a92a28cba1dcc37865514f5b30/c08c5/cv_tasks.jpg)_[Stanford University School of Engineering](https://www.youtube.com/channel/UCdKG2JnvPu6mY1NDXYFfN0g)_

### Localization and Object Detection

[Image Localization](https://theaisummer.com/Localization_and_Object_Detection/) is the task of locating objects in an image and mark them with a bounding box, while object detection includes also the classification of the object.

The interconnected tasks are tackled by a fundamental model (and its improvements) in Computer Vision called R-CNN. RCNN and it’s predecessors Fast RCNN and Faster RCNN take advantage of regions proposals and Convolutional Neural Network.

An external system or the network itself( in the case of Faster RCNN) proposes some regions of interest in the form of a fixed-sized box, which might contain objects. These boxes are classified and corrected via a CNN (such as AlexNet), which decided if the box contains an object, what the object is and fixes the dimensions of the bounding box.

### Single-shot detectors

 [![yolo_app](https://theaisummer.com/static/32bde05b2ffccf127b8a89d76dddb57d/e5166/yolo_app.jpg "yolo_app")](https://theaisummer.com/static/32bde05b2ffccf127b8a89d76dddb57d/eea4a/yolo_app.jpg)_[https://github.com/karolmajek/darknet-pjreddie](https://github.com/karolmajek/darknet-pjreddie)_

Single-shot detectors and it’s most famous member [YOLO (You Only Look Once)](https://theaisummer.com/YOLO/) ditch the idea of region proposals and they use a set of predefined boxes.

These boxes are forwarded to a CNN, which predicts for each one a number of bounding boxes with a confidence score, it detects one object centered in it and it classifies the object into a category. In the end, we keep only the bounding boxes with a high score.

Over the years YOLOv2, YOLOv3, and YOLO900 improved on the original idea both on speed and accuracy.

### Semantic Segmentation

 [![semantic-segmentation](https://theaisummer.com/static/8b58a02198e13d2e29a41b40e7c6a035/8e1fc/semseg.jpg "semantic-segmentation")](https://theaisummer.com/static/8b58a02198e13d2e29a41b40e7c6a035/8e1fc/semseg.jpg)_[ICNet for Real-Time Semantic Segmentation on High-Resolution Images](https://hszhao.github.io/papers/eccv18_icnet.pdf)_

One of the fundamentals tasks in computer vision is the classification of all pixels in an image in classes based on their context, aka [Semantic Segmentation](https://theaisummer.com/Semantic_Segmentation/). In this direction, Fully Convolutional Networks (FCN) and U-Nets are the two most widely-used models.

*   **Fully Convolutional Networks (FCN)** is an encoder-decoder architecture with one convolutional and one deconvolutional network. The encoder downsamples the image to capture semantic and contextual information while the decoder upsamples to retrieve spatial information. That way we manage to retrieve the context of the image with the smaller time and space complexity possible.
    
*   **U-Nets** are based on the ingenious idea of skip-connections. Their encoder has the same size as the decoder and skip-connections transfer information from the first one to the latter in order to increase the resolution of the final output.
    

### Pose Estimation

[Pose Estimation](https://theaisummer.com/Human-Pose-Estimation/) is the problem of localizing human joints in images and videos and it can either be 2D or 3D. In 2D we estimate the (x,y) coordinates of each joint while in 3D the (x,y,z) coordinates.

[PoseNet](https://github.com/tensorflow/tfjs-models/tree/master/posenet) dominates the field (it’s the go-to model for most smartphone applications) of [pose estimation](https://www.fritz.ai/pose-estimation/) and it uses Convolutional Neural Networks (didn’t see that coming, did you?). We feed the image to a CNN and we use a single-pose or a multi-pose algorithm to detect poses. Each pose is associated with a confidence score and some key points coordinates. In the end, we keep the ones with the highest confidence score.

Wrapping up
-----------

There you have it. All the essential Deep Learning algorithms at the time.

Of course, I couldn’t include all the published architectures, because they are literally thousands. But most of them are based on one of those basic models and improve it with different techniques and tricks.

I am also confident that I will need to update this guide pretty soon, as new papers are coming out as we speak. But that is the beauty of Deep Learning. There is so much room for new breakthroughs that it’s kind of scary.

If you think that I forgot something, don’t hesitate to contact us on Social media or via email. I want this post to be as complete as possible.

Now it’s your time. Go and build your own amazing applications using these algorithms. Or even create a new one, that will make the cut in our list. Why not?

Have fun and keep learning AI.

_\* Disclosure: Please note that some of the links above might be affiliate links, and at no additional cost to you, we will earn a commission if you decide to make a purchase after clicking through._
