# Lecture Notes in

# MACHINE LEARNING

(^) **Dr V N Krishnachandran**
_Vidya Centre for Artificial Intelligence Research_


This page is intentionally left blank.


## LECTURE NOTES IN

## MACHINE LEARNING

### Dr V N Krishnachandran

```
Vidya Centre for Artificial Intelligence Research
Vidya Academy of Science & Technology
Thrissur - 680501
```

Copyright © 2018 V. N. Krishnachandran

Published by
Vidya Centre for Artificial Intelligence Research
Vidya Academy of Science & Technology
Thrissur - 680501, Kerala, India

The book was typeset by the author using the LATEX document preparation system.

Cover design: Author

Licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) License. You may
not use this file except in compliance with the License. You may obtain a copy of the License at
https://creativecommons.org/licenses/by/4.0/.

Price: Rs 0.00.

First printing: July 2018


## Preface

The book is exactly what its title claims it to be: lecture notes; nothing more, nothing less!
A reader looking for elaborate descriptive expositions of the concepts and tools of machine
learning will be disappointed with this book. There are plenty of books out there in the market
with different styles of exposition. Some of them give a lot of emphasis on the mathematical theory
behind the algorithms. In some others the emphasis is on the verbal descriptions of algorithms
avoiding the use of mathematical notations and concepts to the maximum extent possible. There is
one book the author of which is so afraid of introducing mathematical symbols that he introduces
as “the Greek letter sigma similar to abturned sideways". But among these books, the author of
these Notes could not spot a book that would give complete worked out examples illustrating the
various algorithms. These notes are expected to fill this gap.
The focus of this book is on giving a quick and fast introduction to the basic concepts and im-
portant algorithms in machine learning. In nearly all cases, whenever a new concept is introduced
it has been illustrated with “toy examples” and also with examples from real life situations. In the
case of algorithms, wherever possible, the working of the algorithm has been illustrated with con-
crete numerical examples. In some cases, the full algorithm may contain heavy use of mathematical
notations and concepts. Practitioners of machine learning sometimes treat such algorithms as “black
box algorithms”. Student readers of this book may skip these details on a first reading.
The book is written primarily for the students pursuing the B Tech programme in Computer
Science and Engineering of the APJ Abdul Kalam Technological University. The Curriculum for
the programme offers a course on machine learning as an elective course in the Seventh Semester
with code and name “CS 467 Machine Learning”. The selection of topics in the book was guided
by the contents of the syllabus for the course. The book will also be useful to faculty members who
teach the course.
Though the syllabus for CS 467 Machine Learning is reasonably well structured and covers most
of the basic concepts of machine learning, there is some lack of clarity on the depth to which the
various topics are to be covered. This ambiguity has been compounded by the lack of any mention
of a single textbook for the course and unfortunately the books cited as references treat machine
learning at varying levels. The guiding principle the author has adopted in the selection of materials
in the preparation of these notes is that, at the end of the course, the student must acquire enough
understanding about the methodologies and concepts underlying the various topics mentioned in the
syllabus.
Any study of machine learning algorithms without studying their implementations in software
packages is definitely incomplete. There are implementations of these algorithms available in the
R and Python programming languages. Two or three lines of code may be sufficient to implement
an algorithm. Since the syllabus for CS 467 Machine Learning does not mandate the study of such
implementations, this aspect of machine learning has not been included in this book. The students
are well advised to refer to any good book or the resources available in the internet to acquire a
working knowledge of these implementations.
Evidently, there are no original material in this book. The readers can see shadows of everything
presented here in other sources which include the reference books listed in the syllabus of the course
referred to earlier, other books on machine learning, published research/review papers and also
several open sources accessible through the internet. However, care has been taken to present the
material borrowed from other sources in a format digestible to the targeted audience. There are

```
iii
```

```
iv
```
more than a hundred figures in the book. Nearly all of them were drawn using the TikZ package for
LATEX. A few of the figures were created using the R programming language. A small number of
figures are reproductions of images available in various websites. There surely will be many errors

- conceptual, technical and printing – in these notes. The readers are earnestly requested to point
out such errors to the author so that an error free book can be brought up in the future.
    The author wishes to put on record his thankfulness to Vidya Centre for Artificial Intelligence
Research (V-CAIR) for agreeing to be the publisher of this book. V-CAIR is a research centre func-
tioning in Vidya Academy of Science & Technology, Thrissur, Kerala, established as part of the
“AI and Deep Learning: Skilling and Research” project launched by Royal Academy of Engineer-
ing, UK, in collaboration with University College, London, Brunel University, London and Bennett
University, India.

```
VAST Campus Dr V N Krishnachandran
July 2018 Department of Computer Applications
Vidya Academy of Science & Technology, Thrissur - 680501
(email:krishnachandran.vn@vidyaacademy.ac.in)
```

## Syllabus

```
Course code Course Name L - T - P - Credits Year of introduction
CS467 Machine Learning 3 - 0 - 0 - 3 2016
```
Course Objectives

- To introduce the prominent methods for machine learning
- To study the basics of supervised and unsupervised learning
- To study the basics of connectionist and other architectures

Syllabus

Introduction to Machine Learning, Learning in Artificial Neural Networks, Decision trees, HMM,
SVM, and other Supervised and Unsupervised learning methods.

Expected Outcome

The students will be able to

```
i) differentiate various learning approaches, and to interpret the concepts of supervised learn-
ing
```
```
ii) compare the different dimensionality reduction techniques
```
```
iii) apply theoretical foundations of decision trees to identify best split and Bayesian classifier
to label data points
```
```
iv) illustrate the working of classifier models like SVM, Neural Networks and identify classifier
model for typical machine learning applications
```
```
v) identify the state sequence and evaluate a sequence emission probability from a given HMM
```
```
vi) illustrate and apply clustering algorithms and identify its applicability in real life problems
```
References

1. Christopher M. Bishop,Pattern Recognition and Machine Learning, Springer, 2006.
2. Ethem Alpayidin,Introduction to Machine Learning(Adaptive Computation and machine
    Learning), MIT Press, 2004.
3. Margaret H. Dunham,Data Mining: Introductory and Advanced Topics, Pearson, 2006.

```
v
```

```
vi
```
4. Mitchell T.,Machine Learning, McGraw Hill.
5. Ryszard S. Michalski, Jaime G. Carbonell, and Tom M. Mitchell,Machine Learning : An
    Artificial Intelligence Approach, Tioga Publishing Company.

Course Plan

```
Module I.Introduction to Machine Learning, Examples of Machine Learning applications -
Learning associations, Classification, Regression, Unsupervised Learning, Reinforce-
ment Learning. Supervised learning- Input representation, Hypothesis class, Version
space, Vapnik-Chervonenkis (VC) Dimension
Hours: 6. Semester exam marks: 15%
```
```
Module II.Probably Approximately Learning (PAC), Noise, Learning Multiple classes, Model
Selection and Generalization, Dimensionality reduction- Subset selection, Principle
Component Analysis
Hours: 8. Semester exam marks: 15%
FIRST INTERNAL EXAMINATION
```
```
Module III.Classification- Cross validation and re-sampling methods- Kfold cross validation,
Boot strapping, Measuring classifier performance- Precision, recall, ROC curves.
Bayes Theorem, Bayesian classifier, Maximum Likelihood estimation, Density func-
tions, Regression
Hours: 8. Semester exam marks: 20%
```
```
Module IV.Decision Trees- Entropy, Information Gain, Tree construction, ID3, Issues in Decision
Tree learning- Avoiding Over-fitting, Reduced Error Pruning, The problem of Missing
Attributes, Gain Ratio, Classification by Regression (CART), Neural Networks- The
Perceptron, Activation Functions, Training Feed Forward Network by Back Propaga-
tion.
Hours: 6. Semester exam marks: 15%
SECOND INTERNAL EXAMINATION
```
```
Module V.Kernel Machines - Support Vector Machine - Optimal Separating hyper plane, Soft-
margin hyperplane, Kernel trick, Kernel functions. Discrete Markov Processes, Hid-
den Markov models, Three basic problems of HMMs - Evaluation problem, finding
state sequence, Learning model parameters. Combining multiple learners, Ways to
achieve diversity, Model combination schemes, Voting, Bagging, Booting
Hours: 8. Semester exam marks: 20%
```
```
Module VI.Unsupervised Learning - Clustering Methods - K-means, Expect-ation-Maxi-mization
Algorithm, Hierarchical Clustering Methods, Density based clustering
Hours: 6. Semester exam marks: 15%
END SEMESTER EXAMINATION
```
Question paper pattern

1. There will be FOUR parts in the question paper: A, B, C, D.
2. Part A

```
a) Total marks: 40
b) TEN questions, each have 4 marks, covering all the SIX modules (THREE questions
from modules I & II; THREE questions from modules III & IV; FOUR questions from
modules V & VI).
```

```
vii
```
```
c) All the TEN questions have to be answered.
```
3. Part B

```
a) Total marks: 18
b) THREE questions, each having 9 marks. One question is from module I; one question
is from module II; one question uniformly covers modules I & II.
c) Any TWO questions have to be answered.
d) Each question can have maximum THREE subparts.
```
4. Part C

```
a) Total marks: 18
b) THREE questions, each having 9 marks. One question is from module III; one question
is from module IV; one question uniformly covers modules III & IV.
c) Any TWO questions have to be answered.
d) Each question can have maximum THREE subparts.
```
5. Part D

```
a) Total marks: 24
b) THREE questions, each having 12 marks. One question is from module V; one question
is from module VI; one question uniformly covers modules V & VI.
c) Any TWO questions have to be answered.
d) Each question can have maximum THREE subparts.
```
6. There will be AT LEAST 60% analytical/numerical questions in all possible combinations of
    question choices.


## Contents

Introduction iii




## List of Figures

```
1.1 Components of learning process.............................. 2
1.2 Example for “examples” and “features” collected in a matrix format (data relates
to automobiles and their features)............................. 5
1.3 Graphical representation of data in Table 1.1. Solid dots represent data in “Pass”
class and hollow dots data in “Fail” class. The class label of the square dot is to be
determined.......................................... 7
1.4 Supervised learning..................................... 12
```
```
2.1 Data in Table 2.1 with hollow dots representing positive examples and solid dots
representing negative examples.............................. 16
2.2 An example hypothesis defined by Eq. (2.5)....................... 17
2.3 Hypothesish′is more general than hypothesish′′if and only ifS′′⊆S′....... 18
2.4 Values ofmwhich define the version space with data in Table 2.1 and hypothesis
space defined by Eq.(2.4).................................. 19
2.5 Scatter plot of price-power data (hollow circles indicate positive examples and
solid dots indicate negative examples).......................... 20
2.6 The version space consists of hypotheses corresponding to axis-aligned rectangles
contained in the shaded region............................... 20
2.7 Examples for overfitting and overfitting models..................... 24
2.8 Fitting a classification boundary.............................. 25
```
```
3.1 Different forms of the set{x∈S∶h(x)= 1 }forD={a;b;c}............. 28
3.2 Geometrical representation of the hypothesisha;b;c................... 30
3.3 A hypothesisha;b;cconsistent with the dichotomy defined by the subset{A;C}of
{A;B;C}.......................................... 30
3.4 There is no hypothesisha;b;cconsistent with the dichotomy defined by the subset
{A;C}of{A;B;C;D}.................................. 30
3.5 An axis-aligned rectangle in the Euclidean plane.................... 32
3.6 Axis-aligned rectangle which gives the tightest fit to the positive examples..... 33
```
```
4.1 Principal components.................................... 39
4.2 Scatter plot of data in Table 4.2.............................. 43
4.3 Coordinate system for principal components....................... 45
4.4 Projections of data points on the axis of the first principal component........ 46
4.5 Geometrical representation of one-dimensional approximation to the data in Table
4.2.............................................. 46
```
```
5.1 One iteration in a 5-fold cross-validation......................... 50
5.2 The ROC space and some special points in the space.................. 56
5.3 ROC curves of three different classifiers A, B, C.................... 57
5.4 ROC curve of data in Table 5.3 showing the points closest to the perfect prediction
point( 0 ; 1 )......................................... 58
```
```
6.1 EventsA;B;Cwhich are not mutually independent: Eqs.(6.1)–(6.3) are satisfied,
but Eq.(6.4) is not satisfied................................. 62
```
```
xi
```

## LIST OF FIGURES xii


LIST OF FIGURES xiii

- 1 Introduction to machine learning Syllabus v
   - 1.1 Introduction
   - 1.2 How machines learn
   - 1.3 Applications of machine learning
   - 1.4 Understanding data
   - 1.5 General classes of machine learning problems
   - 1.6 Different types of learning
   - 1.7 Sample questions
- 2 Some general concepts
   - 2.1 Input representation
   - 2.2 Hypothesis space
   - 2.3 Ordering of hypotheses
   - 2.4 Version space
   - 2.5 Noise
   - 2.6 Learning multiple classes
   - 2.7 Model selection
   - 2.8 Generalisation
   - 2.9 Sample questions
- 3 VC dimension and PAC learning
   - 3.1 Vapnik-Chervonenkis dimension
   - 3.2 Probably approximately correct learning
   - 3.3 Sample questions
- 4 Dimensionality reduction
   - 4.1 Introduction
   - 4.2 Why dimensionality reduction is useful
   - 4.3 Subset selection
   - 4.4 Principal component analysis
   - 4.5 Sample questions
- 5 Evaluation of classifiers
   - 5.1 Methods of evaluation
   - 5.2 Cross-validation
   - 5.3 K-fold cross-validation
   - 5.4 Measuring error
   - 5.5 Receiver Operating Characteristic (ROC)
   - 5.6 Sample questions
- 6 Bayesian classifier and ML estimation CONTENTS ix
   - 6.1 Conditional probability
   - 6.2 Bayes’ theorem
   - 6.3 Naive Bayes algorithm
   - 6.4 Using numeric features with naive Bayes algorithm
   - 6.5 Maximum likelihood estimation (ML estimation)
   - 6.6 Sample questions
- 7 Regression
   - 7.1 Definition
   - 7.2 Criterion for minimisation of error
   - 7.3 Simple linear regression
   - 7.4 Polynomial regression
   - 7.5 Multiple linear regression
   - 7.6 Sample questions
- 8 Decision trees
   - 8.1 Decision tree: Example
   - 8.2 Two types of decision trees
   - 8.3 Classification trees
   - 8.4 Feature selection measures
   - 8.5 Entropy
   - 8.6 Information gain
   - 8.7 Gini indices
   - 8.8 Gain ratio
   - 8.9 Decision tree algorithms
   - 8.10 The ID3 algorithm
   - 8.11 Regression trees
   - 8.12 CART algorithm
   - 8.13 Other decision tree algorithms
   - 8.14 Issues in decision tree learning
   - 8.15 Avoiding overfitting of data
   - 8.16 Problem of missing attributes
   - 8.17 Sample questions
- 9 Neural networks
   - 9.1 Introduction
   - 9.2 Biological motivation
   - 9.3 Artificial neurons
   - 9.4 Activation function
   - 9.5 Perceptron
   - 9.6 Artificial neural networks
   - 9.7 Characteristics of an ANN
   - 9.8 Backpropagation
   - 9.9 Introduction to deep learning
   - 9.10 Sample questions
- 10 Support vector machines
   - 10.1 An example
   - 10.2 Finite dimensional vector spaces
   - 10.3 Hyperplanes
   - 10.4 Two-class data sets
   - 10.5 Linearly separable data
   - 10.6 Maximal margin hyperplanes
   - 10.7 Mathematical formulation of the SVM problem
   - 10.8 Solution of the SVM problem CONTENTS x
   - 10.9 Soft margin hyperlanes
   - 10.10 Kernel functions
   - 10.11 The kernel method (kernel trick)
   - 10.12 Multiclass SVM’s
   - 10.13 Sample questions
- 11 Hidden Markov models
   - 11.1 Discrete Markov processes: Examples
   - 11.2 Discrete Markov processes: General case
   - 11.3 Hidden Markov models
   - 11.4 Three basic problems of HMMs
   - 11.5 HMM application: Isolated word recognition
   - 11.6 Sample questions
- 12 Combining multiple learners
   - 12.1 Why combine many learners
   - 12.2 Ways to achieve diversity
   - 12.3 Model combination schemes
   - 12.4 Ensemble learning⋆.
   - 12.5 Random forest⋆
   - 12.6 Sample questions
- 13 Clustering methods
   - 13.1 Clustering
   - 13.2 k-means clustering
   - 13.3 Multi-modal distributions
   - 13.4 Mixture of normal distributions
   - 13.5 Mixtures in terms of latent variables
   - 13.6 Expectation-maximisation algorithm
   - 13.7 The EM algorithm for Gaussian mixtures
   - 13.8 Hierarchical clustering
   - 13.9 Measures of dissimilarity
   - 13.10 Algorithm for agglomerative hierarchical clustering
   - 13.11 Algorithm for divisive hierarchical clustering
   - 13.12 Density-based clustering
   - 13.13 Sample questions
- Bibliography
- Index
   - (6.2) are not satisfied. 6.2 EventsA;B;Cwhich are not mutually independent: Eq.(6.4) is satisfied but Eqs.(6.1)–
- 6.3 Discretization of numeric data: Example
- 7.1 Errors in observed values
- 7.2 Regression model for Table 7.2
- 7.3 Plot of quadratic polynomial model
- 7.4 The regression plane for the data in Table 7.4
- 8.1 Example for a decision tree
- 8.2 The graph-theoretical representation of the decision tree in Figure 8.6
- 8.3 Classification tree
- 8.4 Classification tree
- 8.5 Classification tree
- 8.6 Plot ofpvs. Entropy
- 8.7 Root node of the decision tree for data in Table 8.9
- 8.8 Decision tree for data in Table 8.9, after selecting the branching feature at root node
- 8.9 Decision tree for data in Table 8.9, after selecting the branching feature at Node
- 8.10 Decision tree for data in Table 8.9
- 8.11 Part of a regression tree for Table 8.11
- 8.12 Part of regression tree for Table 8.11
- 8.13 A regression tree for Table 8.11
- 8.14 Impact of overfitting in decision tree learning
- 9.1 Anatomy of a neuron
- 9.2 Flow of signals in a biological neuron
- 9.3 Schematic representation of an artificial neuron
- 9.4 Simplified representation of an artificial neuron
- 9.5 Threshold activation function
- 9.6 Unit step activation function
- 9.7 The sigmoid activation function
- 9.8 Linear activation function
- 9.9 Piecewise linear activation function
- 9.10 Gaussian activation function
- 9.11 Hyperbolic tangent activation function
- 9.12 Schematic representation of a perceptrn
- 9.13 Representation ofx 1 ANDx 2 by a perceptron
- 9.14 An ANN with only one layer
- 9.15 An ANN with two layers
- 9.16 Examples of different topologies of networks
- 9.17 A simplified model of the error surface showing the direction of gradient
- 9.18 ANN for illustrating backpropagation algorithm
- 9.19 ANN for illustrating backpropagation algorithm with initial values for weights
- 9.20 Notations of backpropagation algorithm
- 9.21 Notations of backpropagation algorithm: Thei-th node in layerj.
- 9.22 A shallow neural network
- 9.23 A deep neural network with three hidden layers
   - “no”) 10.1 Scatter plot of data in Table 10.1 (filled circles represent “yes” and unfilled circles
- 10.2 Scatter plot of data in Table 10.1 with a separating line
- 10.3 Two separating lines for the data in Table 10.1
- 10.4 Shortest perpendicular distance of a separating line from data points
- 10.5 Maximum margin line for data in Table 10.1
- 10.6 Support vectors for data in Table 10.1
   - in Table 10.1 10.7 Boundaries of “street” of maximum width separating “yes” points and “no” points
   - ming language 10.8 Plot of the maximum margin line of data in Table 10.1 produced by the R program-
- 10.9 Half planes defined by a line
- 10.10 Perpendicular distance of a point from a plane
- 10.11 Scatterplot of data in Table 10.2
- 10.12 Maximal separating hyperplane, margin and support vectors
- 10.13 Maximal margin hyperplane of a 2-sample set in 2-dimensional space
- 10.14 Maximal margin hyperplane of a 3-sample set in 2-dimensional space
- 10.15 Soft margin hyperplanes
- 10.16 One-against all
- 10.17 One-against-one
- 11.1 A state diagram showing state transition probabilities
- 11.2 A two-coin model of an HMM
   - symbol HMM 11.3 AnN-state urn and ball model which illustrates the general case of a discrete
- 11.4 Block diagram of an isolated word HMM recogniser
- 12.1 Example of random forest with majority voting
- 13.1 Scatter diagram of data in Table 13.1
- 13.2 Initial choice of cluster centres and the resulting clusters
- 13.3 Cluster centres after first iteration and the corresponding clusters
- 13.4 New cluster centres and the corresponding clusters
- 13.5 Probability distributions
   - Table 13.3 13.6 Graph of pdf defined by Eq.(13.9) superimposed on the histogram of the data in
- 13.7 A dendrogram of the dataset{a;b;c;d;e}
- 13.8 Different ways of drawing dendrogram
   - clusters at different levels 13.9 A dendrogram of the dataset{a;b;c;d;e}showing the distances (heights) of the
- 13.10 Hierarchical clustering using agglomerative method
- 13.11 Hierarchical clustering using divisive method
- 13.12 Length of the solid line “ae” ismax{d(x;y)∶x∈A;y∈B}.
- 13.13 Length of the solid line “bc” ismin{d(x;y)∶x∈A;y∈B}.
- 13.14 Dendrogram for the data given in Table 13.4 (complete linkage clustering)
- 13.15 Dendrogram for the data given in Table 13.4 (single linkage clustering)
- 13.16 Dx= (average of dashed lines)−(average of solid lines)
- 13.17 Clusters of points and noise points not belonging to any of those clusters
   - (d)ra noise point 13.18 Withm 0 = 4 : (a)pa point of high density (b)pa core point (c)pa border point
   - reachable fromp 13.19 Withm 0 = 4 : (a)qis directly density-reachable fromp(b)qis indirectly density-


Chapter 1

## Introduction to machine learning

In this chapter, we consider different definitions of the term “machine learning” and explain what
is meant by “learning” in the context of machine learning. We also discuss the various components
of the machine learning process. There are also brief discussions about different types learning like
supervised learning, unsupervised learning and reinforcement learning.

### 1.1 Introduction

1.1.1 Definition of machine learning

Arthur Samuel, an early American leader in the field of computer gaming and artificial intelligence,
coined the term “Machine Learning” in 1959 while at IBM. He defined machine learning as “the field
of study that gives computers the ability to learn without being explicitly programmed.” However,
there is no universally accepted definition for machine learning. Different authors define the term
differently. We give below two more definitions.

1. Machine learning is programming computers to optimize a performance criterion using exam-
    ple data or past experience. We have a model defined up to some parameters, and learning is
    the execution of a computer program to optimize the parameters of the model using the train-
    ing data or past experience. The model may be predictive to make predictions in the future, or
    descriptive to gain knowledge from data, or both (see [2] p.3).
2. The field of study known as machine learning is concerned with the question of how to con-
    struct computer programs that automatically improve with experience (see [4], Preface.).

Remarks

In the above definitions we have used the term “model” and we will be using this term at several
contexts later in this book. It appears that there is no universally accepted one sentence definition
of this term. Loosely, it may be understood as some mathematical expression or equation, or some
mathematical structures such as graphs and trees, or a division of sets into disjoint subsets, or a set
of logical “if:::then:::else:::” rules, or some such thing. It may be noted that this is not an
exhaustive list.

1.1.2 Definition of learning

Definition

A computer program is said tolearnfrom experienceEwith respect to some class of tasksTand
performance measureP, if its performance at tasksT, as measured byP, improves with experience
E.

#### 1


#### CHAPTER 1. INTRODUCTION TO MACHINE LEARNING 2

Examples

```
i) Handwriting recognition learning problem
```
- TaskT: Recognising and classifying handwritten words within images
- PerformanceP: Percent of words correctly classified
- Training experienceE: A dataset of handwritten words with given classifications

```
ii) A robot driving learning problem
```
- TaskT: Driving on highways using vision sensors
- Performance measureP: Average distance traveled before an error
- training experience: A sequence of images and steering commands recorded while
    observing a human driver

```
iii) A chess learning problem
```
- TaskT: Playing chess
- Performance measureP: Percent of games won against opponents
- Training experienceE: Playing practice games against itself

Definition

A computer program which learns from experience is called amachine learning programor simply
alearning program. Such a program is sometimes also referred to as alearner.

### 1.2 How machines learn

1.2.1 Basic components of learning process

The learning process, whether by a human or a machine, can be divided into four components,
namely, data storage, abstraction, generalization and evaluation. Figure 1.1 illustrates the various
components and the steps involved in the learning process.

```
Data Concepts Inferences
```
```
Data storage Abstraction Generalization Evaluation
```
```
Figure 1.1: Components of learning process
```
1. Data storage
    Facilities for storing and retrieving huge amounts of data are an important component of
    the learning process. Humans and computers alike utilize data storage as a foundation for
    advanced reasoning.
       - In a human being, the data is stored in the brain and data is retrieved using electrochem-
          ical signals.
       - Computers use hard disk drives, flash memory, random access memory and similar de-
          vices to store data and use cables and other technology to retrieve data.


#### CHAPTER 1. INTRODUCTION TO MACHINE LEARNING 3

2. Abstraction
    The second component of the learning process is known asabstraction.
    Abstraction is the process of extracting knowledge about stored data. This involves creating
    general concepts about the data as a whole. The creation of knowledge involves application
    of known models and creation of new models.
    The process of fitting a model to a dataset is known astraining. When the model has been
    trained, the data is transformed into an abstract form that summarizes the original information.
3. Generalization
    The third component of the learning process is known asgeneralisation.
    The term generalization describes the process of turning the knowledge about stored data into
    a form that can be utilized for future action. These actions are to be carried out on tasks that
    are similar, but not identical, to those what have been seen before. In generalization, the goal
    is to discover those properties of the data that will be most relevant to future tasks.
4. Evaluation
    Evaluationis the last component of the learning process.
    It is the process of giving feedback to the user to measure the utility of the learned knowledge.
    This feedback is then utilised to effect improvements in the whole learning process.

### 1.3 Applications of machine learning

Application of machine learning methods to large databases is called data mining. In data mining, a
large volume of data is processed to construct a simple model with valuable use, for example, having
high predictive accuracy.
The following is a list of some of the typical applications of machine learning.

1. In retail business, machine learning is used to study consumer behaviour.
2. In finance, banks analyze their past data to build models to use in credit applications, fraud
    detection, and the stock market.
3. In manufacturing, learning models are used for optimization, control, and troubleshooting.
4. In medicine, learning programs are used for medical diagnosis.
5. In telecommunications, call patterns are analyzed for network optimization and maximizing
    the quality of service.
6. In science, large amounts of data in physics, astronomy, and biology can only be analyzed fast
    enough by computers. The World Wide Web is huge; it is constantly growing and searching
    for relevant information cannot be done manually.
7. In artificial intelligence, it is used to teach a system to learn and adapt to changes so that the
    system designer need not foresee and provide solutions for all possible situations.
8. It is used to find solutions to many problems in vision, speech recognition, and robotics.
9. Machine learning methods are applied in the design of computer-controlled vehicles to steer
    correctly when driving on a variety of roads.
10. Machine learning methods have been used to develop programmes for playing games such as
chess, backgammon and Go.


#### CHAPTER 1. INTRODUCTION TO MACHINE LEARNING 4

### 1.4 Understanding data

Since an important component of the machine learning process is data storage, we briefly consider
in this section the different types and forms of data that are encountered in the machine learning
process.

1.4.1 Unit of observation

By aunit of observationwe mean the smallest entity with measured properties of interest for a study.

Examples

- A person, an object or a thing
- A time point
- A geographic region
- A measurement

Sometimes, units of observation are combined to form units such as person-years.

1.4.2 Examples and features

Datasets that store the units of observation and their properties can be imagined as collections of
data consisting of the following:

- Examples
    An “example” is an instance of the unit of observation for which properties have been recorded.
    An “example” is also referred to as an “instance”, or “case” or “record.” (It may be noted that
    the word “example” has been used here in a technical sense.)
- Features
    A “feature” is a recorded property or a characteristic of examples. It is also referred to as
    “attribute”, or “variable” or “feature.”

Examples for “examples” and “features”

1. Cancer detection
    Consider the problem of developing an algorithm for detecting cancer. In this study we note
    the following.

```
(a) The units of observation are the patients.
(b) The examples are members of a sample of cancer patients.
(c) The following attributes of the patients may be chosen as the features:
```
- gender
- age
- blood pressure
- the findings of the pathology report after a biopsy
2. Pet selection
Suppose we want to predict the type of pet a person will choose.

```
(a) The units are the persons.
(b) The examples are members of a sample of persons who own pets.
```

#### CHAPTER 1. INTRODUCTION TO MACHINE LEARNING 5

Figure 1.2: Example for “examples” and “features” collected in a matrix format (data relates to
automobiles and their features)

```
(c) The features might include age, home region, family income, etc. of persons who own
pets.
```
3. Spam e-mail
    Let it be required to build a learning algorithm to identify spam e-mail.

```
(a) The unit of observation could be an e-mail messages.
(b) The examples would be specific messages.
(c) The features might consist of the words used in the messages.
```
Examples and features are generally collected in a “matrix format”. Fig. 1.2 shows such a data
set.

1.4.3 Different forms of data

1. Numeric data
    If a feature represents a characteristic measured in numbers, it is called a numeric feature.
2. Categorical or nominal
    A categorical feature is an attribute that can take on one of a limited, and usually fixed, number
    of possible values on the basis of some qualitative property. A categorical feature is also called
    a nominal feature.
3. Ordinal data
    This denotes a nominal variable with categories falling in an ordered list. Examples include
    clothing sizes such as small, medium, and large, or a measurement of customer satisfaction
    on a scale from “not at all happy” to “very happy.”

Examples

In the data given in Fig.1.2, the features “year”, “price” and “mileage” are numeric and the features
“model”, “color” and “transmission” are categorical.


#### CHAPTER 1. INTRODUCTION TO MACHINE LEARNING 6

### 1.5 General classes of machine learning problems

1.5.1 Learning associations

1. Association rule learning

Association rule learningis a machine learning method for discovering interesting relations, called
“association rules”, between variables in large databases using some measures of “interestingness”.

2. Example

```
Consider a supermarket chain. The management of the chain is interested in knowing whether
there are any patterns in the purchases of products by customers like the following:
```
```
“If a customer buys onions and potatoes together, then he/she is likely to also buy
hamburger.”
```
```
From the standpoint of customer behaviour, this defines an association between the set of
products {onion, potato} and the set {burger}. This association is represented in the form of
a rule as follows:
{onion, potato}⇒{burger}
The measure of how likely a customer, who has bought onion and potato, to buy burger also
is given by the conditional probability
```
```
P({onion, potato}S{burger}):
```
```
If this conditional probability is 0.8, then the rule may be stated more precisely as follows:
```
```
“80% of customers who buy onion and potato also buy burger.”
```
3. How association rules are made use of

Consider an association rule of the form
X⇒Y;

that is, if people buyXthen they are also likely to buyY.
Suppose there is a customer who buysXand does not buyY. Then that customer is a potential
Ycustomer. Once we find such customers, we can target them for cross-selling. A knowledge of
such rules can be used for promotional pricing or product placements.

4. General case

In finding an association ruleX⇒Y, we are interested in learning a conditional probability of
the formP(YSX)whereYis the product the customer may buy andXis the product or the set of
products the customer has already purchased.
If we may want to make a distinction among customers, we may estimateP(YSX;D)where
Dis a set of customer attributes, like gender, age, marital status, and so on, assuming that we have
access to this information.

5. Algorithms

There are several algorithms for generating association rules. Some of the well-known algorithms
are listed below:

```
a) Apriori algorithm
```
```
b) Eclat algorithm
```
```
c) FP-Growth Algorithm (FP stands for Frequency Pattern)
```

#### CHAPTER 1. INTRODUCTION TO MACHINE LEARNING 7

1.5.2 Classification

1. Definition

In machine learning,classificationis the problem of identifying to which of a set of categories a
new observation belongs, on the basis of a training set of data containing observations (or instances)
whose category membership is known.

2. Example

Consider the following data:

```
Score1 29 22 10 31 17 33 32 20
Score2 43 29 47 55 18 54 40 41
Result Pass Fail Fail Pass Fail Pass Pass Pass
```
```
Table 1.1: Example data for a classification problem
```
Data in Table 1.1 is the training set of data. There are two attributes “Score1” and “Score2”. The
class label is called “Result”. The class label has two possible values “Pass” and “Fail”. The data
can be divided into two categories or classes: The set of data for which the class label is “Pass” and
the set of data for which the class label is“Fail”.
Let us assume that we have no knowledge about the data other than what is given in the table.
Now, the problem can be posed as follows: If we have some new data, say “Score1 = 25” and
“Score2 = 36”, what value should be assigned to “Result” corresponding to the new data; in other
words, to which of the two categories or classes the new observation should be assigned? See Figure
1.3 for a graphical representation of the problem.

```
Score1
```
```
Score2
```
#### ?

#### 0 10 20 30 40

#### 10

#### 20

#### 30

#### 40

#### 50

#### 60

Figure 1.3: Graphical representation of data in Table 1.1. Solid dots represent data in “Pass” class
and hollow dots data in “Fail” class. The class label of the square dot is to be determined.

To answer this question, using the given data alone we need to find the rule, or the formula, or
the method that has been used in assigning the values to the class label “Result”. The problem of
finding this rule or formula or the method is the classification problem. In general, even the general
form of the rule or function or method will not be known. So several different rules, etc. may have
to be tested to obtain the correct rule or function or method.


#### CHAPTER 1. INTRODUCTION TO MACHINE LEARNING 8

3. Real life examples

```
i) Optical character recognition
Optical character recognitionproblem, which is the problem of recognizing character codes
from their images, is an example of classification problem. This is an example where there
are multiple classes, as many as there are characters we would like to recognize. Especially
interesting is the case when the characters are handwritten. People have different handwrit-
ing styles; characters may be written small or large, slanted, with a pen or pencil, and there
are many possible images corresponding to the same character.
```
```
ii) Face recognition
In the case offace recognition, the input is an image, the classes are people to be recognized,
and the learning program should learn to associate the face images to identities. This prob-
lem is more difficult than optical character recognition because there are more classes, input
image is larger, and a face is three-dimensional and differences in pose and lighting cause
significant changes in the image.
```
```
iii) Speech recognition
Inspeech recognition, the input is acoustic and the classes are words that can be uttered.
```
```
iv) Medical diagnosis
Inmedical diagnosis, the inputs are the relevant information we have about the patient and
the classes are the illnesses. The inputs contain the patient’s age, gender, past medical
history, and current symptoms. Some tests may not have been applied to the patient, and
thus these inputs would be missing.
```
```
v) Knowledge extraction
Classification rules can also be used forknowledge extraction. The rule is a simple model
that explains the data, and looking at this model we have an explanation about the process
underlying the data.
```
```
vi) Compression
Classification rules can be used forcompression. By fitting a rule to the data, we get an
explanation that is simpler than the data, requiring less memory to store and less computation
to process.
```
```
vii) More examples
Here are some further examples of classification problems.
```
```
(a) An emergency room in a hospital measures 17 variables like blood pressure, age, etc.
of newly admitted patients. A decision has to be made whether to put the patient in an
ICU. Due to the high cost of ICU, only patients who may survive a month or more are
given higher priority. Such patients are labeled as “low-risk patients” and others are
labeled “high-risk patients”. The problem is to device a rule to classify a patient as a
“low-risk patient” or a “high-risk patient”.
(b) A credit card company receives hundreds of thousands of applications for new cards.
The applications contain information regarding several attributes like annual salary,
age, etc. The problem is to devise a rule to classify the applicants to those who are
credit-worthy, who are not credit-worthy or to those who require further analysis.
(c) Astronomers have been cataloguing distant objects in the sky using digital images cre-
ated using special devices. The objects are to be labeled as star, galaxy, nebula, etc.
The data is highly noisy and are very faint. The problem is to device a rule using which
a distant object can be correctly labeled.
```

#### CHAPTER 1. INTRODUCTION TO MACHINE LEARNING 9

4. Discriminant

Adiscriminantof a classification problem is a rule or a function that is used to assign labels to new
observations.

Examples

```
i) Consider the data given in Table 1.1 and the associated classification problem. We may
consider the following rules for the classification of the new data:
```
```
IF Score1+Score2≥ 60 , THEN “Pass” ELSE “Fail”.
IF Score1≥ 20 AND Score2≥ 40 THEN “Pass” ELSE “Fail”.
```
```
Or, we may consider the following rules with unspecified values forM;m 1 ;m 2 and then by
some method estimate their values.
```
```
IF Score1+Score2≥M, THEN “Pass” ELSE “Fail”.
IF Score1≥m 1 AND Score2≥m 2 THEN “Pass” ELSE “Fail”.
```
```
ii) Consider a finance company which lends money to customers. Before lending money, the
company would like to assess the risk associated with the loan. For simplicity, let us assume
that the company assesses the risk based on two variables, namely, the annual income and
the annual savings of the customers.
Letx 1 be the annual income andx 2 be the annual savings of a customer.
```
- After using the past data, a rule of the following form with suitable values for 1 and
     2 may be formulated:
       IFx 1 > 1 ANDx 2 > 2 THEN “low-risk” ELSE “high-risk”.
    This rule is an example of a discriminant.
- Based on the past data, a rule of the following form may also be formulated:
    IFx 2 − 0 : 2 x 1 > 0 THEN “low-risk” ELSE “high-risk”.
    In this case the rule may be thought of as the discriminant. The functionf(x 1 ;x 2 )=
    x 2 − 0 ; 2 x 1 can also be considered as the discriminant.
5. Algorithms

There are several machine learning algorithms for classification. The following are some of the
well-known algorithms.

```
a) Logistic regression
```
```
b) Naive Bayes algorithm
```
```
c) k-NN algorithm
```
```
d) Decision tree algorithm
```
```
e) Support vector machine algorithm
```
```
f) Random forest algorithm
```

#### CHAPTER 1. INTRODUCTION TO MACHINE LEARNING 10

Remarks

- A classification problem requires that examples be classified into one of two or more classes.
- A classification can have real-valued or discrete input variables.
- A problem with two classes is often called a two-class or binary classification problem.
- A problem with more than two classes is often called a multi-class classification problem.
- A problem where an example is assigned multiple classes is called a multi-label classification
    problem.

1.5.3 Regression

1. Definition

In machine learning, aregression problemis the problem of predicting the value of a numeric vari-
able based on observed values of the variable. The value of the output variable may be a number,
such as an integer or a floating point value. These are often quantities, such as amounts and sizes.
The input variables may be discrete or real-valued.

2. Example

Consider the data on car prices given in Table 1.2.

```
Price Age Distance Weight
(US$) (years) (KM) (pounds)
13500 23 46986 1165
13750 23 72937 1165
13950 24 41711 1165
14950 26 48000 1165
13750 30 38500 1170
12950 32 61000 1170
16900 27 94612 1245
18600 30 75889 1245
21500 27 19700 1185
12950 23 71138 1105
```
```
Table 1.2: Prices of used cars: example data for regression
```
Suppose we are required to estimate the price of a car aged 25 years with distance 53240 KM
and weight 1200 pounds. This is an example of a regression problem beause we have to predict the
value of the numeric variable “Price”.

3. General approach

Letxdenote the set of input variables andythe output variable. In machine learning, the general
approach to regression is to assume a model, that is, some mathematical relation betweenxandy,
involving some parameters say,, in the following form:

```
y=f(x;)
```
The functionf(x;)is called theregression function. The machine learning algorithm optimizes
the parameters in the setsuch that the approximation error is minimized; that is, the estimates
of the values of the dependent variableyare as close as possible to the correct values given in the
training set.


#### CHAPTER 1. INTRODUCTION TO MACHINE LEARNING 11

Example

```
For example, if the input variables are “Age”, “Distance” and “Weight” and the output variable
is “Price”, the model may be
```
```
y=f(x;)
Price =a 0 +a 1 ×(Age)+a 2 ×(Distance)+a 3 ×(Weight)
```
```
wherex=(Age, Distance, Weight)denotes the the set of input variables and=(a 0 ;a 1 ;a 2 ;a 3 )
denotes the set of parameters of the model.
```
4. Different regression models

There are various types of regression techniques available to make predictions. These techniques
mostly differ in three aspects, namely, the number and type of independent variables, the type of
dependent variables and the shape of regression line. Some of these are listed below.

- Simple linear regression: There is only one continuous independent variablexand the as-
    sumed relation between the independent variable and the dependent variableyis

```
y=a+bx:
```
- Multivariate linear regression: There are more than one independent variable, sayx 1 ;:::;xn,
    and the assumed relation between the independent variables and the dependent variable is

```
y=a 0 +a 1 x 1 + ⋯ +anxn:
```
- Polynomial regression: There is only one continuous independent variablexand the assumed
    model is
       y=a 0 +a 1 x+ ⋯ +anxn:
- Logistic regression: The dependent variable is binary, that is, a variable which takes only the
    values 0 and 1. The assumed model involves certain probability distributions.

### 1.6 Different types of learning

In general, machine learning algorithms can be classified into three types.

1.6.1 Supervised learning

Supervised learningis the machine learning task of learning a function that maps an input to an
output based on example input-output pairs.

In supervised learning, each example in the training set is a pair consisting of an input object
(typically a vector) and an output value. A supervised learning algorithm analyzes the training
data and produces a function, which can be used for mapping new examples. In the optimal case,
the function will correctly determine the class labels for unseen instances. Both classification and
regression problems are supervised learning problems.
A wide range of supervised learning algorithms are available, each with its strengths and weak-
nesses. There is no single learning algorithm that works best on all supervised learning problems.


#### CHAPTER 1. INTRODUCTION TO MACHINE LEARNING 12

```
Figure 1.4: Supervised learning
```
Remarks

A “supervised learning” is so called because the process of an algorithm learning from the training
dataset can be thought of as a teacher supervising the learning process. We know the correct answers
(that is, the correct outputs), the algorithm iteratively makes predictions on the training data and
is corrected by the teacher. Learning stops when the algorithm achieves an acceptable level of
performance.

Example

```
Consider the following data regarding patients entering a clinic. The data consists of the
gender and age of the patients and each patient is labeled as “healthy” or “sick”.
```
```
gender age label
M 48 sick
M 67 sick
F 53 healthy
M 49 healthy
F 34 sick
M 21 healthy
```
```
Based on this data, when a new patient enters the clinic, how can one predict whether he/she
is healthy or sick?
```
1.6.2 Unsupervised learning

Unsupervised learningis a type of machine learning algorithm used to draw inferences from datasets
consisting of input data without labeled responses.

In unsupervised learning algorithms, a classification or categorization is not included in the
observations. There are no output values and so there is no estimation of functions. Since the
examples given to the learner are unlabeled, the accuracy of the structure that is output by the
algorithm cannot be evaluated.
The most common unsupervised learning method is cluster analysis, which is used for ex-
ploratory data analysis to find hidden patterns or grouping in data.

Example

```
Consider the following data regarding patients entering a clinic. The data consists of the
gender and age of the patients.
```

#### CHAPTER 1. INTRODUCTION TO MACHINE LEARNING 13

```
gender age
M 48
M 67
F 53
M 49
F 34
M 21
```
```
Based on this data, can we infer anything regarding the patients entering the clinic?
```
1.6.3 Reinforcement learning

Reinforcement learningis the problem of getting an agent to act in the world so as to maximize its
rewards.

A learner (the program) is not told what actions to take as in most forms of machine learning, but
instead must discover which actions yield the most reward by trying them. In the most interesting
and challenging cases, actions may affect not only the immediate reward but also the next situations
and, through that, all subsequent rewards.
For example, consider teaching a dog a new trick: we cannot tell it what to do, but we can
reward/punish it if it does the right/wrong thing. It has to find out what it did that made it get the
reward/punishment. We can use a similar method to train computers to do many tasks, such as
playing backgammon or chess, scheduling jobs, and controlling robot limbs.
Reinforcement learning is different from supervised learning. Supervised learning is learning
from examples provided by a knowledgeable expert.

### 1.7 Sample questions

(a) Short answer questions

1. What is meant by “learning” in the context of machine learning?
2. List out the types of machine learning.
3. Distinguish between classification and regression.
4. What are the differences between supervised and unsupervised learning?
5. What is meant by supervised classification?
6. Explain supervised learning with an example.
7. What do you mean by reinforcement learning?
8. What is an association rule?
9. Explain the concept of Association rule learning. Give the names of two algorithms for gen-
    erating association rules.
10. What is a classification problem in machine learning. Illustrate with an example.
11. Give three examples of classification problems from real life situations.
12. What is a discriminant in a classification problem?
13. List three machine learning algorithms for solving classification problems.
14. What is a binary classification problem? Explain with an example. Give also an example for
a classification problem which is not binary.
15. What is regression problem. What are the different types of regression?


#### CHAPTER 1. INTRODUCTION TO MACHINE LEARNING 14

(b) Long answer questions

1. Give a definition of the term “machine learning”. Explain with an example the concept of
    learning in the context of machine learning.
2. Describe the basic components of the machine learning process.
3. Describe in detail applications of machine learning in any three different knowledge domains.
4. Describe with an example the concept of association rule learning. Explain how it is made
    use of in real life situations.
5. What is the classification problem in machine learning? Describe three real life situations in
    different domains where such problems arise.
6. What is meant by a discriminant of a classification problem? Illustrate the idea with examples.
7. Describe in detail with examples the different types of learning like the supervised learning,
    etc.


Chapter 2

## Some general concepts

In this chapter we introduce some general concepts related to one of the simplest examples of su-
pervised learning, namely, the classification problem. We consider mainly binary classification
problems. In this context we introduce the concepts of hypothesis, hypothesis space and version
space. We conclude the chapter with a brief discussion on how to select hypothesis models and how
to evaluate the performance of a model.

### 2.1 Input representation

The general classification problem is concerned with assigning a class label to an unknown instance
from instances of known assignments of labels. In a real world problem, a given situation or an
object will have large number of features which may contribute to the assignment of the labels.
But in practice, not all these features may be equally relevant or important. Only those which are
significant need be considered as inputs for assigning the class labels. These features are referred to
as the “input features” for the problem. They are also said to constitute an “input representation”
for the problem.

Example

```
Consider the problem of assigning the label “family car” or “not family car” to cars. Let us
assume that the features that separate a family car from other cars are the price and engine
power. These attributes or features constitute the input representation for the problem. While
deciding on this input representation, we are ignoring various other attributes like seating
capacity or colour as irrelevant.
```
### 2.2 Hypothesis space

In the following discussions we consider only “binary classification” problems; that is, classification
problems with only two class labels. The class labels are usually taken as “ 1 ” and “ 0 ”. The label “1”
may indicate “True”, or “Yes”, or “Pass”, or any such label. The label “0” may indicate “False”, or
“No” or “Fail”, or any such label. The examples with class labels 1 are called “positive examples”
and examples with labels “0” are called “negative examples”.

2.2.1 Definition

1. Hypothesis
    In a binary classification problem, ahypothesisis a statement or a proposition purporting to
    explain a given set of facts or observations.

#### 15


#### CHAPTER 2. SOME GENERAL CONCEPTS 16

2. Hypothesis space
    Thehypothesis spacefor a binary classification problem is a set of hypotheses for the problem
    that might possibly be returned by it.
3. Consistency and satisfying
    Letxbe an example in a binary classification problem and letc(x)denote the class label
    assigned tox(c(x)is 1 or 0). LetDbe a set of training examples for the problem. Lethbe a
    hypothesis for the problem andh(x)be the class label assigned toxby the hypothesish.

```
(a) We say that the hypothesishisconsistentwith the set of training examplesDifh(x)=
c(x)for allx∈D.
(b) We say that an examplexsatisfiesthe hypothesishifh(x)= 1.
```
2.2.2 Examples

1. Consider the set of observations of a variablexwith the associated class labels given in Table
    2.1:

```
x 27 15 23 20 25 17 12 30 6 10
Class 1 0 1 1 1 0 0 1 0 0
```
```
Table 2.1: Sample data to illustrate the concept of hypotheses
```
```
Figure 2.1 shows the data plotted on thex-axis.
```
```
x
0 6 10 12 15 17 20 23 25 27 30
```
Figure 2.1: Data in Table 2.1 with hollow dots representing positive examples and solid dots repre-
senting negative examples

```
Looking at Figure 2.1, it appears that the class labeling has been done based on the following
rule.
h′ : IFx≥ 20 THEN “1” ELSE “0”. (2.1)
Note thath′is consistent with the training examples in Table 2.1. For example, we have:
```
```
h′( 27 )= 1 ; c( 27 )= 1 ; h′( 27 )=c( 27 )
h′( 15 )= 0 ; c( 15 )= 0 ; h′( 15 )=c( 15 )
```
```
Note also that, forx= 5 andx= 28 (not in training data),
```
```
h′( 5 )= 0 ; h′( 28 )= 1 :
```
```
The hypothesish′explains the data. The following proposition also explains the data:
```
```
h′′ : IFx≥ 19 THEN “0” ELSE “1”. (2.2)
```
```
It is not enough that the hypothesis explains the given data; it must also predict correctly the
class label of future observations. So we consider a set of such hypotheses and choose the
“best” one. The set of hypotheses can be defined using a parameter, saym, as given below:
```
```
hm : IFx≥mTHEN “1” ELSE ”0”. (2.3)
```

#### CHAPTER 2. SOME GENERAL CONCEPTS 17

```
The set of all hypotheses obtained by assigning different values tomconstitutes the hypothesis
spaceH; that is,
H={hm∶mis a real number}: (2.4)
```
```
For the same data, we can have different hypothesis spaces. For example, for the data in Table
2.1, we may also consider the hypothesis space defined by the following proposition:
```
```
h′m : IFx≤mTHEN “0” ELSE “1”.
```
2. Consider a situation with four binary variablesx 1 ,x 2 ,x 3 ,x 4 and one binary output variable
    y. Suppose we have the following observations.

```
x 1 x 2 x 3 x 4 y
0 0 0 1 1
0 1 0 1 0
1 1 0 0 1
0 0 1 0 0
```
```
The problem is to predict a functionfofx 1 ,x 2 ,x 3 ,x 4 which predicts the value ofyfor any
combination of values ofx 1 ,x 2 ,x 3 ,x 4. In this problem, the hypothesis space is the set of all
possible functionsf. It can be shown that the size of the hypothesis space is 2 (^2
```
(^4) )
= 65536.

3. Consider the problem of assigning the label “family car” or “not family car” to cars. For
    convenience, we shall replace the label “family car” by “1” and “not family car” by “0”.
    Suppose we choose the features “price (’000 $)” and “power (hp)” as the input representation
    for the problem. Further, suppose that there is some reason to believe that for a car to be a
    family car, its price and power should be in certain ranges. This supposition can be formulated
    in the form of the following proposition:

```
IF(p 1 <price<p 2 )AND(e 1 <power<e 2 )THEN “1” ELSE ”0” (2.5)
```
```
for suitable values ofp 1 ,p 2 ,e 1 ande 2. Since a solution to the problem is a proposition of the
form Eq.(2.5) with specific values forp 1 ,p 2 ,e 1 ande 2 , the hypothesis space for the problem
is the set of all such propositions obtained by assigning all possible values forp 1 ,p 2 ,e 1 and
e 2.
```
```
power (hp)
```
```
price (’000 $)
p 1 p 2
```
```
e 1
```
```
e 2
```
```
h(x 1 ;x 2 )= 1
```
```
x 1
```
```
x 2
```
```
hypothesish
```
```
Figure 2.2: An example hypothesis defined by Eq. (2.5)
```
```
It is interesting to observe that the set of points in the power–price plane which satisfies the
condition
(p 1 <price<p 2 )AND(e 1 <power<e 2 )
defines a rectangular region (minus the boundary) in the price–power space as shown in Figure
2.2. The sides of this rectangular region are parallel to the coordinate axes. Such a rectangle
```

#### CHAPTER 2. SOME GENERAL CONCEPTS 18

```
is called anaxis-aligned rectangleIfhis the hypothesis defined by Eq.(2.5), and(x 1 ;x 2 )
is any point in the price–power plane, thenh(x 1 ;x 2 )= 1 if and only if(x 1 ;x 2 )is within
the rectangular region. Hence we may identify the hypothesishwith the rectangular region.
Thus, the hypothesis space for the problem can be thought of as the set of all axis-aligned
rectangles in the price–power plane.
```
4. Consider the trading agent trying to infer which books or articles the user reads based on
    keywords supplied in the article. Suppose the learning agent has the following data (“1"
    indicates “True” and “0” indicates “False”):

```
article crime academic local music reads
a1 true false false true 1
a2 true false false false 1
a3 false true false false 0
a4 false false true false 0
a5 true true false false 1
```
```
The aim is to learn which articles the user reads. The aim is to find a definition such as
```
```
IF (crime OR (academic AND (NOT music))) THEN ”1” ELSE ”0”.
```
```
The hypothesis space H could be all boolean combinations of the input features or could be
more restricted, such as conjunctions or propositions defined in terms of fewer than three
features.
```
### 2.3 Ordering of hypotheses

Definition

LetXbe the set of all possible examples for a binary classification problem and leth′andh′′be
two hypotheses for the problem.

```
S′={x∈X∶h′(x)= 1 }
```
```
S′′={x∈X∶h′′(x)= 1 }
```
```
Figure 2.3: Hypothesish′is more general than hypothesish′′if and only ifS′′⊆S′
```
1. We say thath′ismore general thanh′′if and only if for everyx∈X, ifxsatisfiesh′′thenx
    satisfiesh′also; that is, ifh′′(x)= 1 thenh′(x)= 1 also. The relation “is more general than”
    defines a partial ordering relation in hypothesis space.
2. We say thath′ismore specificthanh′′, ifh′′is more general thanh′.
3. We say thath′isstrictly more general thanh′′ifh′is more general thanh′′andh′′isnot
    more general thanh′.
4. We say thath′isstrictly more specific thanh′′ifh′is more specific thanh′′andh′′isnot
    more specific thanh′.


#### CHAPTER 2. SOME GENERAL CONCEPTS 19

Example

```
Consider the hypothesesh′andh′′defined in Eqs.(2.1),(2.2). Then it is easy to check that if
h′(x)= 1 thenh′′(x)= 1 also. So,h′′is more general thanh′. But,h′is not more general
thanh′′and soh′′is strictly more general thanh′.
```
### 2.4 Version space

Definition

Consider a binary classification problem. LetDbe a set of training examples andHa hypothesis
space for the problem. Theversion spacefor the problem with respect to the setDand the spaceH
is the set of hypotheses fromHconsistent withD; that is, it is the set

```
VSD;H={h∈H∶h(x)=c(x)for allx∈D}:
```
2.4.1 Examples

Example 1

Consider the dataDgiven in Table 2.1 and the hypothesis space defined by Eqs.(2.3)-(2.4).

```
x
0 6 10 12 15 17 20 23 25 27 30
```
```
m
```
Figure 2.4: Values ofmwhich define the version space with data in Table 2.1 and hypothesis space
defined by Eq.(2.4)

From Figure 2.4 we can easily see that the hypothesis space with respect this datasetDand
hypothesis spaceHis as given below:

```
VSD;H={hm∶ 17 <m≤ 20 }:
```
Example 2

Consider the problem of assigning the label “family car” (indicated by “1”) or “not family car”
(indicated by “0”) to cars. Given the following examples for the problem and assuming that the
hypothesis space is as defined by Eq. (2.5), the version space for the problem.

```
x 1 : Price in ’000 ($) 32 82 44 34 43 80 38
x 2 : Power (hp) 170 333 220 235 245 315 215
Class 0 0 1 1 1 0 1
```
```
x 1 47 27 56 28 20 25 66 75
x 2 260 290 320 305 160 300 250 340
Class 1 0 0 0 0 0 0 0
```
Solution

Figure 2.5 shows a scatter plot of the given data. In the figure, the data with class label “1” (family
car) is shown as hollow circles and the data with class labels “0” (not family car) are shown as solid
dots.
A hypothesis as given by Eq.(2.5) with specific values for the parametersp 1 ,p 2 ,e 1 ande 2
specifies an axis-aligned rectangle as shown in Figure 2.2. So the hypothesis space for the problem
can be thought as the set of axis-aligned rectangles in the price-power plane.


#### CHAPTER 2. SOME GENERAL CONCEPTS 20

```
power (hp)
```
```
price (’000 $)
10 20 30 40 50 60 70 80 90
```
#### 150

#### 200

#### 250

#### 300

#### 350

Figure 2.5: Scatter plot of price-power data (hollow circles indicate positive examples and solid dots
indicate negative examples)

```
power (hp)
```
```
price (’000 $)
10 20 30 40 50 60 70 80 90
```
#### 150

#### 200

#### 250

#### 300

#### 350

#### ( 32 ; 170 )

#### ( 66 ; 250 )

#### ( 27 ; 290 )

#### ( 34 ; 235 )

#### ( 38 ; 215 )

#### ( 47 ; 260 )

Figure 2.6: The version space consists of hypotheses corresponding to axis-aligned rectangles con-
tained in the shaded region

The version space consists of all hypotheses specified by axis-aligned rectangles contained in
the shaded region in Figure 2.6. The inner rectangle is defined by

```
( 34 <price< 47 )AND( 215 <power< 260 )
```
and the outer rectangle is defined by

```
( 27 <price< 66 )AND( 170 <power< 290 ):
```
Example 3

Consider the problem of finding a rule for determining days on which one can enjoy water sport. The
rule is to depend on a few attributes like “temp”, ”humidity”, etc. Suppose we have the following
data to help us devise the rule. In the data, a value of “1” for “enjoy” means “yes” and a value of
“0” indicates ”no”.


#### CHAPTER 2. SOME GENERAL CONCEPTS 21

```
Example sky temp humidity wind water forecast enjoy
1 sunny warm normal strong warm same 1
2 sunny warm high strong warm same 1
3 rainy cold high strong warm change 0
4 sunny warm high strong cool change 1
```
Find the hypothesis space and the version space for the problem. (For a detailed discussion of this
problem see [4] Chapter2.)

Solution

We are required to find a rule of the following form, consistent with the data, as a solution of the
problem.

```
(sky=x 1 )∧(temp=x 2 )∧(humidity=x 3 )∧
(wind=x 4 )∧(water=x 5 )∧(forecast=x 6 )↔yes (2.6)
```
where

```
x 1 =sunny, warm,⋆
x 2 =warm, cold,⋆
x 3 =normal, high,⋆
x 4 =strong,⋆
x 5 =warm, cool,⋆
x 6 =same, change,⋆
```
(Here a “⋆” indicates other possible values of the attributes.) The hypothesis may be represented
compactly as a vector
(a 1 ;a 2 ;a 3 ;a 4 ;a 5 ;a 6 )

where, in the positions ofa 1 ;:::;a 6 , we write

- a “?” to indicate that any value is acceptable for the corresponding attribute,
- a ”∅” to indicate that no value is acceptable for the corresponding attribute,
- some specific single required value for the corresponding attribute

For example, the vector
(?, cold, high, ?, ?, ?)

indicates the hypothesis that one enjoys the sport only if “temp” is “cold” and “humidity” is “high”
whatever be the values of the other attributes.
It can be shown that the version space for the problem consists of the following six hypotheses
only:

```
(sunny, warm, ?, strong, ?, ?)
(sunny, ?, ?, strong, ?, ?)
(sunny, warm, ?, ?, ?, ?)
(?, warm, ?, strong, ?, ?)
(sunny, ?, ?, ?, ?, ?)
(?, warm, ?, ?, ?, ?)
```

#### CHAPTER 2. SOME GENERAL CONCEPTS 22

### 2.5 Noise

2.5.1 Noise and their sources

Noiseis any unwanted anomaly in the data ([2] p.25). Noise may arise due to several factors:

1. There may be imprecision in recording the input attributes, which may shift the data points in
    the input space.
2. There may be errors in labeling the data points, which may relabel positive instances as nega-
    tive and vice versa. This is sometimes called teacher noise.
3. There may be additional attributes, which we have not taken into account, that affect the label
    of an instance. Such attributes may be hidden or latent in that they may be unobservable. The
    effect of these neglected attributes is thus modeled as a random component and is included in
    “noise.”

2.5.2 Effect of noise

Noise distorts data. When there is noise in data, learning problems may not produce accurate results.
Also, simple hypotheses may not be sufficient to explain the data and so complicated hypotheses
may have to be formulated. This leads to the use of additional computing resources and the needless
wastage of such resources.
For example, in a binary classification problem with two variables, when there is noise, there
may not be a simple boundary between the positive and negative instances and to separate them. A
rectangle can be defined by four numbers, but to define a more complicated shape one needs a more
complex model with a much larger number of parameters. So, when there is noise, we may make a
complex model which makes a perfect fit to the data and attain zero error; or, we may use a simple
model and allow some error.

### 2.6 Learning multiple classes

So far we have been discussing binary classification problems. In a general case there may be more
than two classes. Two methods are generally used to handle such cases. These methods are known
by the names “one-against-all" and “one-against-one”.

2.6.1 Procedures for learning multiple classes

“One-against all” method

Consider the case where there areKclasses denoted byC 1 ;:::;CK. Each input instance belongs
to exactly one of them.
We view aK-class classification problem asKtwo-class problems. In thei-th two-class prob-
lem, the training examples belonging toCiare taken as the positive examples and the examples of
all other classes are taken as the negative examples. So, we have to findKhypothesesh 1 ;:::;hK
wherehiis defined by

```
hi(x)=
```
#### ⎧⎪

#### ⎪

#### ⎨

#### ⎪⎪

#### ⎩

```
1 ifxis in classCi
0 otherwise
```
For a givenx, ideally only one ofhi(x)is 1 and then we assign the classCitox. But, when
no, or, two or more,hi(x)is 1 , we cannot choose a class. In such a case, we say that the classifier
rejectssuch cases.


#### CHAPTER 2. SOME GENERAL CONCEPTS 23

“One-against-one” method

In theone-against-one(OAO) (also calledone-vs-one(OVO)) strategy, a classifier is constructed
for each pair of classes. If there areKdifferent class labels, a total ofK(K− 1 )~ 2 classifiers are
constructed. An unknown instance is classified with the class getting the most votes. Ties are broken
arbitrarily.
For example, let there be three classes,A,BandC. In the OVO method we construct 3 ( 3 −
1 )~ 2 = 3 binary classifiers. Now, if anyxis to be classified, we apply each of the three classifiers to
x. Let the three classifiers assign the classesA,B,Brespectively tox. Since a label toxis assigned
by the majority voting, in this example, we assign the class label ofBtox.

### 2.7 Model selection

As we have pointed earlier in Section 1.1.1, there is no universally accepted definition of the term
“model”. It may be understood as some mathematical expression or equation, or some mathematical
structures such as graphs and trees, or a division of sets into disjoint subsets, or a set of logical “if
:::then:::else:::” rules, or some such thing.
In order to formulate a hypothesis for a problem, we have to choose some model and the term
“model selection” has been used to refer to the process of choosing a model. However, the term has
been used to indicate several things. In some contexts it has been used to indicates the process of
choosing one particular approach from among several different approaches. This may be choosing
an appropriate algorithms from a selection of possible algorithms, or choosing the sets of features
to be used for input, or choosing initial values for certain parameters. Sometimes “model selection”
refers to the process of picking a particular mathematical model from among different mathematical
models which all purport to describe the same data set. It has also been described as the process of
choosing the right inductive bias.

2.7.1 Inductive bias

In a learning problem we only have the data. But data by itself is not sufficient to find the solution.
We should make some extra assumptions to have a solution with the data we have. The set of
assumptions we make to have learning possible is called theinductive biasof the learning algorithm.
One way we introduce inductive bias is when we assume a hypothesis class.

Examples

- In learning the class of family car, there are infinitely many ways of separating the positive
    examples from the negative examples. Assuming the shape of a rectangle is an inductive bias.
- In regression, assuming a linear function is an inductive bias.

The model selection is about choosing the right inductive bias.

2.7.2 Advantages of a simple model

Even though a complex model may not be making any errors in prediction, there are certain advan-
tages in using a simple model.

1. A simple model is easy to use.
2. A simple model is easy to train. It is likely to have fewer parameters.
    It is easier to find the corner values of a rectangle than the control points of an arbitrary shape.
3. A simple model is easy to explain.


#### CHAPTER 2. SOME GENERAL CONCEPTS 24

4. A simple model would generalize better than a complex model. This principle is known as
    Occam’s razor, which states that simpler explanations are more plausible and any unnecessary
    complexity should be shaved off.

Remarks

A model should not be too simple! With a small training set when the training instances differ a
little bit, we expect the simpler model to change less than a complex model: A simple model is thus
said to have lessvariance. On the other hand, a too simple model assumes more, is more rigid, and
may fail if indeed the underlying class is not that simple. A simpler model has more bias. Finding
the optimal model corresponds to minimizing both the bias and the variance.

### 2.8 Generalisation

How well a model trained on the training set predicts the right output for new instances is called
generalization.
Generalization refers to how well the concepts learned by a machine learning model apply to
specific examples not seen by the model when it was learning. The goal of a good machine learning
model is to generalize well from the training data to any data from the problem domain. This allows
us to make predictions in the future on data the model has never seen. Overfitting and underfitting
are the two biggest causes for poor performance of machine learning algorithms. The model should
be selected having the best generalisation. This is said to be the case if these problems are avoided.

- Underfitting
    Underfitting is the production of a machine learning model that is not complex enough to
    accurately capture relationships between a datasetâA ̆Zs features and a target variable. ́
- Overfitting
    Overfitting is the production of an analysis which corresponds too closely or exactly to a
    particular set of data, and may therefore fail to fit additional data or predict future observations
    reliably.

Example 1

```
(a) Given dataset (b) “Just right” model
```
```
(c) Underfitting model (d) Overfitting model
```
```
Figure 2.7: Examples for overfitting and overfitting models
```

#### CHAPTER 2. SOME GENERAL CONCEPTS 25

Consider a dataset shown in Figure 2.7(a). Let it be required to fit a regression model to the data. The
graph of a model which looks “just right” is shown in Figure 2.7(b). In Figure 2.7(c)we have a linear
regression model for the same dataset and this model does seem to capture the essential features of
the dataset. So this model suffers from underfitting. In Figure 2.7(d) we have a regression model
which corresponds too closely to the given dataset and hence it does not account for small random
noises in the dataset. Hence it suffers from overfitting.

Example 2

```
(a) Underfitting (b) Right fitting (c) Overfitting
```
```
Figure 2.8: Fitting a classification boundary
```
Suppose we have to determine the classification boundary for a dataset two class labels. An example
situation is shown in Figure 2.8 where the curved line is the classification boundary. The three figures
illustrate the cases of underfitting, right fitting and overfitting.

2.8.1 Testing generalisation: Cross-validation

We can measure the generalization ability of a hypothesis, namely, the quality of its inductive bias,
if we have access to data outside the training set. We simulate this by dividing the training set we
have into two parts. We use one part for training (that is, to find a hypothesis), and the remaining
part is called thevalidation setand is used to test the generalization ability. Assuming large enough
training and validation sets, the hypothesis that is the most accurate on the validation set is the best
one (the one that has the best inductive bias). This process is calledcross-validation.

### 2.9 Sample questions

(a) Short answer questions

1. Explain the general-to-specific ordering of hypotheses.
2. In the context of classification problems explain with examples the following: (i) hypothesis
    (ii) hypothesis space.
3. Define the version space of a binary classification problem.
4. Explain the “one-against-all” method for learning multiple classes.
5. Describe the “one-against-one” method for learning multiple classes.
6. What is meant by inductive bias in machine learning? Give an example.
7. What is meant by overfitting of data? Explain with an example.
8. What is meant by overfitting and underfitting of data with examples.


#### CHAPTER 2. SOME GENERAL CONCEPTS 26

(b) Long answer questions

1. Define version space and illustrate it with an example.
2. Given the following data

```
x 0 3 5 9 12 18 23
Label 0 0 0 1 1 1 1
```
```
and the hypothesis space
H={hmSma real number}
wherehmis defined by
IFx≤mTHEN 1 ELSE 0 ;
find the version space the problem with respect toDandH.
```
3. What is meant by “noise” in data? What are its sources and how it is affecting results?
4. Consider the following data:

```
x 2 3 5 8 10 15 16 18 20
y 12 15 10 6 8 10 7 9 10
Class label 0 0 1 1 1 1 0 0 0
```
```
Determine the version space if the hypothesis space consists of all hypotheses of the form
```
```
IF(x 1 <x<x 2 )AND(y 1 <y<y 2 )THEN “1” ELSE ”0”:
```
5. For the date in problem 4, what would be the version space if the hypothesis space consists of
    all hypotheses of the form

```
IF(x−x 1 )^2 +(y−y 1 )^2 ≤r^2 THEN “1” ELSE ”0”:
```
6. What issues are to be considered while selecting a model for applying machine learning in a
    given problem.


Chapter 3

## VC dimension and PAC learning

The concepts of Vapnik-Chervonenkis dimension (VC dimension) and probably approximate correct
(PAC) learning are two important concepts in the mathematical theory of learnability and hence are
mathematically oriented. The former is a measure of the capacity (complexity, expressive power,
richness, or flexibility) of a space of functions that can be learned by a classification algorithm.
It was originally defined by Vladimir Vapnik and Alexey Chervonenkis in 1971. The latter is a
framework for the mathematical analysis of learning algorithms. The goal is to check whether the
probability for a selected hypothesis to be approximately correct is very high. The notion of PAC
learning was proposed by Leslie Valiant in 1984.

### 3.1 Vapnik-Chervonenkis dimension

LetHbe the hypothesis space for some machine learning problem. The Vapnik-Chervonenkis
dimension ofH, also called the VC dimension ofH, and denoted byV C(H), is a measure of the
complexity (or, capacity, expressive power, richness, or flexibility) of the spaceH. To define the VC
dimension we require the notion of the shattering of a set of instances.

3.1.1 Shattering of a set

LetDbe a dataset containingNexamples for a binary classification problem with class labels 0
and 1. LetHbe a hypothesis space for the problem. Each hypothesishinHpartitionsDinto two
disjoint subsets as follows:

```
{x∈DSh(x)= 0 }and{x∈DSh(x)= 1 }:
```
Such a partition ofSis called a “dichotomy” inD. It can be shown that there are 2 Npossible
dichotomies inD. To each dichotomy ofDthere is a unique assignment of the labels “1” and “0”
to the elements ofD. Conversely, ifSis any subset ofDthen,Sdefines a unique hypothesishas
follows:

```
h(x)=
```
#### ⎧⎪

#### ⎪

#### ⎨

#### ⎪⎪

#### ⎩

```
1 ifx∈S
0 otherwise
```
Thus to specify a hypothesish, we need only specify the set{x∈DSh(x)= 1 }.
Figure 3.1 shows all possible dichotomies ofDifDhas three elements. In the figure, we have
shown only one of the two sets in a dichotomy, namely the set{x∈DSh(x)= 1 }. The circles and
ellipses represent such sets.

#### 27


#### CHAPTER 3. VC DIMENSION AND PAC LEARNING 28

```
a
```
```
b c
```
```
a
```
```
b c
```
```
a
```
```
b c
```
```
a
```
```
b c
(i) Emty set (ii) (iii) (iv)
```
```
a
```
```
b c
```
```
a
```
```
b c
```
```
a
```
```
b c
```
```
a
```
```
b c
```
```
(v) (vi) (vii) (viii) Full setD
```
```
Figure 3.1: Different forms of the set{x∈S∶h(x)= 1 }forD={a;b;c}
```
We require the notion of a hypothesis consistent with a set of examples introduced in Section 2.4
in the following definition.

Definition

A set of examplesDis said to beshatteredby a hypothesis spaceHif and only if for every di-
chotomy ofDthere exists some hypothesis inHconsistent with the dichotomy ofD.

3.1.2 Vapnik-Chervonenkis dimension

The following example illustrates the concept of Vapnik-Chervonenkis dimension.

Example

Let the instance spaceXbe the set of all real numbers. Consider the hypothesis space defined by
Eqs.(2.3)-(2.4):
H={hm∶mis a real number};

where
hm ∶ IFx≥mTHEN ”1” ELSE “0”:

```
i) LetDbe a subset ofXcontaining only a single number, say,D={ 3 : 5 }. There are 2
dichotomies for this set. These correspond to the following assignment of class labels:
```
```
x 3.25
Label 0
```
```
x 3.25
Label 1
```
```
h 4 ∈His consistent with the former dichotomy andh 3 ∈His consistent with the latter. So,
to every dichotomy inDthere is a hypothesis inHconsistent with the dichotomy. Therefore,
the setDis shattered by the hypothesis spaceH.
```
```
ii) LetDbe a subset ofXcontaining two elements, say,D={ 3 : 25 ; 4 : 75 }. There are 4 di-
chotomies inDand they correspond to the assignment of class labels shown in Table 3.1.
```
```
In these dichotomies,h 5 is consistent with (a),h 4 is consistent with (b) andh 3 is consistent
with (d). But there is no hypothesishm∈Hconsistent with (c). Thus the two-element setD
is not shattered byH. In a similar way it can be shown that there is no two-element subset
ofXwhich is shattered byH.
```
It follows that the size of the largest finite subset ofXshattered byHis 1. This number is the
VC dimension ofH.


#### CHAPTER 3. VC DIMENSION AND PAC LEARNING 29

```
x 3.25 4.75
Label 0 0
```
```
x 3.25 4.75
Label 0 1
(a) (b)
```
```
x 3.25 4.75
Label 1 0
```
```
x 3.25 4.75
Label 1 1
(c) (d)
```
```
Table 3.1: Different assignments of class labels to the elements of{ 3 : 25 ; 4 : 75 }
```
Definition

TheVapnik-Chervonenkis dimension(VC dimension) of a hypothesis spaceHdefined over an in-
stance space (that is, the set of all possible examples)X, denoted byV C(H), is the size of the
largest finite subset ofXshattered byH. If arbitrarily large subsets ofXcan be shattered byH,
then we defineV C(H)=∞.

Remarks

It can be shown thatV C(H)≤log 2 (SHS)whereHis the number of hypotheses inH.

3.1.3 Examples

1. LetXbe the set of all real numbers (say, for example, the set of heights of people). For any
    real numbersaandbdefine a hypothesisha;bas follows:

```
ha;b(x)=
```
#### ⎧⎪

#### ⎪

#### ⎨

#### ⎪⎪

#### ⎩

```
1 ifa<x<b
0 otherwise
```
```
Let the hypothesis spaceHconsist of all hypotheses of the formha;b. We show thatV C(H)=
2. We have to show that there is a subset ofXof size 2 shattered byHand there is no subset
of size 3 shattered byH.
```
- Consider the two-element setD={ 3 : 25 ; 4 : 75 }. The various dichotomies ofDare
    given in Table 3.1. It can be seen that the hypothesish 5 ; 6 is consistent with (a),h 4 ; 5 is
    consistent with (b),h 3 ; 4 is consistent with (c) andh 3 ; 5 is consistent with (d). So the set
    Dis shattered byH.
- Consider a three-element subsetD={x 1 ;x 2 ;x 3 }. Let us assume thatx 1 <x 2 <x 3 .H
    cannot shatter this subset because the dichotomy represented by the set{x 1 ;x 3 }cannot
    be represented by a hypothesis inH(any interval containing bothx 1 andx 3 will contain
    x 2 also).

```
Therefore, the size of the largest subset ofXshattered byHis 2 and soV C(H)= 2.
```
2. Let the instance spaceXbe the set of all points(x;y)in a plane. For any three real numbers,
    a;b;cdefine a class labeling as follows:

```
ha;b;c(x;y)=
```
#### ⎧⎪

#### ⎪

#### ⎨

#### ⎪⎪

#### ⎩

```
1 ifax+by+c> 0
0 otherwise
```

#### CHAPTER 3. VC DIMENSION AND PAC LEARNING 30

#### O

```
x
```
```
y
```
```
ha;b;c(x;y)= 0
ax+by+c= 0
(assumec< 0 )
```
```
ha;b;c(x;y)= 0
ax+by+c< 0
```
```
ha;b;c(x;y)= 1
ax+by+c> 0
```
```
Figure 3.2: Geometrical representation of the hypothesisha;b;c
```
```
LetHbe the set of all hypotheses of the formha;b;c. We show thatV C(H)= 3. We have
show that there is a subset of size 3 shattered byHand there is no subset of size 4 shattered
byH.
```
- Consider a setD={A;B;C}of three non-collinear points in the plane. There are 8 sub-
    sets ofDand each of these defines a dichotomy ofD. We can easily find 8 hypotheses
    corresponding to the dichotomies defined by these subsets (see Figure 3.3).

#### A

#### B C

```
Figure 3.3: A hypothesisha;b;cconsistent with the dichotomy defined by the subset
{A;C}of{A;B;C}
```
- Consider a setS={A;B;C;D}of four points in the plane. Let no three of these points
    be collinear. Then, the points form a quadrilateral. It can be easily seen that, in this case,
    there is no hypothesis for which the two element set formed by the ends of a diagonal is
    the corresponding dichotomy (see Figure 3.4).

```
A
```
#### B C

#### D

```
Figure 3.4: There is no hypothesisha;b;cconsistent with the dichotomy defined by the
subset{A;C}of{A;B;C;D}
```
```
So the set cannot be shattered byH. If any three of them are collinear, then by some
trial and error, it can be seen that in this case also the set cannot be shattered byH. No
set with four elements cannot be shattered byH.
From the above discussion we conclude thatV C(H)= 3.
```
3. LetXbe set of all conjunctions ofnboolean literals. Let the hypothesis spaceHconsists of
    conjunctions of up tonliterals. It can be shown thatV C(H)=n. (The full details of the
    proof of this is beyond the scope of these notes.)


#### CHAPTER 3. VC DIMENSION AND PAC LEARNING 31

### 3.2 Probably approximately correct learning

In computer science,computational learning theory(or justlearning theory) is a subfield of artificial
intelligence devoted to studying the design and analysis of machine learning algorithms. In compu-
tational learning theory,probably approximately correct learning(PAC learning) is a framework for
mathematical analysis of machine learning algorithms. It was proposed in 1984 by Leslie Valiant.
In this framework, the learner (that is, the algorithm) receives samples and must select a hypoth-
esis from a certain class of hypotheses. The goal is that, with high probability (the “probably” part),
the selected hypothesis will have low generalization error (the “approximately correct” part).
In this section we first give an informal definition of PAC-learnability. After introducing a few
nore notions, we give a more formal, mathematically oriented, definition of PAC-learnability. At the
end, we mention one of the applications of PAC-learnability.

3.2.1 PAC-learnability

To define PAC-learnability we require some specific terminology and related notations.

- LetXbe a set called theinstance spacewhich may be finite or infinite. For example,Xmay
    be the set of all points in a plane.
- Aconcept classCforXis a family of functionsc∶X→{ 0 ; 1 }. A member ofCis called a
    concept. A concept can also be thought of as a subset ofX. IfCis a subset ofX, it defines a
    unique functionC∶X→{ 0 ; 1 }as follows:

```
C(x)=
```
#### ⎧⎪

#### ⎪

#### ⎨

#### ⎪⎪

#### ⎩

```
1 ifx∈C
0 otherwise
```
- Ahypothesishis also a functionh∶X→{ 0 ; 1 }. So, as in the case of concepts, a hypothesis
    can also be thought of as a subset ofX.Hwill denote a set of hypotheses.
- We assume thatFis an arbitrary, but fixed,probability distributionoverX.
- Training examples are obtained by taking random samples fromX. We assume that the
    samples are randomly generated fromXaccording to the probability distributionF.

Now, we give below an informal definition of PAC-learnability.

Definition (informal)

LetXbe an instance space,Ca concept class forX,ha hypothesis inCandFan arbitrary,
but fixed, probability distribution. The concept classCis said to bePAC-learnableif there is an
algorithmAwhich, for samples drawn with any probability distributionFand any conceptc∈C,
will with high probability produce a hypothesish∈Cwhose error is small.

Additional notions

- True error
    To formally define PAC-learnability, we require the concept of the true error of a hypothesis
    hwith respect to a target conceptcdenoted by errorF(h). It is defined by

```
errorF(h)=Px∈F(h(x)≠c(x))
```
```
where the notationPx∈Findicates that the probability is taken forxdrawn fromXaccording
to the distributionF. This error is the probability thathwill misclassify an instancexdrawn
at random fromXaccording to the distributionF. This error is not directly observable to the
learner; it can only see the training error of each hypothesis (that is, how oftenh(x)≠c(x)
over training instances).
```

#### CHAPTER 3. VC DIMENSION AND PAC LEARNING 32

- Length or dimension of an instance
    We require the notion of the length or dimension or size of an instance in the instance spaceX.
    If the instance spaceXis then-dimensional Euclidean space, then each example is specified
    bynreal numbers and so the length of the examples may be taken asn. Similarly, ifXis the
    space of the conjunctions ofnBoolean literals, then the length of the examples may be taken
    asn. These are the commonly considered instance spaces in computational learning theory.
- Size of a concept
    We need the notion of the size of a conceptc. For any conceptc, we define size(c)to be the
    size of the smallest representation ofcusing some finite alphabet.

(For a detailed discussion of these and related ideas, see [6] pp.7-15.)

Definition ([4] p.206)

Consider a concept classCdefined over a set of instancesXof lengthnand a learner (algorithm)L
using hypothesis spaceH.Cis said to be PAC-learnable byLusingHif for allc∈C, distribution
FoverX,such that 0 << 1 ~ 2 andsuch that 0 << 1 ~ 2 , learnerLwill with probability at least
( 1 −)output a hypothesishsuch that errorF(h)≤, in time that is polynomial in 1 ~, 1 ~,nand
size(c).

3.2.2 Examples

To illustrate the definition of PAC-learnability, let us consider some concrete examples.
y

```
x
a b
```
```
c
```
```
d
```
```
(x;y)
```
```
x
```
```
y
```
```
concept/hypothesis
```
```
Figure 3.5: An axis-aligned rectangle in the Euclidean plane
```
Example 1

- Let the instance space be the setXof all points in the Euclidean plane. Each point is repre-
    sented by its coordinates(x;y). So, the dimension or length of the instances is 2.
- Let the concept classCbe the set of all “axis-aligned rectangles” in the plane; that is, the set
    of all rectangles whose sides are parallel to the coordinate axes in the plane (see Figure 3.5).
- Since an axis-aligned rectangle can be defined by a set of inequalities of the following form
    having four parameters
       a≤x≤b; c≤y≤d
    the size of a concept is 4.
- We take the setHof all hypotheses to be equal to the setCof concepts,H=C.


#### CHAPTER 3. VC DIMENSION AND PAC LEARNING 33

- Given a set of sample points labeled positive or negative, letLbe the algorithm which outputs
    the hypothesis defined by the axis-aligned rectangle which gives the tightest fit to the posi-
    tive examples (that is, that rectangle with the smallest area that includes all of the positive
    examples and none of the negative examples) (see Figure 3.6).
       y

```
x
```
```
Figure 3.6: Axis-aligned rectangle which gives the tightest fit to the positive examples
```
It can be shown that, in the notations introduced above, the concept classCis PAC-learnable by
the algorithmLusing the hypothesis spaceHof all axis-aligned rectangles.

Example 2

- LetXthe set of alln-bit strings. Eachn-bit string may be represented by an orderedn-tuple
    (a 1 ;:::;an)where eachaiis either 0 or 1. This may be thought of as an assignment of 0 or
    1 tonboolean variablesx 1 ;:::;xn. The setXis sometimes denoted by{ 0 ; 1 }n.
- To define the concept class, we distinguish certain subsets ofXin a special way. By a literal
    we mean, a Boolean variablexior its negation~xi. We consider conjunctions of literals over
    x 1 ;:::;xn. Each conjunction defines a subset ofX. for example, the conjunctionx 1 ∧x~ 2 ∧x 4
    defines the following subset ofX:

```
{a=(a 1 ;:::;an)∈XSa 1 = 1 ;a 2 = 0 ;a 4 = 1 }
```
```
The concept classCconsists of all subsets ofXdefined by conjunctions of Boolean literals
overx 1 ;:::;xn.
```
- The hypothesis classHis defined as equal to the concept classC.
- LetLbe a certain algorithm called “Find-S algorithm” used to find a most specific hypothesis
    (see [4] p.26).

The concept classCof all subsets ofX={ 0 ; 1 }ndefined by conjunctions of Boolean literals
overx 1 ;:::;xnis PAC-learnable by the Find-S algorithm using the hypothesis spaceH=C.

3.2.3 Applications

To make the discussions complete, we introduce one simple application of the PAC-learning theory.
The application is the derivation of a mathematical expression to estimate the size of samples that
would produce a hypothesis with a given high probability and which has a generalization error of
given low probability.
We use the following assumptions and notations:

- We assume that the hypothesis spaceHisfinite. LetSHSdenote the number of elements inH.


#### CHAPTER 3. VC DIMENSION AND PAC LEARNING 34

- We assume that the concept classCbe equal toH.
- Letmbe the number of elements in the set of samples.
- Letandbe such that 0 <;< 1.
- The algorithm can be anyconsistent algorithm, that is, any algorithm which correctly classifies
    the training examples.

```
It can be shown that, ifmis chosen such that
```
```
m≥
```
#### 1

#### 

```
(ln(SHS)+ln( 1 ~))
```
then any consistent algorithm will successfully produce any concept inHwith probability( 1 −)
and with an error having a maximum probability of.

### 3.3 Sample questions

(a) Short answer questions

1. What is VC dimension?
2. Explain Vapnik-Chervonenkis dimension.
3. Give an informal definition of PAC learnability.
4. Give a precise definition of PAC learnability.
5. Give an application of PAC learnable algorithm.

(b) Long answer questions

1. LetXbe the set of all real numbers. Describe a hypothesis forXfor which the VC dimension
    is 0.
2. LetXbe the set of all real numbers. Describe a hypothesis forXfor which the VC dimension
    is 1.
3. LetXbe the set of all real numbers. Describe a hypothesis forXfor which the VC dimension
    is 2. Describe an example for which the VC dimension is 3.
4. Describe an example of a PAC learnable concept class.
5. An open interval inRis defined as(a;b)={x∈RSa<x<b}. It has two parametersaandb.
    Show that the sets of all open intervals has a VC dimension of 2.


Chapter 4

## Dimensionality reduction

The complexity of any classifier or regressor depends on the number of inputs. This determines both
the time and space complexity and the necessary number of training examples to train such a clas-
sifier or regressor. In this chapter, we discuss various methods for decreasing input dimensionality
without losing accuracy.

### 4.1 Introduction

In many learning problems, the datasets have large number of variables. Sometimes, the number
of variables is more than the number of observations. For example, such situations have arisen in
many scientific fields such as image processing, mass spectrometry, time series analysis, internet
search engines, and automatic text analysis among others. Statistical and machine learning methods
have some difficulty when dealing with such high-dimensional data. Normally the number of input
variables is reduced before the machine learning algorithms can be successfully applied.
In statistical and machine learning, dimensionality reduction or dimension reduction is the pro-
cess of reducing the number of variables under consideration by obtaining a smaller set of principal
variables.
Dimensionality reduction may be implemented in two ways.

- Feature selection
    In feature selection, we are interested in findingkof the total ofnfeatures that give us the
    most information and we discard the other(n−k)dimensions. We are going to discusssubset
    selectionas a feature selection method.
- Feature extraction
    In feature extraction, we are interested in finding a new set ofkfeatures that are the combina-
    tion of the originalnfeatures. These methods may be supervised or unsupervised depending
    on whether or not they use the output information. The best known and most widely used
    feature extraction methods arePrincipal Components Analysis(PCA) andLinear Discrimi-
    nant Analysis(LDA), which are both linear projection methods, unsupervised and supervised
    respectively.

Measures of error

In both methods we require a measure of the error in the model.

- In regression problems, we may use theMean Squared Error(MSE) or theRoot Mean
    Squared Error(RMSE) as the measure of error. MSE is the sum, over all the data points,
    of the square of the difference between the predicted and actual target variables, divided by

#### 35


#### CHAPTER 4. DIMENSIONALITY REDUCTION 36

```
the number of data points. Ify 1 ;:::;ynare the observed values andy^i;:::;y^nare the pre-
dicted values, then
MSE=
```
#### 1

```
n
```
```
n
Q
i= 1
```
```
(yi−^yi)^2
```
- In classification problems, we may use themisclassification rateas a measure of the error.
    This is defined as follows:

```
misclassification rate=
no. of misclassified examples
total no. of examples
```
### 4.2 Why dimensionality reduction is useful

There are several reasons why we are interested in reducing dimensionality.

- In most learning algorithms, the complexity depends on the number of input dimensions, d,
    as well as on the size of the data sample, N, and for reduced memory and computation, we
    are interested in reducing the dimensionality of the problem. Decreasing d also decreases the
    complexity of the inference algorithm during testing.
- When an input is decided to be unnecessary, we save the cost of extracting it.
- Simpler models are more robust on small datasets. Simpler models have less variance, that is,
    they vary less depending on the particulars of a sample, including noise, outliers, and so forth.
- When data can be explained with fewer features, we get a better idea about the process that
    underlies the data, which allows knowledge extraction.
- When data can be represented in a few dimensions without loss of information, it can be
    plotted and analyzed visually for structure and outliers.

### 4.3 Subset selection

In machine learningsubset selection, sometimes also calledfeature selection, orvariable selection,
orattribute selection, is the process of selecting a subset of relevant features (variables, predictors)
for use in model construction.
Feature selection techniques are used for four reasons:

- simplification of models to make them easier to interpret by researchers/users
- shorter training times,
- to avoid the curse of dimensionality
- enhanced generalization by reducing overfitting

The central premise when using a feature selection technique is that the data contains many
features that are either redundant or irrelevant, and can thus be removed without incurring much loss
of information.
There are several approaches to subset selection. In these notes, we discuss two of the simplest
approaches known as forward selection and backward selection methods.

4.3.1 Forward selection

Inforward selection, we start with no variables and add them one by one, at each step adding the one
that decreases the error the most, until any further addition does not decrease the error (or decreases
it only sightly).


#### CHAPTER 4. DIMENSIONALITY REDUCTION 37

Procedure

We use the following notations:
n : number of input variables
x 1 ;:::;xn : input variables
Fi : a subset of the set of input variables
E(Fi) : error incurred on the validation sample when only the inputs
inFiare used

1. SetF 0 =∅andE(F 0 )=∞.
2. Fori= 0 ; 1 ;:::, repeat the following untilE(Fi+ 1 )≥E(Fi):

```
(a) For all possible input variablesxj, train the model with the input variablesFi∪{xj}and
calculateE(Fi∪{xj})on the validation set.
(b) Choose that input variablexmthat causes the least errorE(Fi∪{xj}):
```
```
m=arg min
j
E(Fi∪{xj})
```
```
(c) SetFi+ 1 =Fi∪{xm}.
```
3. The setFiis outputted as the best subset.

Remarks

1. In this procedure, we stop if adding any feature does not decrease the errorE. We may
    even decide to stop earlier if the decrease in error is too small, where there is a user-defined
    threshold that depends on the application constraints.
2. This process may be costly because to decrease the dimensions fromntok, we need to train
    and test the system
       n+(n−l)+(n− 2 )+ ⋯ +(n−k)
    times, which isO(n^2 ).

4.3.2 Backward selection

In sequential backward selection, we start with the set containing all features and at each step remove
the one feature that causes the least error.

Procedure

We use the following notations:
n : number of input variables
x 1 ;:::;xn : input variables
Fi : a subset of the set of input variables
E(Fi) : error incurred on the validation sample when only the inputs
inFiare used

1. SetF 0 ={x 1 ;:::;xn}andE(F 0 )=∞.
2. Fori= 0 ; 1 ;:::, repeat the following untilE(Fi+ 1 )≥E(Fi):

```
(a) For all possible input variablesxj, train the model with the input variablesFi−{xj})
and calculateE(Fi−{xj})on the validation set.
(b) Choose that input variablexmthat causes the least errorE(Fi−{xj}):
```
```
m=arg min
j
E(Fi−{xj})
```

#### CHAPTER 4. DIMENSIONALITY REDUCTION 38

```
(c) SetFi+ 1 =Fi−{xm}.
```
3. The setFiis outputted as the best subset.

### 4.4 Principal component analysis

Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transforma-
tion to convert a set of observations of possibly correlated variables into a set of values of linearly
uncorrelated variables called principal components. The number of principal components is less
than or equal to the smaller of the number of original variables or the number of observations. This
transformation is defined in such a way that the first principal component has the largest possible
variance (that is, accounts for as much of the variability in the data as possible), and each succeeding
component in turn has the highest variance possible under the constraint that it is orthogonal to the
preceding components.

4.4.1 Graphical illustration of the idea

Consider a two-dimensional data, that is, a dataset consisting of examples having two features. Let
each of the features be numeric data. So, each example can be plotted on a coordinate plane (x-
coordinate indicating the first feature andy-coordinate indicating the second feature). Plotting the
example, we get a scatter diagram of the data. Now let us examine some typical scatter diagram
and make some observations regarding the directions in which the points in the scatter diagram are
spread out.
Let us examine the figures in Figure 4.1.

```
(i) Figure 4.1a shows a scatter diagram of a two-dimensional data.
```
```
(ii) Figure 4.1b shows spread of the data in thexdirection and Figure 4.1c shows the spread of
the data in they-direction. We note that the spread in thex-direction is more than the spread
in theydirection.
```
```
(iii) Examining Figures 4.1d and 4.1e, we note that the maximum spread occurs in the direction
shown in Figure 4.1e. Figure 4.1e also shows the point whose coordinates are the mean
values of the two features in the dataset. This direction is called thedirection of the first
principal componentof the given dataset.
```
```
(iv) The direction which is perpendicular (orthogonal) to the direction of the first principal com-
ponent is called thedirection of the second principal componentof the dataset. This direc-
tion is shown in Figure 4.1f. (This is only with reference to a two-dimensional dataset.)
```
```
(v) The unit vectors along the directions of principal components are called theprincipal com-
ponent vectors, or simply,principal components. These are shown in Figure 4.1g.
```
Remark

let us consider a dataset consisting of examples with three or more features. In such a case, we have
ann-dimensional dataset withn≥ 3. In this case, the first principal component is defined exactly as
in item iii above. But, for the second component, it may be noted that there would be many directions
perpendicular to the direction of the first principal component. The direction of the second principal
component is that direction, which is perpendicular to the first principal component, in which the
spread of data is largest. The third and higher order principal components are constructed in a similar
way.


#### CHAPTER 4. DIMENSIONALITY REDUCTION 39

```
(a) Scatter diagram (b) Spread alongx-direction
```
```
(c) Spread alongy-direction (d) Largest spread
```
(e) Direction of largest spread : Direction of the first
principal component (solid dot is the point whose coor-
dinates are the means ofxandy)

```
(f) Directions of principal components
```
```
(g) Principal component vectors (unit vectors in the di-
rections of principal components)
```
```
Figure 4.1: Principal components
```
A warning!

The graphical illustration of the idea of PCA as explained above is slightly misleading. For the sake
of simplicity and easy geometrical representation, in the graphical illustration we have usedrange
as the measure of spread. The direction of the first principal component was taken as the direction of
maximum range. But, due to theoretical reasons, in the implementation of PCA in practice, it is the
variance that is taken as as the measure of spread. The first principal component is the the direction
in which the variance is maximum.


#### CHAPTER 4. DIMENSIONALITY REDUCTION 40

4.4.2 Computation of the principal component vectors

(PCA algorithm)

The following is an outline of the procedure for performing a principal component analysis on a
given data. The procedure is heavily dependent on mathematical concepts. A knowledge of these
concepts is essential to carry out this procedure.

Step 1. Data

```
We consider a dataset havingnfeatures or variables denoted byX 1 ;X 2 ;:::;Xn. Let there
beNexamples. Let the values of thei-th featureXibeXi 1 ;Xi 2 ;:::;XiN(see Table 4.1).
```
```
Features Example 1 Example 2 ⋯ ExampleN
X 1 X 11 X 12 ⋯ X 1 N
X 2 X 21 X 22 ⋯ X 2 N
⋮
Xi Xi 1 Xi 2 ⋯ XiN
⋮
Xn Xn 1 Xn 2 ⋯ XnN
```
```
Table 4.1: Data for PCA algorithm
```
Step 2. Compute the means of the variables

```
We compute the meanXiof the variableXi:
```
```
Xi=^1
N
```
```
(Xi 1 +Xi 2 + ⋯ +XiN):
```
Step 3. Calculate the covariance matrix

```
Consider the variablesXiandXj(iandjneed not be different). The covariance of the
ordered pair(Xi;Xj)is defined as^1
```
```
Cov(Xi;Xj)=
```
#### 1

#### N− 1

```
N
Q
k= 1
```
```
(Xik−Xi)(Xjk−Xj): (4.1)
```
```
We calculate the followingn×nmatrixScalled the covariance matrix of the data. The
element in thei-th rowj-th column is the covariance Cov(Xi;Xj):
```
#### S=

#### ⎡⎢

#### ⎢

#### ⎢⎢

#### ⎢⎢

#### ⎢⎢

#### ⎣

```
Cov(X 1 ;X 1 ) Cov(X 1 ;X 2 ) ⋯ Cov(X 1 ;Xn)
Cov(X 2 ;X 1 ) Cov(X 2 ;X 2 ) ⋯ Cov(X 2 ;Xn)
⋮
Cov(Xn;X 1 ) Cov(Xn;X 2 ) ⋯ Cov(Xn;Xn)
```
#### ⎤⎥

#### ⎥

#### ⎥⎥

#### ⎥⎥

#### ⎥⎥

#### ⎦

Step 4. Calculate the eigenvalues and eigenvectors of the covariance matrix

```
LetSbe the covariance matrix and letIbe the identity matrix having the same dimension
as the dimension ofS.
```
```
i) Set up the equation:
det(S−I)= 0 : (4.2)
This is a polynomial equation of degreenin. It hasnreal roots (some of the
roots may be repeated) and these roots are the eigenvalues ofS. We find thenroots
 1 ; 2 ;:::;nof Eq. (4.2).
```
(^1) There is an alternative definition of covariance. In this definition, covariance is defined as in Eq. (4.1) withN− 1
replaced byN. There are certain theoretical reasons for adopting the definition as given here.


#### CHAPTER 4. DIMENSIONALITY REDUCTION 41

```
ii) If=′is an eigenvalue, then the corresponding eigenvector is a vector
```
#### U=

#### ⎡⎢

#### ⎢

#### ⎢⎢

#### ⎢⎢

#### ⎢

#### ⎢⎣

```
u 1
u 2
⋮
un
```
#### ⎤⎥

#### ⎥

#### ⎥⎥

#### ⎥⎥

#### ⎥

#### ⎥⎦

```
such that
(S−′I)U= 0 :
(This is a system ofnhomogeneous linear equations inu 1 ,u 2 ,:::,unand it al-
ways has a nontrivial solution.) We next find a set ofnorthogonal eigenvectors
U 1 ;U 2 ;:::;Unsuch thatUiis an eigenvector corresponding toi.^2
iii) We now normalise the eigenvectors. Given any vectorXwe normalise it by dividing
Xby its length. The length (or, the norm) of the vector
```
#### X=

#### ⎡⎢

#### ⎢⎢

#### ⎢⎢

#### ⎢⎢

#### ⎢

#### ⎣

```
x 1
x 2
⋮
xn
```
#### ⎤⎥

#### ⎥⎥

#### ⎥⎥

#### ⎥⎥

#### ⎥

#### ⎦

```
is defined as
SSXSS=
```
#### ¼

```
x^21 +x^22 + ⋯ +x^2 n:
Given any eigenvectorU, the corresponding normalised eigenvector is computed as
```
```
1
SSUSS
```
#### U:

```
We compute thennormalised eigenvectorse 1 ;e 2 ;:::;enby
```
```
ei=
```
#### 1

```
SSUiSS
```
```
Ui; i= 1 ; 2 ;:::;n:
```
Step 5. Derive new data set

```
Order the eigenvalues from highest to lowest. The unit eigenvector corresponding to the
largest eigenvalue is the first principal component. The unit eigenvector corresponding to
the next highest eigenvalue is the second principal component, and so on.
```
```
i) Let the eigenvalues in descending order be 1 ≥ 2 ≥:::≥nand let the corre-
sponding unit eigenvectors bee 1 ;e 2 ;:::;en.
ii) Choose a positive integerpsuch that 1 ≤p≤n.
iii) Choose the eigenvectors corresponding to the eigenvalues 1 , 2 ,:::,pand form
the followingp×nmatrix (we write the eigenvectors as row vectors):
```
#### F=

#### ⎡⎢

#### ⎢

#### ⎢⎢

#### ⎢⎢

#### ⎢

#### ⎢⎣

```
eT 1
eT 2
⋮
eTp
```
#### ⎤⎥

#### ⎥

#### ⎥⎥

#### ⎥⎥

#### ⎥

#### ⎥⎦

#### ;

```
whereTin the superscript denotes the transpose.
```
(^2) Fori≠j, the vectorsUiandUjare orthogonal meansUT
iUj=^0 whereTdenotes the transpose.


#### CHAPTER 4. DIMENSIONALITY REDUCTION 42

```
iv) We form the followingn×Nmatrix:
```
#### X=

#### ⎡⎢

#### ⎢

#### ⎢⎢

#### ⎢⎢

#### ⎢

#### ⎢⎣

#### X 11 −X 1 X 12 −X 1 ⋯ X 1 N−X 1

#### X 21 −X 2 X 22 −X 2 ⋯ X 2 N−X 2

#### ⋮

```
Xn 1 −Xn Xn 2 −Xn ⋯ XnN−Xn
```
#### ⎤⎥

#### ⎥

#### ⎥⎥

#### ⎥⎥

#### ⎥

#### ⎥⎦

```
v) Next compute the matrix:
Xnew=FX:
Note that this is ap×Nmatrix. This gives us a dataset ofNsamples havingp
features.
```
Step 6. New dataset

```
The matrixXnewis the new dataset. Each row of this matrix represents the values of a
feature. Since there are onlyprows, the new dataset has only features.
```
Step 7. Conclusion

```
This is how the principal component analysis helps us in dimensional reduction of the
dataset. Note that it is not possible to get back the originaln-dimensional dataset from
the new dataset.
```
4.4.3 Illustrative example

We illustrate the ideas of principal component analysis by considering a toy example. In the discus-
sions below, all the details of the computations are given. This is to give the reader an idea of the
complexity of computations and also to help the reader do a “worked example” by hand computa-
tions without recourse to software packages.

Problem

Given the data in Table 4.2, use PCA to reduce the dimension from 2 to 1.

```
Feature Example 1 Example 2 Example 3 Example 4
X 1 4 8 13 7
X 2 11 4 5 14
```
```
Table 4.2: Data for illustrating PCA
```
Solution

1. Scatter plot of data

We have

```
X 1 =^1
4 (^4 +^8 +^13 +^7 )=^8 ;
X 2 =^1
4 (^11 +^4 +^5 +^14 )=^8 :^5 :
```
Figure 4.2 shows the scatter plot of the data together with the point(X 1 ;X 2 ).


#### CHAPTER 4. DIMENSIONALITY REDUCTION 43

#### X 1

#### X 2

(^02468101214)

#### 2

#### 4

#### 6

#### 8

#### 10

#### 12

#### 14

#### (X 1 ;X 2 )

```
Figure 4.2: Scatter plot of data in Table 4.2
```
2. Calculation of the covariance matrix

The covariances are calculated as follows:

```
Cov(X 1 ;X 2 )=
```
#### 1

#### N− 1

```
N
Q
k= 1
```
```
(X 1 k−X 1 )^2
```
```
=^13 ( 4 − 8 )^2 +( 8 − 8 )^2 +( 13 − 8 )^2 +( 7 − 8 )^2 
= 14
```
```
Cov(X 1 ;X 2 )=
```
#### 1

#### N− 1

```
N
Q
k= 1
```
```
(X 1 k−X 1 )(X 2 k−X 2 )
```
```
=^13 (( 4 − 8 )( 11 − 8 : 5 )+( 8 − 8 )( 4 − 8 : 5 )
+( 13 − 8 )( 5 − 8 : 5 )+( 7 − 8 )( 14 − 8 : 5 )
=− 11
Cov(X 2 ;X 1 )=Cov(X 1 ;X 2 )
=− 11
```
```
Cov(X 2 ;X 2 )=
```
#### 1

#### N− 1

```
N
Q
k= 1
```
```
(X 2 k−X 2 )^2
```
```
=^13 ( 11 − 8 : 5 )^2 +( 4 − 8 : 5 )^2 +( 5 − 8 : 5 )^2 +( 14 − 8 : 5 )^2 
= 23
```
The covariance matrix is

#### S=

```
Cov(X 1 ;X 1 ) Cov(X 1 ;X 2 )
Cov(X 2 ;X 1 ) Cov(X 2 ;X 2 )
```
```
=
```
#### 14 − 11

#### − 11 23

3. Eigenvalues of the covariance matrix

The characteristic equation of the covariance matrix is

```
0 =det(S−I)
```
```
=W
```
#### 14 − − 11

#### − 11 23 −

#### W

#### =( 14 −)( 23 −)−(− 11 )×(− 11 )

#### =^2 − 37 + 201


#### CHAPTER 4. DIMENSIONALITY REDUCTION 44

Solving the characteristic equation we get

```
=^12 ( 37 ±
```
#### √

#### 565 )

#### = 30 : 3849 ; 6 : 6151

```
= 1 ;  2 (say)
```
4. Computation of the eigenvectors

To find the first principal components, we need only compute the eigenvector corresponding to the
largest eigenvalue. In the present example, the largest eigenvalue is 1 and so we compute the
eigenvector corresponding to 1.

```
The eigenvector corresponding to= 1 is a vectorU=
u 1
u 2
satisfying the following equation:
```
#### 

#### 0

#### 0

#### =(S− 1 I)X

#### =

#### 14 − 1 − 11

#### − 11 23 − 1 

```
u 1
u 2	
```
```
=
( 14 − 1 )u 1 − 11 u 2
− 11 u 1 +( 23 − 1 )u 2
```
This is equivalent to the following two equations:

```
( 14 − 1 )u 1 − 11 u 2 = 0
− 11 u 1 +( 23 − 1 )u 2 = 0
```
Using the theory of systems of linear equations, we note that these equations are not independent
and solutions are given by
u 1
11

#### =

```
u 2
14 − 1
```
```
=t;
```
that is
u 1 = 11 t; u 2 =( 14 − 1 )t;

wheretis any real number. Takingt= 1 , we get an eigenvector corresponding to 1 as

```
U 1 =
```
#### 11

#### 14 − 1

#### :

To find a unit eigenvector, we compute the length ofX 1 which is given by

```
SSU 1 SS=
```
#### »

#### 112 +( 14 − 1 )^2

#### =

#### »

#### 112 +( 14 − 30 : 3849 )^2

#### = 19 : 7348

Therefore, a unit eigenvector corresponding tolambda 1 is

```
e 1 =
```
#### 11 ~SSU 1 SS

#### ( 14 − 1 )~SSU 1 SS

#### =

#### 11 ~ 19 : 7348

#### ( 14 − 30 : 3849 )~ 19 : 7348

#### =

#### 0 : 5574

#### − 0 : 8303

By carrying out similar computations, the unit eigenvectore 2 corresponding to the eigenvalue
= 2 can be shown to be

```
e 2 =
```
#### 0 : 8303

#### 0 : 5574	 :


#### CHAPTER 4. DIMENSIONALITY REDUCTION 45

#### X 1

#### X 2

(^02468101214)

#### 2

#### 4

#### 6

#### 8

#### 10

#### 12

#### 14

```
e 1
```
```
e 2
```
#### (X 1 ;X 2 )

```
Figure 4.3: Coordinate system for principal components
```
5. Computation of first principal components

Let
X 1 k
X 2 k
be thek-th sample in Table 4.2. The first principal component of this example is given

by (here “T” denotes the transpose of the matrix)

```
eT 1 
X 1 k−X 1
X 2 k−X 2
```
#### = 0 : 5574 − 0 : 8303 

```
X 1 k−X 1
X 2 k−X 2
= 0 : 5574 (X 1 k−X 1 )− 0 : 8303 (X 2 k−X 2 ):
```
For example, the first principal component corresponding to the first example

#### X 11

#### X 21

#### =

#### 4

#### 11

```
is
```
calculated as follows:

####  0 : 5574 − 0 : 8303 

#### X 11 −X 1

#### X 21 −X 2

#### = 0 : 5574 (X 11 −X 1 )− 0 : 8303 (X 21 −X 2 )

#### = 0 : 5574 ( 4 − 8 )− 0 : 8303 ( 11 − 8 ; 5 )

#### =− 4 : 30535

The results of calculations are summarised in Table 4.3.

```
X 1 4 8 13 7
X 2 11 4 5 14
First principal components -4.3052 3.7361 5.6928 -5.1238
```
```
Table 4.3: First principal components for data in Table 4.2
```
6. Geometrical meaning of first principal components

As we have seen in Figure 4.1, we introduce new coordinate axes. First we shift the origin to
the “center”(X 1 ;X 2 )and then change the directions of coordinate axes to the directions of the
eigenvectorse 1 ande 2 (see Figure 4.3).
Next, we drop perpendiculars from the given data points to thee 1 -axis (see Figure 4.4). The first
principal components are thee 1 -coordinates of the feet of perpendiculars, that is, the projections on
thee 1 -axis. The projections of the data points one 1 -axis may be taken as approximations of the
given data points hence we may replace the given data set with these points. Now, each of these


#### CHAPTER 4. DIMENSIONALITY REDUCTION 46

#### X 1

#### X 2

(^02468101214)

#### 2

#### 4

#### 6

#### 8

#### 10

#### 12

#### 14

```
e 1
```
```
( 4 ; 11 ) e^2
```
#### ( 8 ; 4 )

#### ( 13 ; 5 )

#### ( 7 ; 14 )

#### (X 1 ;X 2 )

```
Figure 4.4: Projections of data points on the axis of the first principal component
```
```
PC1 components -4.305187 3.736129 5.692828 -5.123769
```
```
Table 4.4: One-dimensional approximation to the data in Table 4.2
```
approximations can be unambiguously specified by a single number, namely, thee 1 -coordinate of
approximation. Thus the two-dimensional data set given in Table 4.2 can be represented approxi-
mately by the following one-dimensional data set (see Figure 4.5):

#### X 1

#### X 2

#### 0 2 4 6 8 10 12 14

#### 2

#### 4

#### 6

#### 8

#### 10

#### 12

#### 14

```
e 1
```
```
e 2
(4,11)
```
#### (8,4)

#### (13, 5)

#### (7,14)

#### (X 1 ;X 2 )

#### X 1

#### X 2

#### 0 2 4 6 8 10 12 14

#### 2

#### 4

#### 6

#### 8

#### 10

#### 12

#### 14

```
e 1
```
```
e 2
```
#### (X 1 ;X 2 )

```
Figure 4.5: Geometrical representation of one-dimensional approximation to the data in Table 4.2
```
### 4.5 Sample questions

(a) Short answer questions

1. What is dimensionality reduction? How is it implemented?
2. Explain why dimensionality reduction is useful in machine learning.
3. What are the commonly used dimensionality reduction techniques in machine learning?
4. How is the subset selection method used for dimensionality reduction?
5. Explain the method of principal component analysis in machine learning.
6. What are the first principal components of a data?
7. Is subset selection problem an unsupervised learning problem? Why?


#### CHAPTER 4. DIMENSIONALITY REDUCTION 47

8. Is principal component analysis a supervised learning problem? Why?

(b) Long answer questions

1. Describe the forward selection algorithm for implementing the subset selection procedure for
    dimensionality reduction.
2. Describe the backward selection algorithm for implementing the subset selection procedure
    for dimensionality reduction.
3. What is the first principal component of a data? How one can compute it?
4. Describe with the use of diagrams the basic principle of PCA.
5. Explain the procedure for the computation of the principal components of a given data.
6. Describe how principal component analysis is carried out to reduce dimensionality of data
    sets.
7. Given the following data, compute the principal component vectors and the first principal
    components:

```
x 2 3 7
y 11 14 26
```
8. Given the following data, compute the principal component vectors and the first principal
    components:

```
x -3 1 -2
y 2 -1 3
```

Chapter 5

## Evaluation of classifiers

In machine learning, there are several classification algorithms and, given a certain problem, more
than one may be applicable. So there is a need to examine how we can assess how good a se-
lected algorithm is. Also, we need a method to compare the performance of two or more different
classification algorithms. These methods help us choose the right algorithm in a practical situation.

### 5.1 Methods of evaluation

5.1.1 Need for multiple validation sets

When we apply a classification algorithm in a practical situation, we always do a validation test.
We keep a small sample of examples as validation set and the remaining set as the training set. The
classifier developed using the training set is applied to the examples in the validation set. Based on
the performance on the validation set, the accuracy of the classifier is assessed. But, the performance
measure obtained by a single validation set alone does not give a true picture of the performance of a
classifier. Also these measures alone cannot be meaningfully used to compare two algorithms. This
requires us to have different validation sets.
Cross-validation in general, andk-fold cross-validation in particular, are two common method
for generating multiple training-validation sets from a given dataset.

5.1.2 Statistical distribution of errors

We use a classification algorithm on a dataset and generate a classifier. If we do the training once,
we have one classifier and one validation error. To average over randomness (in training data, initial
weights, etc.), we use the same algorithm and generate multiple classifiers. We test these classifiers
on multiple validation sets and record a sample of validation errors. We base our evaluation of the
classification algorithm on thestatistical distribution of these validation errors. We can use this
distribution for assessing theexpected errorrate of the classification algorithm for that problem, or
compare it with the error rate distribution of some other classification algorithm.
A detailed discussion of these ideas is beyond the scope of these notes.

5.1.3 No-free lunch theorem

Whatever conclusion we draw from our analysis is conditioned on the dataset we are given. We
are not comparing classification algorithms in a domain-independent way but on some particular
application. We are not saying anything about the expected error-rate of a learning algorithm, or
comparing one learning algorithm with another algorithm, in general. Any result we have is only
true for the particular application. There is no such thing as the “best” learning algorithm. For any

#### 48


#### CHAPTER 5. EVALUATION OF CLASSIFIERS 49

learning algorithm, there is a dataset where it is very accurate and another dataset where it is very
poor. This is called theNo Free Lunch Theorem.^1

5.1.4 Other factors

Classification algorithms can be compared based not only on error rates but also on several other
criteria like the following:

- risks when errors are generalized using loss functions
- training time and space complexity,
- testing time and space complexity,
- interpretability, namely, whether the method allows knowledge extraction which can be checked
    and validated by experts, and
- easy programmability.

### 5.2 Cross-validation

To test the performance of a classifier, we need to have a number of training/validation set pairs
from a datasetX. To get them, if the sampleXis large enough, we can randomly divide it then
divide each part randomly into two and use one half for training and the other half for validation.
Unfortunately, datasets are never large enough to do this. So, we use the same data split differently;
this is calledcross-validation.
Cross-validation is a technique to evaluate predictive models by partitioning the original sample
into a training set to train the model, and a test set to evaluate it.
Theholdout methodis the simplest kind of cross validation. The data set is separated into two
sets, called the training set and the testing set. The algorithm fits a function using the training set
only. Then the function is used to predict the output values for the data in the testing set (it has never
seen these output values before). The errors it makes are used to evaluate the model.

### 5.3 K-fold cross-validation

InK-fold cross-validation, the datasetXis divided randomly intoKequal-sized parts,Xi,i=
1 ;:::;K. To generate each pair, we keep one of theKparts out as the validation setVi, and combine
the remainingK− 1 parts to form the training setTi. Doing thisKtimes, each time leaving out
another one of theKparts out, we getKpairs(Vi;Ti):

```
V 1 =X 1 ; T 1 =X 2 ∪X 3 ∪:::∪XK
V 2 =X 2 ; T 2 =X 1 ∪X 3 ∪:::∪XK
⋯
VK=XK; TK=X 1 ∪X 2 ∪:::∪XK− 1
```
Remarks

1. There are two problems with this: First, to keep the training set large, we allow validation sets
    that are small. Second, the training sets overlap considerably, namely, any two training sets
    shareK− 2 parts.

(^1) “We have dubbed the associated results NFL theorems because they demonstrate that if an algorithm performs well on
a certain class of problems then it necessarily pays for that with degraded performance on the set of all remaining prob-
lems.”(David Wolpert and William Macready in [7])


#### CHAPTER 5. EVALUATION OF CLASSIFIERS 50

2. Kis typically 10 or 30. AsKincreases, the percentage of training instances increases and we
    get more robust estimators, but the validation set becomes smaller. Furthermore, there is the
    cost of training the classifierKtimes, which increases asKis increased.

```
test set
```
```
test set
```
```
test set
```
```
test set
```
```
test set
```
```
training set
```
```
training set
```
```
training set
```
```
training set
```
```
training set
```
```
training set
```
```
training set
```
```
training set
```
```
1-st fold
```
```
2-nd fold
```
```
3-rd fold
```
```
4-th fold
```
```
5-th fold
```
```
Figure 5.1: One iteration in a 5-fold cross-validation
```
Leave-one-out cross-validation

An extreme case ofK-fold cross-validation isleave-one-outwhere given a dataset ofNinstances,
only one instance is left out as the validation set and training uses the remainingN− 1 instances.
We then getNseparate pairs by leaving out a different instance at each iteration. This is typically
used in applications such as medical diagnosis, where labeled data is hard to find.

5.3.1 5 × 2 cross-validation

In this method, the datasetXis divided into two equal partsX
( 1 )
1 andX

( 2 )
1. We take as the training
set andX 1 (^2 )as the validation set. We then swap the two sets and takeX 1 (^2 )as the training set and

X 1 (^1 )as the validation set. This is the first fold. the process id repeated four more times to get ten
pairs of training sets and validation sets.

```
T 1 =X 1 (^1 ); V 1 =X( 12 )
T 2 =X 1 (^2 ); V 2 =X( 11 )
T 3 =X 2 (^1 ); V 3 =X( 22 )
T 4 =X 2 (^2 ); V 4 =X( 21 )
⋮
T 9 =X 5 (^1 ); V 3 =X( 52 )
T 10 =X 5 (^2 ); V 10 =X( 51 )
```
It has been shown that after five folds, the validation error rates become too dependent and do
not add new information. On the other hand, if there are fewer than five folds, we get fewer data
(fewer than ten) and will not have a large enough sample to fit a distribution and test our hypothesis.


#### CHAPTER 5. EVALUATION OF CLASSIFIERS 51

5.3.2 Bootstrapping

Bootstrapping in statistics

In statistics, the term “bootstrap sampling”, the “bootstrap” or “bootstrapping” for short, refers to
process of “random sampling with replacement”.

Example

For example, let there be five balls labeled A, B, C, D, E in an urn. We wish to select different
samples of balls from the urn each sample containing two balls. The following procedure may be
used to select the samples. This is an example for bootstrap sampling.

1. We select two balls from the basket. Let them be A and E. Record the labels.
2. Put the two balls back in the basket.
3. We select two balls from the basket. Let them be C and E. Record the labels.
4. Put the two balls back into the basket.

This is repeated as often as required. So we get different samples of size 2, say, A, E; B, E; etc.
These samples are obtained by sampling with replacement, that is, by bootstrapping.

Bootstrapping in machine learning

In machine learning, bootstrapping is the process of computing performance measures using several
randomly selected training and test datasets which are selected through a precess of sampling with
replacement, that is, through bootstrapping. Sample datasets are selected multiple times.
The bootstrap procedure will create one or more new training datasets some of which are re-
peated. The corresponding test datasets are then constructed from the set of examples that were not
selected for the respective training datasets.

### 5.4 Measuring error

5.4.1 True positive, false positive, etc.

Definitions

Consider a binary classification model derived from a two-class dataset. Let the class labels becand
¬c. Letxbe a test instance.

1. True positive
    Let the true class label ofxbec. If the model predicts the class label ofxasc, then we say
    that the classification ofxistrue positive.
2. False negative
    Let the true class label ofxbec. If the model predicts the class label ofxas¬c, then we say
    that the classification ofxisfalse negative.
3. True negative
    Let the true class label ofxbe¬c. If the model predicts the class label ofxas¬c, then we say
    that the classification ofxistrue negative.
4. False positive
    Let the true class label ofxbe¬c. If the model predicts the class label ofxasc, then we say
    that the classification ofxisfalse positive.


#### CHAPTER 5. EVALUATION OF CLASSIFIERS 52

```
Actual label ofxisc Actual label ofxis¬c
Predicted label ofxisc True positive False positive
Predicted label ofxis¬c False negative True negative
```
5.4.2 Confusion matrix

A confusion matrix is used to describe the performance of a classification model (or “classifier”) on
a set of test data for which the true values are known. Aconfusion matrixis a table that categorizes
predictions according to whether they match the actual value.

Two-class datasets

For a two-class dataset, a confusion matrix is a table with two rows and two columns that reports the
number of false positives, false negatives, true positives, and true negatives.
Assume that a classifier is applied to a two-class test dataset for which the true values are known.
Let TP denote the number of true positives in the predicted values, TN the number of true negatives,
etc. Then the confusion matrix of the predicted values can be represented as follows:

```
Actual condition
is true
```
```
Actual condition
is false
Predicted condi-
tion is true
```
#### TP FP

```
Predicted condi-
tion is false
```
#### FN FN

```
Table 5.1: Confusion matrix for two-class dataset
```
Multiclass datasets

Confusion matrices can be constructed for multiclass datasets also.

Example

If a classification system has been trained to distinguish between cats, dogs and rabbits, a confusion
matrix will summarize the results of testing the algorithm for further inspection. Assuming a sample
of 27 animals - 8 cats, 6 dogs, and 13 rabbits, the resulting confusion matrix could look like the table
below: This confusion matrix shows that, for example, of the 8 actual cats, the system predicted that

```
Actual “cat” Actual “dog” Actual “rabbit”
Predicted “cat” 5 2 0
Predicted “dog” 3 3 2
Predicted “ rabbit” 0 1 11
```
three were dogs, and of the six dogs, it predicted that one was a rabbit and two were cats.

5.4.3 Precision and recall

In machine learning, precision and recall are two measures used to assess the quality of results
produced by a binary classifier. They are formally defined as follows.


#### CHAPTER 5. EVALUATION OF CLASSIFIERS 53

Definitions

Let a binary classifier classify a collection of test data. Let

```
TP=Number of true positives
TN =Number of true negatives
FP=Number of false positives
FN=Number of false negatives
```
TheprecisionPis defined as

```
P=
```
#### TP

#### TP+FP

TherecallRis defined as

```
R=
```
#### TP

#### TP+FN

Problem 1

Suppose a computer program for recognizing dogs in photographs identifies eight dogs in a picture
containing 12 dogs and some cats. Of the eight dogs identified, five actually are dogs while the rest
are cats. Compute the precision and recall of the computer program.

Solution

We have:

```
TP= 5
FP= 3
FN= 7
```
TheprecisionPis

```
P=
```
#### TP

#### TP+FP

#### =

#### 5

#### 5 + 3

#### =

#### 5

#### 8

TherecallRis

```
R=
```
#### TP

#### TP+FN

#### =

#### 5

#### 5 + 7

#### =

#### 5

#### 12

Problem 2

Let there be 10 balls (6 white and 4 red balls) in a box and let it be required to pick up the red balls
from them. Suppose we pick up 7 balls as the red balls of which only 2 are actually red balls. What
are the values of precision and recall in picking red ball?

Solution

Obviously we have:

```
TP= 2
FP= 7 − 2 = 5
FN= 4 − 2 = 2
```
TheprecisionPis

```
P=
```
#### TP

#### TP+FP

#### =

#### 2

#### 2 + 5

#### =

#### 2

#### 7

TherecallRis

```
R=
```
#### TP

#### TP+FN

#### =

#### 2

#### 2 + 2

#### =

#### 1

#### 2


#### CHAPTER 5. EVALUATION OF CLASSIFIERS 54

Problem 3

Assume the following: A database contains 80 records on a particular topic of which 55 are relevant
to a certain investigation. A search was conducted on that topic and 50 records were retrieved. Of the
50 records retrieved, 40 were relevant. Construct the confusion matrix for the search and calculate
the precision and recall scores for the search.

Solution

Each record may be assigned a class label “relevant" or “not relevant”. All the 80 records were
tested for relevance. The test classified 50 records as “relevant”. But only 40 of them were actually
relevant. Hence we have the following confusion matrix for the search:

```
Actual ”relevant”
Actual “not rele-
vant”
Predicted “rele-
vant”
```
#### 40 10

```
Predicted “not
relevant”
```
#### 15 25

```
Table 5.2: Example for confusion matrix
```
#### TP= 40

#### FP= 10

#### FN= 15

TheprecisionPis

```
P=
```
#### TP

#### TP+FP

#### =

#### 40

#### 40 + 10

#### =

#### 4

#### 5

TherecallRis

```
R=
```
#### TP

#### TP+FN

#### =

#### 40

#### 40 + 15

#### =

#### 40

#### 55

5.4.4 Other measures of performance

Using the data in the confusion matrix of a classifier of two-class dataset, several measures of per-
formance have been defined. A few of them are listed below.

1. Accuracy=

#### TP+TN

#### TP+TN+FP+FN

2. Error rate= 1 −Accuracy
3. Sensitivity=

#### TP

#### TP+FN

4. Specificity=

#### TN

#### TN+FP

5. F-measure=

#### 2 ×TP

#### 2 ×TP+FP+FN

### 5.5 Receiver Operating Characteristic (ROC)

The acronym ROC stands for Receiver Operating Characteristic, a terminology coming from signal
detection theory. The ROC curve was first developed by electrical engineers and radar engineers
during World War II for detecting enemy objects in battlefields. They are now increasingly used in
machine learning and data mining research.


#### CHAPTER 5. EVALUATION OF CLASSIFIERS 55

TPR and FPR

Let a binary classifier classify a collection of test data. Let, as before,

```
TP=Number of true positives
TN =Number of true negatives
FP=Number of false positives
FN=Number of false negatives
```
```
Now we introduce the following terminology:
```
```
TPR=True Positive Rate
```
```
=
```
#### TP

#### TP+FN

```
=Fraction of positive examples correctly classified
=Sensitivity
FPR=False Positive Rate
```
```
=
```
#### FP

#### FP+TN

```
=Fraction of negative examples incorrectly classified
= 1 −Specificity
```
ROC space

We plot the values of FPR along the horizontal axis (that is ,x-axis) and the values of TPR along
the vertical axis (that is,y-axis) in a plane. For each classifier, there is a unique point in this plane
with coordinates(FPR;TPR). The ROC space is the part of the plane whose points correspond to
(FPR;TPR). Each prediction result or instance of a confusion matrix represents one point in the
ROC space.
The position of the point(FPR;TPR)in the ROC space gives an indication of the performance
of the classifier. For example, let us consider some special points in the space.

Special points in ROC space

1. The left bottom corner point( 0 ; 0 ): Always negative prediction
    A classifier which produces this point in the ROC space never classifies an example as positive,
    neither rightly nor wrongly, because for this point TP = 0 and FP = 0. It always makes
    negative predictions. All positive instances are wrongly predicted and all negative instances
    are correctly predicted. It commits no false positive errors.
2. The right top corner point( 1 ; 1 ): Always positive prediction
    A classifier which produces this point in the ROC space always classifies an example as posi-
    tive because for this point FN= 0 and TN = 0. All positive instances are correctly predicted
    and all negative instances are wrongly predicted. It commits no false negative errors.
3. The left top corner point( 0 ; 1 ): Perfect prediction
    A classifier which produces this point in the ROC space may be thought as a perfect classifier.
    It produces no false positives and no false negatives.
4. Points along the diagonal: Random performance
    Consider a classifier where the class labels are randomly guessed, say by flipping a coin. Then,
    the corresponding points in the ROC space will be lying very near the diagonal line joining
    the points( 0 ; 0 )and( 1 ; 1 ).


#### CHAPTER 5. EVALUATION OF CLASSIFIERS 56

#### .0

#### .0

#### .1

#### .1

#### .2

#### .2

#### .3

#### .3

#### .4

#### .4

#### .5

#### .5

#### .6

#### .6

#### .7

#### .7

#### .8

#### .8

#### .9

#### .9

#### 1

#### 1

```
False Positive Rate (FPR)→
```
```
True Positive Rate (TPR)
```
#### →

```
ROC space
```
```
Always negative prediction
```
```
Perfect prediction Always positive prediction
```
```
Point on diagonal
(Random performance)
```
```
Figure 5.2: The ROC space and some special points in the space
```
ROC curve

In the case of certain classification algorithms, the classifier may depend on a parameter. Different
values of the parameter will give different classifiers and these in turn give different values to TPR
and FPR. The ROC curve is the curve obtained by plotting in the ROC space the points(TPR;FPR)
obtained by assigning all possible values to the parameter in the classifier.


#### CHAPTER 5. EVALUATION OF CLASSIFIERS 57

#### .0

#### .0

#### .1

#### .1

#### .2

#### .2

#### .3

#### .3

#### .4

#### .4

#### .5

#### .5

#### .6

#### .6

#### .7

#### .7

#### .8

#### .8

#### .9

#### .9

#### 1

#### 1

```
False Positive Rate (FPR)→
```
```
True Positive Rate (TPR)
```
#### →

```
ROC space
```
#### A

#### B

#### C

```
Figure 5.3: ROC curves of three different classifiers A, B, C
```
The closer the ROC curve is to the top left corner( 0 ; 1 )of the ROC space, the better the accuracy
of the classifier. Among the three classifiers A, B, C with ROC curves as shown in Figure 5.3, the
classifier C is closest to the top left corner of the ROC space. Hence, among the three, it gives the
best accuracy in predictions.

Example

```
Cut-off value of BMI
Breast cancer Normal persons
TPR FPR
TP FN FP TN
18 100 0 200 0 1.00 1.000
20 100 0 198 2 1.00 0.990
22 99 1 177 23 0.99 0.885
24 95 5 117 83 0.95 0.585
26 85 15 80 120 0.85 0.400
28 66 34 53 147 0.66 0.265
30 47 53 27 173 0.47 0.135
32 34 66 17 183 0.34 0.085
34 21 79 14 186 0.21 0.070
36 17 83 6 194 0.17 0.030
38 7 93 4 196 0.07 0.020
40 1 99 1 199 0.01 0.005
```
```
Table 5.3: Data on breast cancer for various values of BMI
```
The body mass index (BMI) of a person is defined as (weight(kg)/height(m)^2 ). Researchers have
established a link between BMI and the risk of breast cancer among women. The higher the BMI
the higher the risk of developing breast cancer. The critical threshold value of BMI may depend on
several parameters like food habits, socio-cultural-economic background, life-style, etc. Table 5.3


#### CHAPTER 5. EVALUATION OF CLASSIFIERS 58

gives real data of a breast cancer study with a sample having 100 patients and 200 normal persons.^2
The table also shows the values of TPR and FPR for various cut-off values of BMI. The ROC curve
of the data in Table 5.3 is shown in Figure 5.4.

#### .0

#### .0

#### .1

#### .1

#### .2

#### .2

#### .3

#### .3

#### .4

#### .4

#### .5

#### .5

#### .6

#### .6

#### .7

#### .7

#### .8

#### .8

#### .9

#### .9

#### 1

#### 1

```
False Positive Rate (FPR)→
```
```
True Positive Rate (TPR)
```
#### →

```
ROC space
```
```
Cut-off BMI = 26
```
```
Cut-off BMI = 28
```
```
AUC=Area of shaded region
```
Figure 5.4: ROC curve of data in Table 5.3 showing the points closest to the perfect prediction point
( 0 ; 1 )

Area under the ROC curve

The measure of the area under the ROC curve is denoted by the acronym AUC (see Figure 5.4). The
value of AUC is a measure of the performance of a classifier. For the perfect classifier, AUC=1.0.

### 5.6 Sample questions

(a) Short answer questions

1. What is cross-validation in machine learning?
2. What is meant by 5 × 2 cross-validation?
3. What is meant by leave-one-out cross validation?
4. What is meant by the confusion matrix of a binary classification problem.
5. Define the following terms: precision, recall, sensitivity, specificity.
6. What is ROC curve in machine learning?
7. What are true positive rates and false positive rates in machine learning?
8. What is AUC in relation to ROC curves?

(^2) https://www.ncbi.nlm.nih.gov/ pmc/articles/PMC3755824/


#### CHAPTER 5. EVALUATION OF CLASSIFIERS 59

(b) Long answer questions

1. Explain cross-validation in machine learning. Explain the different types of cross-validations.
2. What is meant by true positives etc.? What is meant by confusion matrix of a binary classifi-
    cation problem? Explain how this can be extended to multi-class problems.
3. What are ROC space and ROC curve in machine learning? In ROC space, which points
    correspond to perfect prediction, always positive prediction and always negative prediction?
    Why?
4. Consider a two-class classification problem of predicting whether a photograph contains a
    man or a woman. Suppose we have a test dataset of 10 records with expected outcomes and a
    set of predictions from our classification algorithm.

```
Expected Predicted
1 man woman
2 man man
3 woman woman
4 man man
5 woman man
6 woman woman
7 woman woman
8 man man
9 man woman
10 woman woman
```
```
(a) Compute the confusion matrix for the data.
(b) Compute the accuracy, precision, recall, sensitivity and specificity of the data.
```
5. Suppose 10000 patients get tested for flu; out of them, 9000 are actually healthy and 1000
    are actually sick. For the sick people, a test was positive for 620 and negative for 380. For
    the healthy people, the same test was positive for 180 and negative for 8820. Construct a
    confusion matrix for the data and compute the accuracy, precision and recall for the data.
6. Given the following data, construct the ROC curve of the data. Compute the AUC.

```
Threshold TP TN FP FN
1 0 25 0 29
2 7 25 0 22
3 18 24 1 11
4 26 20 5 3
5 29 11 14 0
6 29 0 25 0
7 29 0 25 0
```
7. Given the following hypothetical data at various cut-off points of mid-arm circumference of
    mid-arm circumference to detect low birth-weight construct the ROC curve for the data.


#### CHAPTER 5. EVALUATION OF CLASSIFIERS 60

```
Mid-arm circumference (cm) Normal birth-weight Low birth-weight
TP TN
≤ 8 : 3 13 867
≤ 8 : 4 24 844
≤ 8 : 5 73 826
≤ 8 : 6 90 800
≤ 8 : 7 113 783
≤ 8 : 8 119 735
≤ 8 : 9 121 626
≤ 9 : 0 125 505
≤ 9 : 1 127 435
≤ 9 : 2 and above 130 0
```

Chapter 6

## Bayesian classifier and ML

## estimation

The Bayesian classifier is an algorithm for classifying multiclass datasets. This is based on the
Bayes’ theorem in probability theory. Bayes in whose name the theorem is known was an English
statistician who was known for having formulated a specific case of a theorem that bears his name.
The classifier is also known as “naive Bayes Algorithm” where the word “naive” is an English word
with the following meanings: simple, unsophisticated, or primitive. We first explain Bayes’ theorem
and then describe the algorithm. Of course, we require the notion of conditional probability.

### 6.1 Conditional probability

The probability of the occurrence of an eventAgiven that an eventBhas already occurred is called
theconditional probability ofAgivenBand is denoted byP(ASB). We have

```
P(ASB)=
```
#### P(A∩B)

#### P(B)

```
if P(B)≠ 0 :
```
6.1.1 Independent events

1. Two eventsAandBare said to beindependentif
    P(A∩B)=P(A)P(B):
2. Three eventsA;B;Care said to bepairwise independentif

```
P(B∩C)=P(B)P(C)
P(C∩A)=P(C)P(A)
P(A∩B)=P(A)P(B)
```
3. Three eventsA;B;Care said to bemutually independentif

```
P(B∩C)=P(B)P(C) (6.1)
P(C∩A)=P(C)P(A) (6.2)
P(A∩B)=P(A)P(B) (6.3)
P(A∩B∩C)=P(A)P(B)P(C) (6.4)
```
4. In general, a family ofkeventsA 1 ;A 2 ;:::;Akis said to be mutually independent if for any
    subfamily consisting ofAi 1 ;:::Aimwe have
       P(Ai 1 ∩:::∩Aim)=P(Ai 1 ):::P(Aim):

#### 61


#### CHAPTER 6. BAYESIAN CLASSIFIER AND ML ESTIMATION 62

Remarks

Consider events and respective probabilities as shown in Figure 6.1. It can be seen that, in this case,
the conditions Eqs.(6.1)–(6.3) are satisfied, but Eq.(6.4) is not satisfied. But if the probabilities are
as in Figure 6.2, then Eq.(6.4) is satisfied but all the conditions in Eqs.(6.1)–(6.2) are not satisfied.

Figure 6.1: EventsA;B;Cwhich are not mutually independent: Eqs.(6.1)–(6.3) are satisfied, but
Eq.(6.4) is not satisfied.

Figure 6.2: EventsA;B;Cwhich are not mutually independent: Eq.(6.4) is satisfied but Eqs.(6.1)–
(6.2) are not satisfied.

### 6.2 Bayes’ theorem

6.2.1 Theorem

LetAandBany two events in a random experiment. IfP(A)≠ 0 , then

#### P(BSA)=

#### P(ASB)P(B)

#### P(A)

#### :

6.2.2 Remarks

1. The importance of the result is that it helps us to “invert” conditional probabilities, that is, to
    express the conditional probabilityP(ASB)in terms of the conditional probabilityP(BSA).
2. The following terminology is used in this context:
    - Ais called thepropositionandBis called theevidence.
    - P(A)is called theprior probabilityof proposition andP(B)is called the prior proba-
       bility of evidence.


#### CHAPTER 6. BAYESIAN CLASSIFIER AND ML ESTIMATION 63

- P(ASB)is called theposterior probabilityofAgivenB.
- P(BSA)is called thelikelihoodofBgivenA.

6.2.3 Generalisation

Let the sample space be divided into disjoint eventsB 1 ;B 2 ;:::;BnandAbe any event. Then we
have

```
P(BkSA)=
```
```
P(ASBk)P(Bk)
∑ni= 1 P(ASBi)P(Bi)
```
6.2.4 Examples

Problem 1

Consider a set of patients coming for treatment in a certain clinic. LetAdenote the event that
a “Patient has liver disease” andBthe event that a “Patient is an alcoholic.” It is known from
experience that 10% of the patients entering the clinic have liver disease and 5% of the patients are
alcoholics. Also, among those patients diagnosed with liver disease, 7% are alcoholics. Given that
a patient is alcoholic, what is the probability that he will have liver disease?

Solution

Using the notations of probability, we have

```
P(A)=10%= 0 : 10
P(B)=5%= 0 : 05
P(BSA)=7%= 0 : 07
```
```
P(ASB)=
```
#### P(BSA)P(A)

#### P(B)

#### =

#### 0 : 07 × 0 : 10

#### 0 : 05

#### = 0 : 14

Problem 2

Three factories A, B, C of an electric bulb manufacturing company produce respectively 35%. 35%
and 30% of the total output. Approximately 1.5%, 1% and 2% of the bulbs produced by these
factories are known to be defective. If a randomly selected bulb manufactured by the company was
found to be defective, what is the probability that the bulb was manufactures in factory A?

Solution

LetA;B;Cdenote the events that a randomly selected bulb was manufactured in factory A, B, C
respectively. LetDdenote the event that a bulb is defective. We have the following data:

```
P(A)= 0 : 35 ; P(B)= 0 : 35 ; P(C)= 0 : 30
P(DSA)= 0 : 015 ; P(DSB)= 0 : 010 ; P(DSC)= 0 : 020
```
We are required to findP(ASD). By the generalisation of the Bayes’ theorem we have:

#### P(ASD)=

#### P(DSA)P(A)

#### P(DSA)P(A)+P(DSB)P(B)+P(DSC)P(C)

#### =

#### 0 : 015 × 0 : 35

#### 0 : 015 × 0 : 35 + 0 : 010 × 0 : 35 + 0 : 020 × 0 : 30

#### = 0 : 356 :


#### CHAPTER 6. BAYESIAN CLASSIFIER AND ML ESTIMATION 64

### 6.3 Naive Bayes algorithm

6.3.1 Assumption

The naive Bayes algorithm is based on the following assumptions:

- All thefeatures are independentand are unrelated to each other. Presence or absence of a
    feature does not influence the presence or absence of any other feature.
- The data hasclass-conditional independence, which means that events are independent so
    long as they are conditioned on the same class value.

These assumptions are, in general, true in many real world problems. It is because of these assump-
tions, the algorithm is called anaivealgorithm.

6.3.2 Basic idea

Suppose we have a training data set consisting ofNexamples havingnfeatures. Let the features
be named as(F 1 ;:::;Fn). A feature vector is of the form(f 1 ;f 2 ;:::;fn). Associated with each
example, there is a certain class label. Let the set of class labels be{c 1 ;c 2 ;:::;cp}.
Suppose we are given a test instance having the feature vector

```
X=(x 1 ;x 2 ;:::;xn):
```
We are required to determine the most appropriate class label that should be assigned to the test
instance. For this purpose we compute the following conditional probabilities

```
P(c 1 SX);P(c 2 SX);:::;P(cpSX): (6.5)
```
and choose the maximum among them. Let the maximum probability beP(ciSX). Then, we choose
cias the most appropriate class label for the training instance havingXas the feature vector.
The direct computation of the probabilities given in Eq.(6.5) are difficult for a number of reasons.
The Bayes’ theorem can b applied to obtain a simpler method. This is explained below.

6.3.3 Computation of probabilities

Using Bayes’ theorem, we have:

```
P(ckSX)=
```
```
P(XSck)P(ck)
P(X)
```
#### (6.6)

Since, by assumption, the data has class-conditional independence, we note that the events “x 1 Sck”,
“x 2 Sck”,⋯,xnSckare independent (because they are all conditioned on the same class labelck).
Hence we have

```
P(XSck)=P((x 1 ;x 2 ;:::;xn)Sck)
=P(x 1 Sck)P(x 2 Sck)⋯P(xnSck)
```
Using this in Eq,(6.6) we get

```
P(ckSX)=
```
```
P(x 1 Sck)P(x 2 Sck)⋯P(xnSck)P(ck)
P(X)
```
#### :

Since the denominatorP(X)is independent of the class labels, we have

```
P(ckSX)∝P(x 1 Sck)P(x 2 Sck)⋯P(xnSck)P(ck):
```
So it is enough to find the maximum among the following values:

```
P(x 1 Sck)P(x 2 Sck)⋯P(xnSck)P(ck); k= 1 ;:::;p:
```

#### CHAPTER 6. BAYESIAN CLASSIFIER AND ML ESTIMATION 65

Remarks

The various probabilities in the above expression are computed as follows:

```
P(ck)=
```
```
No. of examples with class labelck
Total number of examples
```
```
P(xjSck)=
```
```
No. of examples withjth feature equal toxjand class labelck
No. of examples with class labelck
```
6.3.4 The algorithm

Algorithm: Naive Bayes

Let there be a training data set havingnfeaturesF 1 ;:::;Fn. Letf 1 denote an arbitrary value ofF 1 ,
f 2 ofF 2 , and so on. Let the set of class labels be{c 1 ;c 2 ;:::;cp}. Let there be given a test instance
having the feature vector
X=(x 1 ;x 2 ;:::;xn):

We are required to determine the most appropriate class label that should be assigned to the test
instance.

Step1. Compute the probabilitiesP(ck)fork= 1 ;:::;p.

Step2. Form a table showing the conditional probabilities

```
P(f 1 Sck); P(f 2 Sck); ::: ;P(fnSck)
```
```
for all values off 1 ;f 2 ;:::;fnand fork= 1 ;:::;p.
```
Step3. Compute the products

```
qk=P(x 1 Sck)P(x 2 Sck)⋯P(xnSck)P(ck)
```
```
fork= 1 ;:::;p.
```
Step4. Findjsuchqj=max{q 1 ;q 2 ;:::;qp}.

Step5. Assign the class labelcjto the test instanceX.

Remarks

In the above algorithm, Steps 1 and 2 constitute the learning phase of the algorithm. The remaining
steps constitute the testing phase. For testing purposes, only the table of probabilities is required;
the original data set is not required.

6.3.5 Example

Problem

Consider a training data set consisting of the fauna of the world. Each unit has three features named
“Swim”, “Fly” and “Crawl”. Let the possible values of these features be as follows:

```
Swim Fast, Slow, No
Fly Long, Short, Rarely, No
Crawl Yes, No
```
For simplicity, each unit is classified as “Animal”, “Bird” or “Fish”. Let the training data set be as in
Table 6.1. Use naive Bayes algorithm to classify a particular species if its features are (Slow, Rarely,
No)?


#### CHAPTER 6. BAYESIAN CLASSIFIER AND ML ESTIMATION 66

```
Sl. No. Swim Fly Crawl Class
1 Fast No No Fish
2 Fast No Yes Animal
3 Slow No No Animal
4 Fast No No Animal
5 No Short No Bird
6 No Short No Bird
7 No Rarely No Animal
8 Slow No Yes Animal
9 Slow No No Fish
10 Slow No Yes Fish
11 No Long No Bird
12 Fast No No Bird
```
```
Table 6.1: Sample data set for naive Bayes algorithm
```
Solution

In this example, the features are

```
F 1 =“Swim”; F 2 =“Fly”; F 3 =“Crawl”:
```
The class labels are

```
c 1 =“Animal”; c 2 =“ Bird”; c 3 =“Fish”:
```
The test instance is (Slow, Rarely, No) and so we have:

```
x 1 =“Slow”; x 2 =“Rarely”; x 3 =“No”:
```
We construct the frequency table shown in Table 6.2 which summarises the data. (It may be noted
that the construction of the frequency table is not part of the algorithm.)

```
Class
```
```
Features
Swim (F 1 ) Fly (F 2 ) Crawl (F 3 ) Total
Fast Slow No Long Short Rarely No Yes No
Animal (c 1 ) 2 2 1 0 0 1 4 2 3 5
Bird (c 2 ) 1 0 3 1 2 0 1 1 3 4
Fish (c 3 ) 1 2 0 0 0 0 3 0 3 3
Total 4 4 4 1 2 1 8 4 8 12
```
```
Table 6.2: Frequency table for the data in Table 6.1
```
Step1. We compute following probabilities.

```
P(c 1 )=
No. of records with class label “Animal”
Total number of examples
= 5 ~ 12
```
```
P(c 2 )=
```
```
No. of records with class label “Bird”
Total number of examples
= 4 ~ 12
```
```
P(c 3 )=
No of records with class label “Fish”
Total number of examples
= 3 ~ 12
```

#### CHAPTER 6. BAYESIAN CLASSIFIER AND ML ESTIMATION 67

Step2. We construct the following table of conditional probabilities:

```
Class
```
```
Features
Swim (F 1 ) Fly (F 2 ) Crawl (F 3 )
f 1 f 2 f 3
Fast Slow No Long Short Rarely No Yes No
Animal (c 1 ) 2/5 2/5 1/5 0/5 0/5 1/5 4/5 2/5 3/5
Bird (c 2 ) 1/4 0/4 3/4 1/4 2/4 0/4 1/4 0/4 4/4
Fish (c 3 ) 1 3 2/3 0/3 0/3 0/3 0/3 3/3 0/3 3/3
```
```
Table 6.3: Table of the conditional probabilitiesP(fiSck)
```
```
Note: The conditional probabilities are calculated as follows:
```
```
P((F 1 =Slow)Sc 1 )=
```
```
No. of records withF 1 = Slow and class labelc 1
No. of records with class labelc 1
= 2 ~ 5 :
```
Step3. We now calculate the following numbers:

```
q 1 =P(x 1 Sc 1 )P(x 2 Sc 1 )P(x 3 Sc 1 )P(c 1 )
=( 2 ~ 5 )×( 1 ~ 5 )×( 3 ~ 5 )×( 5 ~ 12 )
= 0 : 02
q 2 =P(x 1 Sc 2 )P(x 2 Sc 2 )P(x 3 Sc 2 )P(c 2 )
=( 0 ~ 4 )×( 0 ~ 4 )×( 3 ~ 4 )×( 4 ~ 12 )
= 0
q 3 =P(x 1 Sc 3 )P(x 2 Sc 3 )P(x 3 Sc 3 )P(c 3 )
=( 2 ~ 3 )×( 0 ~ 3 )×( 3 ~ 3 )×( 3 ~ 12 )
= 0
```
Step4. Now
max{q 1 ;q 2 ;q 3 }= 0 : 05 :

Step5. The maximum isq 1 an it corresponds to the class label

```
c 1 = “ Animal”:
```
```
So we assign the class label “Animal” to the test instance “(Slow, Rarely, No)”.
```
### 6.4 Using numeric features with naive Bayes algorithm

The naive Bayes algorithm can be applied to a data set only if the features are categorical. This is
so because, the various probabilities are computed using the various frequencies and the frequencies
can be counted only if each feature has a limited set of values.
If a feature is numeric, it has to bediscretizedbefore applying the algorithm. The discretization
is effected by putting the numeric values into categories known asbins. Because of thisdiscretization
is also known asbinning. This is ideal when there are large amounts of data.
There are several different ways to discretize a numeric feature.

1. If there are natural categories orcut pointsin the distribution of values, use these cut points to
    create the bins. For example, let the data consists of records of times when certain activities
    were carried out. The the categories, or bins, may be created as in Figure 6.3.


#### CHAPTER 6. BAYESIAN CLASSIFIER AND ML ESTIMATION 68

```
Figure 6.3: Discretization of numeric data: Example
```
2. If there are no obvious cut points, we may discretize the feature using quantiles. We may
    divide the data into three bins with tertiles, four bins with quartiles, or five bins with quintiles,
    etc.

### 6.5 Maximum likelihood estimation (ML estimation)

To develop a Bayesian classifier, we need the probabilitiesP(xSck)for the class labelsc 1 ;:::;ck.
These probabilities are estimated from the given data. There is need to know whether the sample
is truly random so that the computed probabilities are good approximations to true probabilities. If
they are good approximations of true probabilities, then there would be an underlying probability
distribution. Suppose we have reasons to believe that the underlying distribution has a particular
form, say binomial, Poisson or normal. These forms are defined by probability functions or proba-
bility density functions. There are parameters which define these functions, and these parameters are
to be estimated to test whether a given data follow some particular distribution. Maximum likelihood
estimation is particular method to estimate the parameters of a probability distribution.

Definition

Maximum likelihood estimation(MLE) is a method of estimating the parameters of a statistical
model, given observations. MLE attempts to find the parameter values thatmaximize the likelihood
function, given the observations. The resulting estimate is called amaximum likelihood estimate,
which is also abbreviated as MLE.

6.5.1 The general MLE method

Suppose we have a random sampleX={x 1 ;:::;xn}taken from a probability distribution having
the probability mass function or probability density functionp(xS)wherexdenotes a value of the
random variable anddenotes the set of parameters that appear in the function.
Thelikelihoodof sampleXis a function of the parameterand is defined as

```
l()=p(x 1 S)p(x 2 S):::p(xnS):
```
In maximum likelihood estimation, we find the value ofthat makes the value of the likelihood
function maximum. For computation convenience, we define thelog likelihood functionas the
logarithm of the likelihood function:

```
L()=logl()
=logp(x 1 S)+logp(x 2 S)+ ⋯ +logp(xnS):
```

#### CHAPTER 6. BAYESIAN CLASSIFIER AND ML ESTIMATION 69

A value ofthat maximizesL()will also maximisel()and vice-versa. Hence, in maximum like-
lihood estimation, we findthat maximizes the log likelihood function. Sometimes the maximum
likelihood estimate ofis denoted by^.

6.5.2 Special cases

1. Bernoulli density

In a Bernoulli distribution there are two outcomes: An event occurs or it does not, for example, an
instance is a positive example of the class, or it is not. The event occurs and the Bernoulli random
variableXtakes the value 1 with probabilityp, and the nonoccurrence of the event has probability
1 −pand this is denoted byXtaking the value 0.
The probability function ofXis given by
f(xSp)=px( 1 −p)^1 −x; x= 0 ; 1 :

In this function, the probabilitypis the only parameter.

Estimation ofp

Consider a random sampleX={x 1 ;:::;xn}taken from a Bernoulli distribution with the probability
functionf(xSp). The log likelihood function is

```
L(p)=logf(x 1 Sp)+ ⋯ +logf(xnSp)
=logpx^1 ( 1 −p)^1 −x^1 + ⋯ +logpxn( 1 −p)^1 −xn
=[x 1 logp+( 1 −x 1 )log( 1 −p)]+ ⋯ +[xnlogp+( 1 −xn)log( 1 −p)]
```
To find the value ofpthat maximizesL(p)we set up the equation

```
dL
dp
```
#### = 0 ;

that is,

```

```
```
x 1
p
```
#### −

```
1 −x 1
1 −p
```
#### + ⋯ +

```
xn
p
```
#### −

```
1 −xn
1 −p
```
#### = 0 :

Solving this equation, we have the maximum likelihood estimate ofpas

```
p^=
```
#### 1

```
n
```
```
(x 1 + ⋯ +xn):
```
2. Multinomial density

Suppose that the outcome of a random event is one ofKclasses, each of which has a probability of
occurringpiwith
p 1 + ⋯ +pK= 1 :

We represent each outcome by an orderedK-tuplex=(x 1 ;:::;xK)where exactly one ofx 1 ;:::;xK
is 1 and all others are 0 .xi= 1 if the outcome in thei-th class occurs. The probability function can
be expressed as
f(xSp;:::;pK)=px 11 :::pxKK:

Here,p 1 ;:::;pKare the parameters.
We choosenrandom samples. Thei-the sample may be represented by
xi=(x 1 i;:::;xKi):

The values of the parameters that maximizes the likelihood function can be shown to be

```
p^k=
```
#### 1

```
n
```
```
(xk 1 +xk 2 + ⋯ +xkn):
```
(We leave the details of the derivation as an exercise.)


#### CHAPTER 6. BAYESIAN CLASSIFIER AND ML ESTIMATION 70

3. Gaussian (normal) density

A continuous random variableXhas the Gaussian or normal distribution if its density function is

```
f(xS;)=
```
#### 1

#### 

#### √

#### 2 

```
exp−
```
```
(x−)^2
2 ^2
```
```
; −∞<x<∞:
```
Hereandare the parameters.
Given a samplex 1 ;x 2 ;:::;xnfrom the distribution. the log likelihood function is

#### L(;)=−

```
n
2
```
```
log( 2 )−nlog−
```
#### 1

#### 2 ^2

```
(x 1 −)^2 + ⋯ +(xn−)^2 :
```
Setting up the equations
dL
d

#### = 0 ;

```
dL
d
```
#### = 0

and solving forandwe get the maximum likelihood estimates ofandas

#### ^=

#### 1

```
n
```
```
(x 1 + ⋯ +xn)
```
#### ^^2 =

#### 1

```
n
```
```
((x 1 −^)^2 + ⋯ +(xn−^)^2 )
```
(We leave the details of the derivation as an exercise.)

### 6.6 Sample questions

(a) Short answer questions

1. What are the assumptions under the naive Bayes algorithm?
2. Why is naive Bayes algorithm “naive”?
3. Given an instanceXof a feature vector and a class labelck, explain how Bayes theorem is
    used to compute the probabilityP(ckSX).
4. What does a naive Bayes classifier do?
5. What is naive Bayes used for?
6. Is naive Bayes supervised or unsupervised? Why?
7. What is meant by the likelihood of a random sample taken from population?
8. How do we use numeric features in naive Bayes algorithm?

(b) Long answer questions

1. State Bayes theorem and illustrate it with an example.
2. Explain naive Bayes algorithm.
3. Use naive Bayes algorithm to determine whether a red domestic SUV car is a stolen car or not
    using the following data:


#### CHAPTER 6. BAYESIAN CLASSIFIER AND ML ESTIMATION 71

```
Example no. Colour Type Origin Whether stolen
1 red sports domestic yes
2 red sports domestic no
3 red sports domestic yes
4 yellow sports domestic no
5 yellow sports imported yes
6 yellow SUV imported no
7 yellow SUV imported yes
8 yellow SUV domestic no
9 red SUV imported no
10 red sports imported yes
```
4. Based on the following data determine the gender of a person having height 6 ft., weight 130
    lbs. and foot size 8 in. (use naive Bayes algorithm).

```
person height (feet) weight (lbs) foot size (inches)
male 6.00 180 10
male 6.00 180 10
male 5.50 170 8
male 6.00 170 10
female 5.00 130 8
female 5.50 150 6
female 5.00 130 6
female 6.00 150 8
```
5. Given the following data on a certain set of patients seen by a doctor, can the doctor conclude
    that a person having chills, fever, mild headache and without running nose has the flu?

```
chills running nose headache fever has flu
Y N mild Y N
Y Y no N Y
Y N strong Y Y
N Y mild Y Y
N N no N N
N Y strong Y Y
N Y strong N N
Y Y mild Y Y
```
6. Explain the general MLE method for estimating the parameters of a probability distribution.
7. Find the ML estimate for the parameterpin the binomial distribution whose probability func-
    tion is
       f(x)=

```
n
x
```
```
px( 1 −p)n−x; x= 0 ; 1 ; 2 ;:::;n
```
8. Compute the ML estimate for the parameterin the Poisson distribution whose probability
    function is
       f(x)=e−

```
x
x!
```
```
; x= 0 ; 1 ; 2 ;:::
```
```
Find the ML estimate of the parameterpin the geometric distribution defined by the proba-
bility mass function
f(x)=( 1 −p)px; x= 1 ; 2 ; 3 ;:::
```

Chapter 7

## Regression

We have seen in Section 1.5.3 that regression is a supervised learning problem where there is an
inputxan outputyand the task is to learn the mapping from the input to the output. We have also
seen that the approach in machine learning is that we assume a model, that is, a relation betweenx
andycontaining a set of parameters, say,in the following form:

```
y=g(x;):
```
g(x;)is the regression function. The machine learning program optimizes the parameterssuch
that the approximation error is minimized, that is, our estimates are as close as possible to the correct
values given in the training set. In this chapter we discuss a method, known as ordinary least squares
method, to estimate the parameters. In fact this method can be derived from the maximum likelihood
estimation method discussed in Section 6.5.

### 7.1 Definition

Aregression problemis the problem of determining a relation between one or more independent
variables and an output variable which is a real continuous variable, given a set of observed values
of the set of independent variables and the corresponding values of the output variable.

7.1.1 Examples

1. Let us say we want to have a system that can predict the price of a used car. Inputs are the
    car attributes âA ̆T brand, year, engine capacity, mileage, and other information âˇ A ̆T that weˇ
    believe affect a car’s worth. The output is the price of the car.
2. Consider the navigation of a mobile robot, say an autonomous car. The output is the angle by
    which the steering wheel should be turned at each time, to advance without hitting obstacles
    and deviating from the route. Inputs are provided by sensors on the car like a video camera,
    GPS, and so forth.
3. In finance, the capital asset pricing model uses regression for analyzing and quantifying the
    systematic risk of an investment.
4. In economics, regression is the predominant empirical tool. For example, it is used to predict
    consumption spending, inventory investment, purchases of a country’s exports, spending on
    imports, labor demand, and labor supply.

7.1.2 Different regression models

The different regression models are defined based on type of functions used to represent the relation
between the dependent variableyand the independent variables.

#### 72


#### CHAPTER 7. REGRESSION 73

1. Simple linear regression
    Assume that there is only one independent variablex. If the relation betweenxandyis
    modeled by the relation
       y=a+bx
    then we have a simple linear regression.
2. Multiple regression
    Let there be more than one independent variable, sayx 1 ,x 2 ,:::,xn, and let the relation
    betweenyand the independent variables be modeled as

y= 0 + 1 x 1 + ⋯ + (^) nxn
then it is case of multiple linear regression or multiple regression.

3. Polynomial regression
    Let there be only one variablexand let the relation betweenx ybe modeled as

```
y=a 0 +a 1 x+a 2 x^2 + ⋯ +anxn
```
```
for some positive integern> 1 , then we have a polynomial regression.
```
4. Logistic regression
    Logistic regression is used when the dependent variable is binary (0/1, True/False, Yes/No)
    in nature. Even though the output is a binary variable, what is being sought is a probability
    function which may take any value from 0 to 1.

### 7.2 Criterion for minimisation of error

In regression, we would like to write the numeric outputy, called the dependent variable, as a
function of the inputx, called the independent variable. We assume that the output is the sum of a
functionf(x)of the input and some random error denoted by:

```
y=f(x)+:
```
Here the functionf(x)is unknown and we would like to approximate it by some estimatorg(x;)
containing a set of parameters. We assume that the random errorfollows normal distribution
with mean 0.
Letx 1 ;:::;xnbe a random sample of observations of the input variablexandy 1 ;:::;ynthe
corresponding observed values of the output variabley.
Using the assumption that the errorfollows normal distribution, we can apply the method of
maximum likelihood estimation to estimate the values of the parameter. It can be shown that the
values ofwhich maximizes the likelihood function are the values ofthat minimizes the following
sum of squares:
E()=(y 1 −g(x 1 ;))^2 + ⋯ +(yn−g(xn;))^2 :

The method of finding the value ofas that value ofthat minimizesE()is known as theordinary
least squares method.
The full details of the derivation of the above result are beyond the scope of these notes.


#### CHAPTER 7. REGRESSION 74

```
x x 1 x 2 ⋯ xn
y y 1 y 2 ⋯ yn
```
```
Table 7.1: Data set for simple linear regression
```
```
x
```
```
y
```
```
Actual value
```
```
Predicted value
```
```
Error
```
```
Regression model
```
```
Figure 7.1: Errors in observed values
```
### 7.3 Simple linear regression

Letxbe the independent predictor variable andythe dependent variable. Assume that we have a set
of observed values ofxandy:
A simple linear regression model defines the relationship betweenxandyusing a line defined
by an equation in the following form:
y= +x

In order to determine the optimal estimates of and , an estimation method known asOrdinary
Least Squares(OLS) is used.

The OLS method

In the OLS method, the values ofy-intercept and slope are chosen such that they minimize the sum
of the squared errors; that is, the sum of the squares of the vertical distance between the predicted
y-value and the actualy-value (see Figure 7.1). Lety^ibe the predicted value ofyi. Then the sum of
squares of errors is given by

#### E=

```
n
Q
i= 1
```
```
(yi−y^i)^2
```
#### =

```
n
Q
i= 1
```
```
[yi−( +xi)]^2
```
So we are required to find the values of and such thatEis minimum. Using methods of calculus,
we can show that the values ofaandb, which are respectively the values of and for whichEis
minimum, can be obtained by solving the following equations.

```
n
Q
i= 1
```
```
yi=na+b
```
```
n
Q
i= 1
```
```
xi
```

#### CHAPTER 7. REGRESSION 75

```
n
Q
i= 1
```
```
xiyi=a
```
```
n
Q
i= 1
```
```
xi+b
```
```
n
Q
i= 1
```
```
x^2 i
```
Formulas to findaandb

Recall that the means ofxandyare given by

```
x=
```
#### 1

```
n
Qxi
```
```
y=
```
#### 1

```
n
Qyi
```
and also that the variance ofxis given by

```
Var(x)=
```
#### 1

```
n− 1
Q(xi−xi)^2 :
```
Thecovariance ofxandy, denoted by Cov(x;y)is defined as

```
Cov(x;y)=
```
#### 1

```
n− 1
```
```
Q(xi−x)(yi−y)
```
It can be shown that the values ofaandbcan be computed using the following formulas:

```
b=
```
```
Cov(x;y)
Var(x)
a=y−bx
```
Remarks

It is interesting to note why the least squares method discussed above is christened as “ordinary”
least squares method. Several different variants of the least squares method have been developed
over the years. For example, in theweighted least squaresmethod, the coefficientsaandbare
estimated such that the weighted sum of squares of errors

#### E=

```
n
Q
i= 1
```
```
wi[yi−(a+bxi)]^2 ;
```
for some positive constantsw 1 ;:::;wn, is minimum. There are also methods known by the names
generalised least squaresmethod,partial least squaresmethod,total least squaresmethod, etc. The
reader may refer toWikipedia, a free online encyclopedia, to obtain further information about these
methods.
The OLS method has a long history. The method is usually credited to Carl Friedrich Gauss
(1795), but it was first published by Adrien-Marie Legendre (1805).

Example

Obtain a linear regression for the data in Table 7.2 assuming thatyis the independent variable.

```
x 1 : 0 2 : 0 3 : 0 4 : 0 5 : 0
y 1 : 00 2 : 00 1 : 30 3 : 75 2 : 25
```
```
Table 7.2: Example data for simple linear regression
```

#### CHAPTER 7. REGRESSION 76

```
Figure 7.2: Regression model for Table 7.2
```
Solution

In the usual notations of simple linear regression, we have

```
n= 5
```
```
x=
```
#### 1

#### 5

#### ( 1 : 0 + 2 : 0 + 3 : 0 + 4 : 0 + 5 : 0 )

#### = 3 : 0

```
y=
```
#### 1

#### 5

#### ( 1 : 00 + 2 : 00 + 1 : 30 + 3 : 75 + 2 : 25 )

#### = 2 : 06

```
Cov(x;y)=
```
#### 1

#### 4

#### [( 1 : 0 − 3 : 0 )( 1 : 00 − 2 : 06 )+ ⋯ +( 5 : 0 − 3 : 0 )( 2 : 25 − 2 : 06 )]

#### = 1 : 0625

```
Var(x)=
```
#### 1

#### 4

#### [( 1 : 0 − 3 : 0 )^2 + ⋯ +( 5 : 0 − 3 : 0 )^2 ]

#### = 2 : 5

```
b=
```
#### 1 : 0625

#### 2 : 5

#### = 0 : 425

```
a= 2 : 06 − 0 : 425 × 3 : 0
= 0 : 785
```
Therefore, the linear regression model for the data is

```
y= 0 : 785 + 0 : 425 x: (7.1)
```
Remark

Figure 7.2 in page 76 shows the data in Table 7.2 and the line given by Eq. (7.1). The figure was
created using R.


#### CHAPTER 7. REGRESSION 77

### 7.4 Polynomial regression

Letxbe the independent predictor variable andythe dependent variable. Assume that we have a set
of observed values ofxandyas in Table 7.1 in page 74.
A polynomial regression model defines the relationship betweenxandyby an equation in the
following form:

y= 0 + 1 x+ 2 x^2 + ⋯ + (^) kxk:
To determine the optimal values of the parameters 0 , 1 ,:::, (^) kthe method of ordinary least
squares is used. The values of the parameters are those values which minimizes the sum of squares:

#### E=

```
n
Q
i= 1
```
[yi−( 0 + 1 xi+ 2 x^2 i+ ⋯ + (^) kxki)]^2 :
The optimal values of the parameters are obtained by solving the following system of equations:
@E
@i
= 0 ; i= 0 ; 1 ;:::;k: (7.2)
Let the values of values of the parameters which minimizesEbe
(^) i=ai; i= 0 ; 1 ; 2 ;:::;n: (7.3)
Simplifying Eq. (7.2) and using Eq. (7.3), we can see that the values ofaican be obtained by
solving the the following system of(k+ 1 )linear equations:
Qyi=^0 n+^1 (Qxi)+ ⋯ +^ k(Qxki)
Qyixi=^0 (Qxi)+^1 (Qx^2 i)+ ⋯ +^ k(Qxki+^1 )
Qyix^2 i= 0 (Qx^2 i)+ 1 (Qx^3 i)+ ⋯ + (^) k(Qxki+^2 )
⋮
Qyixki=^0 (Qxki)+^1 (Qxki+^1 )+ ⋯ +^ k(Qx^2 ik)
Solving this system of linear equations we get the optimal values for the parameters.
Remarks
The linear system of equations to findai’s, has a compact matrix representation. We write:

#### D=

#### ⎡⎢

#### ⎢⎢

#### ⎢⎢

#### ⎢⎢

#### ⎢

#### ⎣

```
1 x 1 x^21 ⋯ xk 1
1 x 2 x^22 ⋯ xk 2
⋮
1 xn x^2 n ⋯ xkn
```
#### ⎤⎥

#### ⎥⎥

#### ⎥⎥

#### ⎥⎥

#### ⎥

#### ⎦

```
; y⃗=
```
#### ⎡⎢

#### ⎢⎢

#### ⎢⎢

#### ⎢⎢

#### ⎢

#### ⎣

```
y 1
y 2
⋮
yn
```
#### ⎤⎥

#### ⎥⎥

#### ⎥⎥

#### ⎥⎥

#### ⎥

#### ⎦

```
; ⃗a=
```
#### ⎡⎢

#### ⎢⎢

#### ⎢⎢

#### ⎢⎢

#### ⎢

#### ⎣

```
a 0
a 1
⋮
ak
```
#### ⎤⎥

#### ⎥⎥

#### ⎥⎥

#### ⎥⎥

#### ⎥

#### ⎦

Then we have
⃗a=(DTD)−^1 DTy;⃗

where the superscriptTdenotes the transpose of the matrix.

7.4.1 Example

Find a quadratic regression model for the following data:

```
x 3 4 5 6 7
y 2.5 3.2 3.8 6.5 11.5
```

#### CHAPTER 7. REGRESSION 78

```
Figure 7.3: Plot of quadratic polynomial model
```
Solution

Let the quadratic regression model be

```
y= 0 + 1 x+ 2 x^2 :
```
The values of 0 , 1 and 2 which minimises the sum of squares of errors area 0 ,a 1 anda 2 which
satisfy the following system of equations:

```
Qyi=na 0 +a 1 (Qxi)+a 2 (Qx^2 i)
Qyixi=a 0 (Qxi)+a 1 (Qx^2 i)+a 2 (Qx^3 i)
Qyix^2 i=a 0 (Qx^2 i)+a 1 (Qx^3 i)+a 2 (Qx^4 i)
```
Using the given data we have

```
27 : 5 = 5 a 0 + 25 a 1 + 135 a 2
158 : 8 = 25 a 0 + 135 a 1 + 775 a 2
966 : 2 = 135 a 0 + 775 a 1 + 4659 a 2
```
Solving this system of equations we get

```
a 0 = 12 : 4285714
a 1 =− 5 : 5128571
a 2 = 0 : 7642857
```
The required quadratic polynomial model is

```
y= 12 : 4285714 − 5 : 5128571 x+ 0 : 7642857 x^2 :
```
Figure 7.3 shows plots of the data and the quadratic polynomial model.

### 7.5 Multiple linear regression

We assume that there areNindependent variablesx 1 ,x 2 ,⋯,xN. Let the dependent variable bey.
Let there also benobserved values of these variables:


#### CHAPTER 7. REGRESSION 79

```
Variables Values (examples)
(features) Example 1 Example 2 ⋯ Examplen
x 1 x 11 x 12 ⋯ x 1 n
x 2 x 21 x 22 ⋯ x 2 n
⋯
xN xN 1 xN 2 ⋯ xNn
y(outcomes) y 1 y 2 ⋯ yn
```
```
Table 7.3: Data for multiple linear regression
```
The multiple linear regression model defines the relationship between theNindependent vari-
ables and the dependent variable by an equation of the following form:

y= 0 + 1 x 1 + ⋯ + (^) NxN
As in simple linear regression, here also we use the ordinary least squares method to obtain the
optimal estimates of 0 , 1 ,⋯, (^) N. The method yields the following procedure for the computation
of these optimal estimates. Let

#### X=

#### ⎡⎢

#### ⎢⎢

#### ⎢⎢

#### ⎢⎢

#### ⎢⎣

```
1 x 11 x 21 ⋯ xN 1
1 x 12 x 22 ⋯ xN 2
⋮
1 x 1 n x 2 n ⋯ xNn
```
#### ⎤⎥

#### ⎥⎥

#### ⎥⎥

#### ⎥⎥

#### ⎥⎦

#### ; Y=

#### ⎡⎢

#### ⎢⎢

#### ⎢⎢

#### ⎢⎢

#### ⎢⎣

```
y 1
y 2
⋮
yn
```
#### ⎤⎥

#### ⎥⎥

#### ⎥⎥

#### ⎥⎥

#### ⎥⎦

#### ; B=

#### ⎡⎢

#### ⎢⎢

#### ⎢⎢

#### ⎢⎢

#### ⎢⎣

(^0)
(^1)
⋮
(^) N

#### ⎤⎥

#### ⎥⎥

#### ⎥⎥

#### ⎥⎥

#### ⎥⎦

Then it can be shown that the regression coefficients are given by

```
B=(XTX)−^1 XTY
```
7.5.1 Example

Example

Fit a multiple linear regression model to the following data:

```
x 1 1 1 2 0
x 2 1 2 2 1
y 3.25 6.5 3.5 5.0
```
```
Table 7.4: Example data for multi-linear regression
```
Solution

In this problem, there are two independent variables andfour sets of values of the variables. Thus,
in the notations used above, we haven= 2 andN= 4. The multiple linear regression model for this
problem has the form
y= 0 + 1 x 1 + 2 x 2 :
The computations are shown below.

#### X=

#### ⎡

#### ⎢⎢

#### ⎢⎢

#### ⎢⎢

#### ⎢⎢

#### ⎣

#### 1 1 1

#### 1 1 2

#### 1 2 2

#### 1 0 1

#### ⎤

#### ⎥⎥

#### ⎥⎥

#### ⎥⎥

#### ⎥⎥

#### ⎦

#### ; Y=

#### ⎡

#### ⎢⎢

#### ⎢⎢

#### ⎢⎢

#### ⎢⎢

#### ⎣

#### 3 : 25

#### 6 : 5

#### 3 : 5

#### 5 : 0

#### ⎤

#### ⎥⎥

#### ⎥⎥

#### ⎥⎥

#### ⎥⎥

#### ⎦

#### ; B=

#### ⎡⎢

#### ⎢⎢

#### ⎢⎢

#### ⎣

(^0)
(^1)
(^2)

#### ⎤⎥

#### ⎥⎥

#### ⎥⎥

#### ⎦


#### CHAPTER 7. REGRESSION 80

#### XTX=

#### ⎡⎢

#### ⎢⎢

#### ⎢⎢

#### ⎣

#### 4 4 6

#### 4 6 7

#### 6 7 10

#### ⎤⎥

#### ⎥⎥

#### ⎥⎥

#### ⎦

#### (XTX)−^1 =

#### ⎡

#### ⎢⎢

#### ⎢⎢

#### ⎢

#### ⎣

```
11
4
```
```
1
1 2 −^2
2 1 −^1
− 2 − 1 2
```
#### ⎤

#### ⎥⎥

#### ⎥⎥

#### ⎥

#### ⎦

#### B=(XTX)−^1 XTY

#### =

#### ⎡⎢

#### ⎢⎢

#### ⎢

#### ⎢⎣

#### 2 : 0625

#### − 2 : 3750

#### 3 : 2500

#### ⎤⎥

#### ⎥⎥

#### ⎥

#### ⎥⎦

The required model is
y= 2 : 0625 − 2 : 3750 x 1 + 3 : 2500 x 2 :

```
x 1
```
```
x 2
```
```
y
```
#### (1,1,3.25)

#### (1,2,6.25)

#### (2,2,3.25)

#### (0,1,5.0)

```
y= 2 : 0625 − 2 : 3750 x 1 + 3 : 2500 x 2
```
```
Figure 7.4: The regression plane for the data in Table 7.4
```
### 7.6 Sample questions

(a) Short answer questions

1. What are the different types of regression.
2. Is regression a supervised learning? Why?
3. Explain the ordinary least squares method for regression.
4. What are linear, multinomial and polynomial regressions.
5. If model used for regression is
    y=a+b(x− 1 )^2 ;
is it a multinomial regression? If not, what type of regression is it?
6. What does the line of regression tell you?


#### CHAPTER 7. REGRESSION 81

(b) Long answer questions

1. Discuss linear regression with an example.
2. In the table below, thexirow shows scores in an aptitude test. Similarly, theyirow shows
    statistics grades. If a student made an 80 on the aptitude test, what grade would we expect her
    to make in statistics?

```
Studenti 1 2 3 4 5
xi 95 85 80 70 60
yi 85 95 70 65 70
```
3. Use the following data to construct a linear regression model for the auto insurance premium
    as a function of driving experience.

```
Driving experience (in years) 5 2 12 9 15 6 25 16
Monthly auto insurance premium ($) 64 87 50 71 44 56 42 60
```
4. Determine the regression equation by finding the regression slope coefficient and the intercept
    value using the following data.

```
x 55 60 65 70 80
y 52 54 56 58 62
```
5. The following table contains measurements of yield from an experiment done at five different
    temperature levels. The variables arey= yield andx= temperature in degrees Fahrenheit.
    Compute a second degree polynomial regression model to predict the yield given the temper-
    ature.

```
Temperature (x) Yield (y)
50 3.0
70 2.7
80 2.6
90 2.9
100 3.3
```
6. An experiment was done to assess how moisture content and sweetness of a pastry product
    affect a tasterâA ̆Zs rating of the product. The following table summarises the findings. ́

```
Rating Moisture Sweetness
64 4 2
73 4 4
61 4 2
76 4 4
72 6 2
80 6 4
71 6 2
83 6 4
83 8 2
89 8 4
86 8 2
93 8 4
88 10 2
95 10 4
94 10 2
100 10 4
```

#### CHAPTER 7. REGRESSION 82

```
Compute a linear regression model to predict the rating of the pastry product.
```
7. The following data contains the Performance IQ scores (PIQ) (in appropriate scales), brain
    sizes (in standard units), heights (in inches) and weights (in pounds) of 15 American college
    students. Obtain a linear regression model to predict the PIQ given the values of the other
    features.

```
PIQ Brain Height Weight
124 81.69 64.5 118
150 103.84 73.3 143
128 96.54 68.8 172
134 95.15 65.0 147
110 92.88 69.0 146
131 99.13 64.5 138
98 85.43 66.0 175
84 90.49 66.3 134
147 95.55 68.8 172
124 83.39 64.5 118
128 107.95 70.0 151
124 92.41 69.0 155
147 85.65 70.5 155
90 87.89 66.0 146
96 86.54 68.0 135
```
8. Use the following data to generate a linear regression model for annual salary as function of
    GPA and number of months worked.

```
Example no. Annual salary ($) GPA Months worked
1 20000 2.8 48
2 24500 3.4 24
3 23000 3.2 24
4 25000 3.8 24
5 20000 3.2 48
6 22500 3.4 36
7 27500 4.0 24
8 19000 2.6 48
9 24000 3.2 36
10 28500 3.8 12
```

Chapter 8

## Decision trees

“Decision tree learning is a method for approximating discrete valued target functions, in which the
learned function is represented by a decision tree. Decision tree learning is one of the most widely
used and practical methods for inductive inference.” ([4] p.52)

### 8.1 Decision tree: Example

Consider the following situation. Somebody is hunting for a job. At the very beginning, he decides
that he will consider only those jobs for which the monthly salary is at least Rs.50,000. Our job
hunter does not like spending much time traveling to place of work. He is comfortable only if the
commuting time is less than one hour. Also, he expects the company to arrange for a free coffee
every morning! The decisions to be made before deciding to accept or reject a job offer can be
schematically represented as in Figure 8.6. This figure represents adecision tree^1.

```
Root node
Salary≥Rs.50000?
```
```
Commute one hour?
```
```
Decline offer
```
```
Yes
```
```
Offers free coffee?
```
```
Accept offer
```
```
Yes
```
```
Decline offer
```
```
No
```
```
No
```
```
Yes
```
```
Decline offer
```
```
No
```
```
Figure 8.1: Example for a decision tree
```
Here, the term “tree” refers to the concept of a tree in graph theory in mathematics^2 .In graph
theory, a tree is defined as an undirected graph in which any two vertices are connected by exactly
one path. Using the conventions of graph theory, the decision tree shown in Figure 8.6 can be
represented as a graph-theoretical tree as in Figure 8.2. Since a decision tree is a graph-theoretical
tree, all terminology related to graph-theoretical trees can be applied to describe decision trees also.
For example, in Figure 8.6, the nodes or vertices shown as ellipses are called theleaf nodes. All
other nodes, except the root node, are called theinternal nodes.

(^1) In such diagrams, the “tree” is shown upside down with the root node at the top and all the leaves at the bottom.
(^2) The term “tree” was coined in 1857 by the British mathematician Arthur Cayley (see Wikipedia).

#### 83


#### CHAPTER 8. DECISION TREES 84

```
Root node
```
```
Yes
```
```
Yes No
```
```
No
```
```
Yes No
```
```
Figure 8.2: The graph-theoretical representation of the decision tree in Figure 8.6
```
### 8.2 Two types of decision trees

There are two types of decision trees.

1. Classification trees
    Tree models where the target variable can take a discrete set of values are calledclassification
    trees. In these tree structures, leaves represent class labels and branches represent conjunc-
    tions of features that lead to those class labels.
2. Regression trees
    Decision trees where the target variable can take continuous values (real numbers) like the
    price of a house, or a patient’s length of stay in a hospital, are calledregression trees.

### 8.3 Classification trees

We illustrate the concept with an example.

8.3.1 Example

Data

```
Nam
Features
Class label
gives birth
aquatic
animal
```
```
aerial
animal
has legs
```
```
human yes no no yes mammal
python no no no no reptile
salmon no yes no no fish
frog no semi no yes amphibian
bat yes no yes yes bird
pigeon no no yes yes bird
cat yes no no yes mammal
shark yes yes no no fish
turtle no semi no yes amphibian
salamander no semi no yes amphibian
```
```
Table 8.1: The vertebrate data set
```

#### CHAPTER 8. DECISION TREES 85

Consider the data given in Table 8.1 which specify the features of certain vertebrates and the class
to which they belong. For each species, four features have been identified: “gives birth”, ”aquatic
animal”, “aerial animal” and “has legs”. There are five class labels, namely, “amphibian”, “bird”,
“fish”, “mammal” and “reptile”. The problem is how to use this data to identify the class of a newly
discovered vertebrate.

Construction of the tree

Step 1

We split the set of examples given in Table 8.1 into disjoint subsets according to the values of the
feature “gives birth”. Since there are only two possible values for this feature, we have only two
subsets: One subset consisting of those examples for which the value of “gives birth” is “yes” and
one subset for which the value is “no”. The former is given in Table 8.2 and the latter in Table 8.3.
This stage of the classification can be represented as in Figure 8.3.

```
Name Gives
birth
```
```
Aquatic
animal
```
```
Aerial
animal
```
```
Has legs Class la-
bel
human yes no no yes mammal
bat yes no yes yes bird
cat yes no no yes mammal
shark yes yes no no fish
```
```
Table 8.2: The subset of Table 8.1 with “gives birth”=”yes"
```
```
Name gives birth aquatic
animal
```
```
aerial
animal
```
```
has legs Class la-
bel
python no no no no reptile
salmon no yes no no fish
frog no semi no yes amphibian
pigeon no no yes yes bird
turtle no semi no yes amphibian
salamander no semi no yes amphibian
```
```
Table 8.3: The subset of Table 8.1 with “gives birth”=”no"
```
```
Root node
Table 8.1:
gives birth?
```
```
Table 8.2:
aquatic?
```
```
Yes
```
```
Table 8.3:
aquatic?
```
```
No
```
```
Figure 8.3: Classification tree
```
Step 2

We now consider the examples in Table 8.2. We split these examples based on the values of the
feature “aquatic animal”. There are three possible values for this feature. However, only two of


#### CHAPTER 8. DECISION TREES 86

```
Name gives birth aquatic
animal
```
```
aerial
animal
```
```
has legs Class la-
bel
human yes no no yes mammal
bat yes no yes yes bird
cat yes no no yes mammal
```
```
Table 8.5: The vertebrate data set
```
```
Root node
Table 8.1:
gives birth?
```
```
Table 8.2:
aquatic?
```
```
Table 8.4
```
```
fish
```
```
yes
```
```
Table 8.5:
aerial?
```
```
Part of
Table 8.5
```
```
bird
```
```
yes
```
```
Part of
Table 8.5
```
```
mammal
```
```
no
```
```
no
```
```
Yes
```
```
Table 8.3:
aquatic?
```
```
no
```
```
Figure 8.4: Classification tree
```
these appear in Table 8.2. Accordingly, we need consider only two subsets. These are shown in
Tables 8.4 and 8.5.

```
Name gives birth aquatic
animal
```
```
aerial
animal
```
```
has legs Class la-
bel
shark yes yes no no fish
```
```
Table 8.4: The vertebrate data set
```
- Table 8.4 contains only one example and hence no further splitting is required. It leads to the
    assignment of the class label “fish”.
- The examples in Table 8.5 need to be split into subsets based on the values of “aerial animal”.
    It can be seen that these subsets immediately lead to unambiguous assignment of class labels:
    The value of “no” leads to “mammal” and the value “yes” leads to ”bird”.

At this stage, the classification tree is as shown in Figure 8.4


#### CHAPTER 8. DECISION TREES 87

Step 3

Next we consider the examples in Table 8.3 and split them into disjoint subsets based on the values
of “aquatic animal”. We get the examples in Table 8.6 for “yes”, the examples in Table??for “no”
and the examples in Table??for “semi”. We now split the resulting subsets based on the values of

```
Name gives birth aquatic
animal
```
```
aerial
animal
```
```
has legs Class la-
bel
salmon no yes no no fish
```
```
Table 8.6: The vertebrate data set
```
```
Name gives birth aquatic
animal
```
```
aerial
animal
```
```
has legs Class la-
bel
frog no semi no yes amphibian
turtle no semi no yes amphibian
salamander no semi no yes amphibian
```
```
Table 8.7: The vertebrate data set
```
```
Name gives birth aquatic
animal
```
```
aerial
animal
```
```
has legs Class la-
bel
python no no no no reptile
pigeon no no yes yes bird
```
```
Table 8.8: The vertebrate data set
```
“has legs”, etc. Putting all these together, we get the the diagram in Figure 8.5 as the classification
tree for the data in Table 8.1.

8.3.2 Classification tree in rule format

The classification tree shown in Figure 8.5 can be presented as a set of rules in the form of an
algorithm.

Algorithm for classification of vertebrates

```
1.ifgive birth = ”yes”then
```
2. ifaquatic = “yes”then
3. return class = “fish”
4. else
5. ifaerial = “yes”then
6. return class = “bird”
7. else
8. return class = “mammal”
9. end if
10. end if
11. else
12. ifaquatic = “yes”then
13. return class = “fish”


#### CHAPTER 8. DECISION TREES 88

```
Root node
Table 8.1:
gives birth?
```
```
Table 8.2:
aquatic?
```
```
Table 8.4
```
```
fish
```
```
yes
```
```
Table 8.5:
aerial?
```
```
Part of
Table 8.5
```
```
bird
```
```
yes
```
```
Part of
Table 8.5
```
```
mammal
```
```
no
```
```
no
```
```
yes
```
```
Table 8.3:
aquatic?
```
```
Table 8.6
```
```
fish
```
```
yes
```
```
Table 8.7
```
```
amph
```
```
semi
```
```
Table 8.8
aerial?
```
```
Part of
Table 8.8
```
```
bird
```
```
yes
```
```
Part of
Table 8.8
```
```
reptile
```
```
no
```
```
no
```
```
no
```
```
Figure 8.5: Classification tree
```
14. end if
15. ifaquatic = “semi”then
16. return class = “amphibian”
17. else
18. ifaerial = “yes”then
19. return class = “amphibian”
20. else
21. return class = “reptile”
22. end if
23. end if
24. end if

8.3.3 Some remarks

1. On the elements of a classification tree

The various elements in a classification tree are identified as follows.

- Nodes in the classification tree are identified by the feature names of the given data.
- Branches in the tree are identified by the values of features.
- The leaf nodes identified by are the class labels.


#### CHAPTER 8. DECISION TREES 89

2. On the order in which the features are selected

In the example discussed above, initially we chose the feature “gives birth” to split the data set
into disjoint subsets and then the feature “aquatic animal”, and so on. There was no theoretical
justification for this choice. We could as well have chosen the feature “aquatic animal”, or any other
feature, as the initial feature for splitting the data. The classification tree depends on the order in
which the features are selected for partitioning the data.

3. Stopping criteria

A real-world data will contain much more example record than the example we considered earlier.
In general, there will be a large number of features each feature having several possible values. Thus,
the corresponding classification trees will naturally be more complex. In such cases, it may not be
advisable to construct all branches and leaf nodes of the tree. The following are some of commonly
used criteria for stopping the construction of further nodes and branches.

- All (or nearly all) of the examples at the node have the same class.
- There are no remaining features to distinguish among the examples.
- The tree has grown to a predefined size limit.

### 8.4 Feature selection measures

If a dataset consists ofnattributes then deciding which attribute is to be to placed at the root or at
different levels of the tree as internal nodes is a complicated problem. It is not enough that we just
randomly select any node to be the root. If we do this, it may give us bad results with low accuracy.
The most important problem in implementing the decision tree algorithm is deciding which
features are to be considered as the root node and at each level. Several methods have been developed
to assign numerical values to the various features such that the values reflect the relative importance
of the various features. These are called thefeature selection measures. Two of the popular feature
selection measures areinformation gainandGini index. These are explained in the next section.

### 8.5 Entropy

The degree to which a subset of examples contains only a single class is known aspurity, and any
subset composed of only a single class is called apureclass. Informally, entropy^3 is a measure of
“impurity” in a dataset. Sets with high entropy are very diverse and provide little information about
other items that may also belong in the set, as there is no apparent commonality.
Entropy is measured in bits. If there are only two possible classes, entropy values can range from
0 to 1. Fornclasses, entropy ranges from 0 tolog 2 (n). In each case, the minimum value indicates
that the sample is completely homogeneous, while the maximum value indicates that the data are as
diverse as possible, and no group has even a small plurality.

8.5.1 Definition

Consider a segmentSof a dataset havingcnumber of class labels. Letpibe the proportion of
examples inShaving theith class label. The entropy ofSis defined as

```
Entropy(S)=
```
```
c
Q
i= 1
```
```
−pilog 2 (pi):
```
(^3) From German Entropie “measure of the disorder of a system,” coined in 1865 (on analogy of Energie) by German
physicist Rudolph Clausius (1822-1888), in his work on the laws of thermodynamics, from Greek entropia “a turning toward,”
from en “in” + trope “a turning, a transformation,”


#### CHAPTER 8. DECISION TREES 90

```
Figure 8.6: Plot ofpvs. Entropy
```
Remark

In the expression for entropy, the value of 0 ×log 2 ( 0 )is taken as zero.

Special case

Let the data segmentShas only two class labels, say, “yes” and “no”. Ifpis the proportion of
examples having the label “yes” then the proportion of examples having label “no” will be 1 −p. In
this case, the entropy ofSis given by

```
Entropy(S)=−plog 2 (p)−( 1 −p)log 2 ( 1 −p):
```
If we plot the values of graph of Entropy(S)for all possible values ofp, we get the diagram shown
in Figure 8.6^4.

8.5.2 Examples

Let “xxx” be some class label. We denote bypxxxthe proportion of examples with class label “xxx”.

1. Entropy of data in Table 8.1
    LetSbe the data in Table 8.1. The class labels are ”amphi”, “bird”, ”fish”, ”mammal” and
    ”reptile”. InSwe have the following numbers.

```
Number of examples with class label “amphi” = 3
Number of examples with class label “bird” = 2
Number of examples with class label “fish” = 2
Number of examples with class label “mammal” = 2
Number of examples with class label “reptile” = 1
Total number of examples = 10
```
```
Therefore, we have:
```
```
Entropy(S)= Q
for all classes “xxx”
```
```
−pxxxlog 2 (pxxx)
```
(^4) Plot created using R language.


#### CHAPTER 8. DECISION TREES 91

```
=−pamphilog 2 (pamphi)−pbirdlog 2 (pbird)
−pfishlog 2 (pfish)−pmammallog 2 (pmammal)
−preptilelog 2 (preptile)
=−( 3 ~ 10 )log 2 ( 3 ~ 10 )−( 2 ~ 10 )log 2 ( 2 ~ 10 )
−( 2 ~ 10 )log 2 ( 2 ~ 10 )−( 2 ~ 10 )log 2 ( 2 ~ 10 )
−( 1 ~ 10 )log 2 ( 1 ~ 10 )
= 2 : 2464
```
2. Entropy of data in Table 8.2
    Consider the segmentSof the data in Table 8.1 given in Table 8.2. For quick reference, the
    table has been reproduced below:

```
Name Gives
birth
```
```
Aquatic
animal
```
```
Aerial
animal
```
```
Has legs Class la-
bel
human yes no no yes mammal
bat yes no yes yes bird
cat yes no no yes mammal
shark yes yes no no fish
```
```
Three class labels appear in this segment, namely, “bird”, “fish” and “mammal”. We have:
```
```
Number of examples with class label “bird” 1
Number of examples with class label “fish” 1
Number of examples with class label “mammal” 2
Total number of examples 4
```
```
Therefore we have
```
```
Entropy(S)= Q
for all classes “xxx”
```
```
−pxxxlog 2 (pxxx)
```
```
=−pbirdlog 2 (pbird)−pfishlog 2 (pfish)
−pmammallog 2 (pmammal)
=−( 1 ~ 4 )log 2 ( 1 ~ 4 )−( 1 ~ 4 )log 2 ( 1 ~ 4 )−( 2 ~ 4 )log 2 ( 2 ~ 4 )
=−( 1 ~ 4 )×(− 2 )−( 1 ~ 4 )×(− 2 )−( 2 ~ 4 )×(− 1 )
= 1 : 5 (8.1)
```
3. Entropy of data in Table 8.3
    Consider the segmentSof the data in Table 8.1 given in Table 8.3. For quick reference, the
    table has been reproduced below:

```
Name gives birth aquatic
animal
```
```
aerial
animal
```
```
has legs Class la-
bel
python no no no no reptile
salmon no yes no no fish
frog no semi no yes amphibian
pigeon no no yes yes bird
turtle no semi no yes amphibian
salamander no semi no yes amphibian
```
```
Four class labels appear in this segment, namely, “amphi”, “bird”, “fish” and “reptile”. We
have:
```

#### CHAPTER 8. DECISION TREES 92

```
Number of examples with class label “amphi” 3
Number of examples with class label “bird” 1
Number of examples with class label “fish” 1
Number of examples with class label “reptile” 1
Total number of examples 6
```
```
Therefore, we have:
```
```
Entropy(S)= Q
for all classes “xxx”
```
```
−pxxxlog 2 (pxxx)
```
```
=−pamphilog 2 (pamphi)−pbirdlog 2 (pbird)−pfishlog 2 (pfish)
−preptilelog 2 (preptile)
=−( 3 ~ 6 )log 2 ( 3 ~ 6 )−( 1 ~ 6 )log 2 ( 1 ~ 6 )−( 1 ~ 6 )log 2 ( 1 ~ 6 )
−( 1 ~ 6 )log 2 ( 1 ~ 6 )
= 1 : 7925 (8.2)
```
### 8.6 Information gain

8.6.1 Definition

LetSbe a set of examples,Abe a feature (or, an attribute),Svbe the subset ofSwithA=v,
and Values(A)be the set of all possible values ofA. Then theinformation gain of an attributeA
relative to the setS, denoted by Gain(S;A), is defined as

```
Gain(S;A)=Entropy(S)− Q
v∈Values(A)
```
```
SSvS
SSS
```
```
×Entropy(Sv):
```
whereSSSdenotes the number of elements inS.

8.6.2 Example 1

Consider the dataSgiven in Table 8.1. We have have already seen that

```
SSS= 10
Entropy(S)= 2 : 2464 :
```
We denote the information gain corresponding to the feature “xxx” by Gain(S;xxx).

1. Computation of Gain(S;gives birth)

```
A 1 =gives birth
Values ofA 1 ={“yes”;“no”}
SA 1 =yes=Data in Table 8.2
SSA 1 =yesS= 4
Entropy(SA 1 =yes)= 1 : 5 (See Eq.(8.1))
SA 1 =no=Data in Table 8.3
SSA 1 =noS= 6
Entropy(SA 1 =no)= 1 : 7925 (See Eq.(8.2))
```
```
Now we have
```
```
Gain(S;A 1 )=Entropy(S)− Q
v∈Values(A 1 )
```
```
SSvS
SSS
```
```
×Entropy(Sv)
```

#### CHAPTER 8. DECISION TREES 93

```
=Entropy(S)−
```
```
SSA 1 =yesS
SSS
```
```
×Entropy(SA 1 =yes)
```
#### −

```
SSA 1 =noS
SSS
```
```
×Entropy(SA 1 =no)
```
```
= 2 : 2464 −( 4 ~ 10 )× 1 : 5 −( 6 ~ 10 )× 1 : 7925
= 0 : 5709
```
2. Computation of Gain(S;aquatic)

```
A 2 =aquatic
Values ofA 2 ={“yes”;“no”;“semi”}
SA 2 =yes=See Table 8.1
SSA 2 =yesS= 2
Entropy(SA 2 =yes)=−pfishlog 2 (pfish)
=−( 2 ~ 2 )log 2 ( 2 ~ 2 )
= 0
SA 2 =no=See Table 8.1
SSA 2 =noS= 5
Entropy(SA 2 =no)=−pmammallog 2 (pmammal)−preptilelog 2 (preptile)
−pbirdlog 2 (pbird)
=−( 2 ~ 5 )×log 2 ( 2 ~ 5 )−( 1 ~ 5 )×log 2 ( 1 ~ 5 )
−( 2 ~ 5 )×log 2 ( 2 ~ 5 )
= 1 : 5219
SA 2 =semi=See Table 8.1
SSA 2 =semiS= 3
Entropy(SA 2 =semi)=−pamphilog 2 (pamphi)
=−( 3 ~ 3 )×log 2 ( 3 ~ 3 )
= 0
```
```
Gain(S;A 2 )=Entropy(S)− Q
v∈Values(A 2 )
```
```
SSvS
SSS
```
```
×Entropy(Sv)
```
```
=Entropy(S)−
```
```
SSA 1 =yesS
SSS
```
```
×Entropy(SA 1 =yes)
```
#### −

```
SSA 1 =noS
SSS
```
```
×Entropy(SA 1 =no)
```
#### −

```
SSA 1 =semiS
SSS
```
```
×Entropy(SA 1 =semi)
```
```
= 2 : 2464 −( 2 ~ 10 )× 0 −( 5 ~ 10 )× 1 : 5219 −( 3 ~ 3 )× 0
= 1 : 48545
```
3. Computations of Gain(S;aerial animal)and Gain(S;has legs)
    These are left as exercises.

### 8.7 Gini indices

The Gini split index of a data set is another feature selection measure in the construction of classifi-
cation trees. This measure is used in the CART algorithm.


#### CHAPTER 8. DECISION TREES 94

8.7.1 Gini index

Consider a data setShavingrclass labelsc 1 ;:::;cr. Letpibe the proportion of examples having
the class labelci. The Gini index of the data setS, denoted by Gini(S), is defined by

```
Gini(S)= 1 −
```
```
r
Q
i= 1
```
```
p^2 i:
```
Example

LetSbe the data in Table 8.1. There are four class labels ”amphi”, “bird”, ”fish”, ”mammal” and
”reptile”. The numbers of examples having these class labels are as follows:

```
Number of examples with class label “amphi” = 3
Number of examples with class label “bird” = 2
Number of examples with class label “fish” = 2
Number of examples with class label “mammal” = 2
Number of examples with class label “reptile” = 1
Total number of examples = 10
```
The Gini index ofSis given by

```
Gini(S)= 1 −
```
```
r
Q
i= 1
```
```
p^2 i
```
```
= 1 −( 3 ~ 10 )^2 −( 2 ~ 10 )^2 −( 2 ~ 10 )^2 −( 2 ~ 10 )^2 −( 1 ~ 10 )^2
= 0 : 78
```
8.7.2 Gini split index

LetSbe a set of examples,Abe a feature (or, an attribute),Svbe the subset ofSwithA=v,
and Values(A)be the set of all possible values ofA. Then theGini split index ofArelative toS,
denoted by Ginisplit(S;A), is defined as

```
Ginisplit(S;A)= Q
v∈Values(A)
```
```
SSvS
SSS
```
```
×Gini(Sv):
```
whereSSSdenotes the number of elements inS.

### 8.8 Gain ratio

Thegain ratiois a third feature selection measure in the construction of classification trees.
LetSbe a set of examples,Aa feature havingcdifferent values and let the set of values ofAbe
denoted by Values(A). We defined the information gain ofArelative toS, denoted by Gain(S;A),
by

```
Gain(S;A)=Entropy(S)− Q
v∈Values(A)
```
```
SSvS
SSS
```
```
×Entropy(Sv):
```
We now define thesplit informationofArelative toS, dented by SplitInformation(S;A), by

```
SplitInformation(S;A)=−
```
```
c
Q
i= 1
```
```
SSiS
SSS
```
```
log 2
```
```
SSiS
SSS
```
whereS 1 ;:::Scare thecsubsets of examples resulting from partitioningSinto thecvalues of the
attributeA. Thegain ratioofArelative toS, denoted by GainRatio(S;A), by

```
GainRatio(S;A)=
```
```
Gain(S,A)
SplitInformation(S;A)
```
#### :


#### CHAPTER 8. DECISION TREES 95

8.8.1 Example

Consider the dataSgiven in Table 8.1. LetAdenote the attribute “gives birth”.We have have already
seen that

```
SSS= 10
Entropy(S)= 2 : 2464
Gain(S;A)= 0 : 5709
```
Now we have

```
SplitInformation(S;A)=−
```
```
SSyesS
SSS
```
```
log 2
```
```
SSyesS
SSS
```
#### −

```
SSnoS
SSS
```
```
log 2
```
```
SSnoS
SSS
```
```
=−
```
#### 4

#### 10

```
×log 2
```
#### 4

#### 10

#### −

#### 6

#### 10

```
×log 2
```
#### 6

#### 10

#### = 0 : 9710

```
GainRatio=
```
#### 0 : 5709

#### 0 : 9710

#### = 0 : 5880

In a similar way we can compute the gain ratios Gain(S;“aquatic”), Gain(S;“aerial”)and Gain(S;“has legs”).

### 8.9 Decision tree algorithms

8.9.1 Outline

Decision tree algorithm: Outline

1. Place the “best” feature (or, attribute) of the dataset at the root of the tree.
2. Split the training set into subsets. Subsets should be made in such a way that each subset
    contains data with the same value for a feature.
3. Repeat Step 1 and Step 2 on each subset until we find leaf nodes in all the branches of the tree.

8.9.2 Some well-known decision tree algorithms

1. ID3 (Iterative Dichotomiser 3) developed by Ross Quinlan
2. C4.5 developed by Ross Quinlan
3. C5.0 developed by Ross Quinlan
4. CART (Classification And Regression Trees)
5. 1R (One Rule) developed by Robert Holte in 1993.
6. RIPPER (Repeated Incremental Pruning to Produce Error Reduction) Introduced in 1995 by
    William W. Cohen.

As an example of decision tree algorithms, we discuss the details of the ID3 algorithm and illustrate
it with an example.


#### CHAPTER 8. DECISION TREES 96

### 8.10 The ID3 algorithm

Ross Quinlan, while working at University of Sydney, developed the ID3 (Iterative Dichotomiser
3)^5 algorithm and published it in 1975.

Assumptions

- The algorithm uses information gain to select the most useful attribute for classification.
- We assume that there are only two class labels, namely, “+” and “−”. The examples with class
    labels “+” are called positive examples and others negative examples.

8.10.1 The algorithm

Notations

The following notations are used in the algorithm:

```
S The set of examples
C The set of class labels
F The set of features
A An arbitrary feature (attribute)
Values(A) The set of values of the featureA
v An arbitrary value ofA
Sv The set of examples withA=v
Root The root node of a tree
```
Algorithm ID3(S,F,C)

```
1.Create a root node for the tree.
2.if(all examples inSare positive)then
```
3. return single node tree Root with label “+”
4.end if
5.if(all examples are negative)then
6. return single node tree Root with label “–”
7.end if
8.if(number of features is 0)then
9. return single node tree Root with label equal to the most common class label.
10. else
11. LetAbe the feature inFwith the highest information gain.
12. AssignAto the Root node in decision tree.
13. for all(valuesvofA)do
14. Add a new tree branch below Root corresponding tov.
15. if(Svis empty)then
16. Below this branch add a leaf node with label equal to the most common class
label in the setS.
17. else
18. Below this branch add the subtree formed by applying the same algorithm ID3
with the values ID3(Sv;C;F−{A}).
19. end if
20. end for
21. end if

(^5) dichotomy: A division into two parts or classifications especially when they are sharply distinguished or opposed


#### CHAPTER 8. DECISION TREES 97

8.10.2 Example

Problem

Use ID3 algorithm to construct a decision tree for the data in Table 8.9.

```
Day outlook temperature humidity wind playtennis
D1 sunny hot high weak no
D2 sunny hot high strong no
D3 overcast hot high weak yes
D4 rain mild high weak yes
D5 rain cool normal weak yes
D6 rain cool normal strong no
D7 overcast cool normal strong yes
D8 sunny mild high weak no
D9 sunny cool normal weak yes
D10 rain mild normal weak yes
D11 sunny mild normal strong yes
D12 overcast mild high strong yes
D13 overcast hot normal weak yes
D14 rain mild high strong no
```
```
Table 8.9: Training examples for the target concept “PlayTennis”
```
Solution

Note that, in the given data, there are four features but only two class labels (or, target variables),
namely, “yes” and “no”.

Step 1

We first create a root node for the tree (see Figure 8.7).

```
Root node
Table 8.9
```
```
Figure 8.7: Root node of the decision tree for data in Table 8.9
```
Step 2

Note that not all examples are positive (class label “yes”) and not all examples are negative (class
label “no”). Also the number of features is not zero.

Step 3

We have to decide which feature is to be placed at the root node. For this, we have to calculate the
information gains corresponding to each of the four features. The computations are shown below.

```
(i) Calculation of Entropy(S)
Entropy(S)=−pyeslog 2 (pyes)−pnolog 2 (pno)
=−( 9 ~ 14 )×log 2  9 ~ 14 −( 5 ~ 14 )×log 2  5 ~ 14 
= 0 : 9405
```

#### CHAPTER 8. DECISION TREES 98

```
(ii) Calculation of Gain(S;outlook)
The values of the attribute “outlook” are “sunny”, “ overcast” and “rain”. We have to calculate
Entropy(Sv)forv=sunny,v=overcast andv=rain.
```
```
Entropy(Ssunny)=−( 3 ~ 5 )×log 2  3 ~ 5 −( 2 ~ 5 )×log 2  2 ~ 5 
= 0 : 9710
Entropy(Sovercast)=−( 4 ~ 4 )×log 2  4 ~ 4 
= 0
Entropy(Srain)=−( 3 ~ 5 )×log 2  3 ~ 5 −( 2 ~ 5 )×log 2  2 ~ 5 
= 0 : 9710
```
```
Gain(S;outlook)=Entropy(S)−
```
```
SSsunnyS
SSS
```
```
×Entropy(Ssunny)
```
#### −

```
SSovercastS
SSS
```
```
×Entropy(Sovercast)
```
#### −

```
SSrainS
SSS
```
```
×Entropy(Srain)
```
```
= 0 : 9405 −( 5 ~ 14 )× 0 : 9710 −( 4 ~ 14 )× 0
−( 5 ~ 14 )× 0 : 9710
= 0 : 2469
```
```
(iii) Calculation of Gain(S;temperature)
The values of the attribute “temperature” are “hot”, “mild” and “cool”. We have to calculate
Entropy(Sv)forv=hot,v=mild andv=cool.
```
```
Entropy(Shot)=−( 2 ~ 4 )×log 2  2 ~ 4 −( 2 ~ 4 )×log 2  2 ~ 4 
= 1 : 0000
Entropy(Smild)=−( 4 ~ 6 )×log 2  4 ~ 6 −( 2 ~ 6 )×log 2  2 ~ 6 
= 0 : 9186
Entropy(Scool)=−( 3 ~ 4 )×log 2  3 ~ 4 −( 1 ~ 4 )×log 2  1 ~ 4 
= 0 : 8113
```
```
Gain(S;temperature)=Entropy(S)−
SShotS
SSS
```
```
×Entropy(Shot)
```
#### −

```
SSmildS
SSS
```
```
×Entropy(Smild)
```
#### −

```
SScoolS
SSS
```
```
×Entropy(Scool)
```
```
= 0 : 9405 −( 4 ~ 14 )× 1 : 0000 −( 6 ~ 14 )× 0 : 9186
−( 4 ~ 14 )× 0 : 8113
= 0 : 0293
```
```
(iv) Calculation of Gain(S;humidity)and Gain(S;wind)
The following information gains can be calculated in a similar way:
```
```
Gain(S;humidity)= 0 : 151
Gain(S;wind)= 0 : 048
```

#### CHAPTER 8. DECISION TREES 99

Step 4

We find the highest information gain whic is the maximum among Gain(S;outlook), Gain(S;temperature),
Gain(S;humidity)and Gain(S;wind). Therefore, we have:

```
highest information gain=max{ 0 : 2469 ; 0 : 0293 ; 0 : 151 ; 0 : 048 }
= 0 : 2469
```
This corresponds to the feature “outlook”. Therefore, we place “outlook” at the root node. We now
split the root node in Figure 8.7 into three branches according to the values of the feature “outlook”
as in Figure 8.8.

```
Root node
Table 8.9
outlook?
```
```
Node 1
```
```
sunny
```
```
Node 2
```
```
overcast
```
```
Node 3
```
```
rain
```
```
Figure 8.8: Decision tree for data in Table 8.9, after selecting the branching feature at root node
```
Step 5

LetS(^1 )=Soutlook=sunny. We haveUS(^1 )U= 5. The examples inS(^1 )are shown in Table 8.10.

```
Day outlook temperature humidity wind playtennis
D1 sunny hot high weak no
D2 sunny hot high strong no
D8 sunny mild high weak no
D9 sunny cool normal weak yes
D11 sunny mild normal strong yes
```
```
Table 8.10: Training examples with outlook = “sunny”
```
```
Gain(S(^1 );temp)=Entropy(S(^1 ))−
```
```
UStemp = hot(^1 ) U
TS(^1 )T
```
```
×Entropy(Stemp = hot(^1 ) )
```
#### −

```
US(temp = mild^1 ) U
TS(^1 )T
```
```
×Entropy(S
( 1 )
temp = mild)
```
#### −

```
US(temp = cool^1 ) U
TS(^1 )T
```
```
×Entropy(S(temp = cool^1 ) )
```
```
=−( 2 ~ 5 )log 2 ( 2 ~ 5 )−( 3 ~ 5 )log 2 ( 3 ~ 5 )
−( 2 ~ 5 )×−( 2 ~ 2 )log( 2 ~ 2 ))
−( 2 ~ 5 )×−( 1 ~ 2 )log( 1 ~ 2 )−( 1 ~ 2 )log 2 ( 1 ~ 2 )
−( 1 ~ 5 )×−( 1 ~ 1 )log( 1 ~ 1 )
= 0 : 5709
```

#### CHAPTER 8. DECISION TREES 100

```
Gain(S(^1 );hum)=Entropy(S(^1 ))−
```
```
UShum = high(^1 ) U
TS(^1 )T
```
```
×Entropy(Shum = high(^1 ) )
```
#### −

```
US(hum = normal^1 ) U
TS(^1 )T
```
```
×Entropy(Shum = normal(^1 ) )
```
```
=−( 2 ~ 5 )log 2 ( 2 ~ 5 )−( 3 ~ 5 )log 2 ( 3 ~ 5 )
−( 3 ~ 5 )×−( 3 ~ 3 )log( 3 ~ 3 ))
−( 2 ~ 5 )×−( 2 ~ 2 )log( 2 ~ 2 )
= 0 : 9709
```
```
Gain(S(^1 );wind)=Entropy(S(^1 ))−
```
```
USwind = weak(^1 ) U
TS(^1 )T
```
```
×Entropy(Swind = weak(^1 ) )
```
#### −

```
US(wind = strong^1 ) U
TS(^1 )T
```
```
×Entropy(Swind = strong(^1 ) )
```
```
=−( 2 ~ 5 )log 2 ( 2 ~ 5 )−( 3 ~ 5 )log 2 ( 3 ~ 5 )
−( 3 ~ 5 )×−( 2 ~ 3 )log( 2 ~ 3 )−( 1 ~ 3 )log 2 ( 1 ~ 3 ))
−( 2 ~ 5 )×−( 1 ~ 2 )log( 1 ~ 2 )−( 1 ~ 2 )log( 1 ~ 2 )
= 0 : 0110
```
The maximum of Gain(S(^1 );temp), Gain(S(^1 );hum)and Gain(S(^1 );wind)is Gain(S(^1 );hum).
Hence we place “humidity” at Node 1 and split this node into two branches according to the values
of the feature “humidity” to get the tree in Figure 8.9.

```
Root node
Table 8.9
outlook?
```
```
Node 1:
humidity?
```
```
Node 4
```
```
high
```
```
Node 5
```
```
normal
```
```
sunny
```
```
Node 2
```
```
overcast
```
```
Node 3
```
```
rain
```
```
Figure 8.9: Decision tree for data in Table 8.9, after selecting the branching feature at Node 1
```
Step 6

It can be seen that all the examples in the the data set corresponding to Node 4 in Figure 8.9 have
the same class label “no” and all the examples corresponding to Node 5 have the same class label
“yes”. So we represent Node 4 as a leaf node with value “no” and Node 5 as a leaf node with value
“yes”. Similarly, all the examples corresponding to Node 2 have the same class label “yes”. So
we convert Node 2 as a leaf node with value “ yes. Finally, letS(^2 )=Soutlook = rain. The highest
information gain for this data set is Gain(S(^2 );humidity). The branches resulting from splitting this
node corresponding to the values “high” and “normal” of “humidity” lead to leaf nodes with class
labels “no” and ”yes”. With these changes, we get the tree in Figure 8.10.


#### CHAPTER 8. DECISION TREES 101

```
Root node
Table 8.9
outlook?
```
```
Node 1:
humidity?
```
```
no
```
```
high
```
```
yes
```
```
normal
```
```
sunny
```
```
yes
```
```
overcast
```
```
Node 3:
humidity?
```
```
no
```
```
high
```
```
yes
```
```
normal
```
```
rain
```
```
Figure 8.10: Decision tree for data in Table 8.9
```
### 8.11 Regression trees

Aregression problemis the problem of determining a relation between one or more independent
variables and an output variable which is a real continuous variable and then using the relation
to predict the values of the dependent variables. Regression problems are in general related to
prediction of numerical values of variables. Trees can also be used to make such predictions. A tree
used for making predictions of numerical variables is called aprediction treeor aregression tree.

8.11.1 Example

Using the data in Table 8.11, construct a tree to predict the values ofy.

```
x 1 1 3 4 6 10 15 2 7 16 0
x 2 12 23 21 10 27 23 35 12 27 17
y 10.1 15.3 11.5 13.9 17.8 23.1 12.7 43.0 17.6 14.9
```
```
Table 8.11: Data for regression tree
```
Solution

We shall construct araw decision tree(a tree constructed without using any standard algorithm) to
predict the value ofycorresponding to any untabulated values ofx 1 andx 2.

Step 1. We arbitrarily split the values ofx 1 into two sets: One set defined byx 1 < 6 and the other
set defined byx 1 ≥ 6. This splits the data into two parts. This yields the tree in Figure??.

```
x 1 1 3 4 2 0
x 2 12 23 21 35 17
y 10.1 15.3 11.5 12.7 14.9
```
```
Table 8.12: Data for regression tree
```
Step 2. In Figure 8.12, consider the node specified by Table 8.12. We arbitrarily split the values
ofx 2 into two sets: one specified byx 2 < 21 and one specified byx 2 ≥ 21. Similarly, the
node specified by Table 8.13, we split the values ofx 2 into sets: one specified byx 2 < 23


#### CHAPTER 8. DECISION TREES 102

```
x 1 6 10 15 7 16
x 2 10 27 23 12 27
y 13.9 17.8 23.1 43.0 17.6
```
```
Table 8.13: Data for regression tree
```
```
Tab 8.11
```
```
Tab 8.12 Tab 8.13
```
```
x 1 < 6 x 1 ≥ 6
```
```
Figure 8.11: Part of a regression tree for Table 8.11
```
```
and one specified byx 2 ≥ 23. The split data are given in Table 8.14(a) - (d). This gives us
the tree in Figure 8.12.
```
```
Tab 8.11
```
```
Tabe 8.12 Tab 8.13
```
```
x 1 < 6 x 1 ≥ 6
```
```
Tab 8.14(a) Tab 8.14(b)
```
```
x 2 < 21 x 2 ≥ 21
```
```
Tab 8.14(c) Tab 8.14(d)
```
```
x 2 < 23 x 2 ≥ 23
```
```
Figure 8.12: Part of regression tree for Table 8.11
```
Step 3. We next make the nodes specified by Table 8.14(a),:::, Tab 8.14(d) into leaf nodes. In
each of these leaf nodes, we write the average of the values in the corresponding table (this
is a standard procedure). For, example, at Table 8.14(a), we write^12 ( 10 : 1 + 14 : 9 )= 12 : 5.
Then we get Figure 8.13.

```
x 1 1 0
x 2 12 17
y 10.1 14.9
```
```
x 1 3 4 2
x 2 23 21 35
y 15.3 11.5 12.7
```
```
(a) (b)
```
```
x 1 6 7
x 2 10 12
y 13.9 43.0
```
```
x 1 10 15 16
x 2 27 23 27
y 17.8 23.1 17.6
```
```
(c) (d)
```
```
Table 8.14: Data for regression tree
```

#### CHAPTER 8. DECISION TREES 103

```
x 1 < 6 x 1 ≥ 6
```
#### 12.5 13.17

```
x 2 < 21 x 2 ≥ 21
```
#### 28.45 19.5

```
x 2 < 23 x 2 ≥ 23
```
```
Figure 8.13: A regression tree for Table 8.11
```
Step 4. Figure 8.13 is the final raw regression tree for predicting the values ofybased on the data
in Table 8.11.

8.11.2 An algorithm for constructing regression trees

Starting with a learning sample, three elements are necessary to determine a regression tree:

1. A way to select a split at every intermediate node
2. A rule for determining when a node is terminal
3. A rule for assigning a value for the output variable to every terminal node

Notations

```
x 1 ;x 2 ;:::;xn : The input variables
N : Number of samples in the data set
y 1 ;y 2 ;:::;yN : The values of the output variables
T : A tree
c : A leaf ofT
nc : Number of data elements in the leafc
C : The set of indices of data elements which
are in the leafc
mc : The mean of the values ofywhich are in
the leafc
ST : Sum of squares of errors inT
```
We have

```
mc=
```
#### 1

```
nc
```
#### Q

```
i∈C
```
```
yi
```
```
ST= Q
c∈leaves(T)
```
#### Q

```
i∈C
```
```
(yi−mc)^2
```
Algorithm

Step 1. Start with a single node containing all data points. CalculatemcandST.

Step 1. If all the points in the node have the same value for all the independent variables, stop.

Step 1. Otherwise, search over all binary splits of all variables for the one which will reduceSTas
much as possible.


#### CHAPTER 8. DECISION TREES 104

```
(a) If the largest decrease inSTwould be less than some threshold, or one of the
resulting nodes would contain less thanqpoints, stop and ifcis a node where we
have stopped, then assign the valuemcto the node.
(b) Otherwise, take that split, creating two new nodes.
```
Step 1. In each new node, go back to Step 1.

Remarks

1. We have seen entropy and information defined for discrete variables. We can define them for
    continuous variables also. But in the case of regression trees, it is more common to use the
    sum of squares. The above algorithm is based on sum of squares of errors.
2. The CART algorithm mentioned below searches every distinct value of every predictor vari-
    able to find the predictor variable and split value which will reduceSTas much as possible.
3. In the above algorithm, we have given the simplest criteria for stopping growing of trees.
    More sophisticated criteria which produce much less error have been developed.

8.11.3 Example

Consider the data given in Table 8.11.

1. Computation ofSTfor the entire data set. Initially, there is only one node. So, we have:

```
mc=
```
#### 1

```
nc
```
#### Q

```
c∈C
```
```
yi
```
#### =

#### 1

#### 10

#### ( 10 : 1 + 15 : 3 + ⋯ + 14 : 9 )

#### = 17 : 99

#### ST= Q

```
c∈leaves(T)
```
#### Q

```
i∈C
```
```
(yi−mc)^2
```
#### =( 10 : 1 − 17 : 99 )^2 +( 15 : 3 − 17 : 99 )^2 + ⋯ +( 14 : 9 − 17 : 99 )^2

#### = 817 : 669

2. As suggested in the remarks above, we have to search every distinct value ofx 1 andx 2 to find
    the predictor variable and split value which will reduceSTas much as possible.
3. Let us consider the value 6 ofx 1. This splits the data set into two partsc 1 andc 2. Letc 1 be
    the part defined byx 1 < 6 andc 2 the part defined byx 1 ≥ 6 .S 1 is given in Table 8.12 andS 2
    by Table 8.13.Now
       leaves(T)={c 1 ;c 2 }:
    LetT 1 be the tree corresponding to this partition. Then

```
ST 1 = Q
c∈leaves(T 1 )
```
#### Q

```
i∈C
```
```
(yi−mc)^2
```
#### = Q

```
i∈C 1
```
```
(yi−mc 1 )^2 +Q
i∈C 2
```
```
(yi−mc 2 )^2
```
```
mc 1 =
```
#### 1

```
nc 1
```
#### Q

```
i∈C 1
```
```
yi
```
#### =

#### 1

#### 5

#### ( 10 : 1 + 15 : 3 + 11 : 5 + 12 : 7 + 14 : 9 )

#### = 12 : 9


#### CHAPTER 8. DECISION TREES 105

```
mc 2 =
```
#### 1

```
nc 2
```
#### Q

```
i∈C 2
```
```
yi
```
#### =

#### 1

#### 5

#### ( 13 : 9 + 17 : 8 + 23 : 1 + 43 : 0 + 17 : 6 )

#### = 23 : 08

#### ST 1 =[( 10 : 1 − 12 : 9 )^2 + ⋯ +( 14 : 9 − 12 : 9 )^2 ]+

#### [( 13 : 9 − 23 : 08 )^2 + ⋯ +( 17 : 6 − 23 : 08 )^2 ]

#### = 558 : 588

```
The reduction in sum of squares of errors is
```
```
ST−ST 1 = 817 : 669 − 558 : 588 = 259 : 081 :
```
4. In this way, we have compute the reduction in the sum of squares of errors corresponding to
    all other values ofx 1 and each of the values ofx 2 and choose the one for which the reduction
    is maximum.
5. The process has be continued. (Software package may be required to complete the problem.)

### 8.12 CART algorithm

We have seen how decision trees can be used to create a model that predicts the value of a target (or
dependent variable) based on the values of several input or independent variables.
The CART, orClassification And Regression Treesmethodology, was introduced in 1984 by Leo
Breiman, Jerome Friedman, Richard Olshen and Charles Stone as an umbrella term to refer to the
following types of decision trees:

- Classification treeswhere the target variable is categorical and the tree is used to identify the
    “class” within which a target variable would likely fall into.
- Regression treeswhere the target variable is continuous and tree is used to predict it’s value.

The main elements of CART are:

- Rules for splitting data at a node based on the value of one variable
- Stopping rules for deciding when a branch is terminal and can be split no more
- A prediction for the target variable in each terminal node

### 8.13 Other decision tree algorithms

8.13.1 The C4.5 algorithm

The C4.5 algorithm is an algorithm developed by Ross Quinlan as an improvement of the ID3
algorithm. The following are some of the improvements incorporated in C4.5.

- Handling both continuous and discrete attributes
- Handling training data with missing attribute values
- Handling attributes with differing costs
- Pruning trees after creation


#### CHAPTER 8. DECISION TREES 106

8.13.2 The C5.0 algorithm

The C5.0 algorithm represents a further improvement on the C4.5 algorithm. This was also devel-
oped by Ross Quinlan.

- Speed - C5.0 is significantly faster than C4.5.
- Memory usage - C5.0 is more memory efficient than C4.5.
- C5.0 gets similar results to C4.5 with considerably smaller decision trees.

The C5.0 algorithm is one of the most well-known implementations of the the decision tree
algorithm. The source code for a single-threaded version of the algorithm is publicly available,
and it has been incorporated into programs such as R. The C5.0 algorithm has become the industry
standard to produce decision trees.

### 8.14 Issues in decision tree learning

In thie next feww sections, we discuss some of the practical issues in learning decision trees.

### 8.15 Avoiding overfitting of data

When we construct a decision tree, the various branches are grown (that is, sub-branches are con-
structed) just deeply enough to perfectly classify the training examples. This leads to difficulties
when there is noise in the data or when the number of training examples are too small. In these
cases the algorithm can produce trees that overfit the training examples.

Definition

We say that a hypothesisoverfitsthe training examples if some other hypothesis that fits the train-
ing examples less well actually performs better over the entire distribution of instances, including
instances beyond the training set.

Impact of overfitting

Figure 8.14 illustrates the impact of overfitting in a typical decision tree learning. From the figure,
we can see that the accuracy of the tree over training examples increases monotonically whereas the
accuracy measured over independent test samples first increases then decreases.

8.15.1 Approaches to avoiding overfitting

The main approach to avoid overfitting ispruning. Pruning is a technique that reduces the size
of decision trees by removing sections of the tree that provide little power to classify instances.
Pruning reduces the complexity of the final classifier, and hence improves predictive accuracy by
the reduction of overfitting.

- We may apply pruning earlier, that is, before it reaches the point where it perfectly classifies
    the training data.
- We may allow the tree to overfit the data, and then post-prune the true.

Now there is the problem of what criterion is to be used to determine the correct final tree
size. One commonly used criterion is to use a separate set of examples, distinct from the training
examples, to evaluate the utility of post-pruning nodes from the tree.


#### CHAPTER 8. DECISION TREES 107

```
Figure 8.14: Impact of overfitting in decision tree learning
```
```
Case Temperature Headache Nausea Decision (Flue)
1 high? no yes
2 very high yes no yes
3? no no no
4 high yes yes yes
5 high? yes no
6 normal yes no no
7 normal no yes no
8? yes? yes
```
```
Table 8.15: A dataset with missing attribute values
```
8.15.2 Reduced error pruning

Inreduced-error pruning, we consider each of the decision tress to be a candidate for pruning. Prun-
ing a decision node consists of removing the subtree rooted at that node, making it a leaf node, and
assigning it the most common classification of the training examples affiliated to that node. Nodes
are removed only if the resulting pruned tree performs no worse than the original over validation set.
Nodes are pruned iteratively, always choosing the node whose removal most increases the accuracy
over the validation set. Pruning of nodes is continued until further pruning decreases the accuracy
over the validation set.

### 8.16 Problem of missing attributes

Table 8.15 shows a dataset with missing attribute values. the missing values are indicated by “?”s.
The following are some of the methods used to handle the problem of missing attributes.

- Deleting cases with missing attribute values
- Replacing a missing attribute value by the most common value of that attribute


#### CHAPTER 8. DECISION TREES 108

- Assigning all possible values to the missing attribute value
- Replacing a missing attribute value by the mean for numerical attributes
- Assigning to a missing attribute value the corresponding value taken from the closesttcases,
    or replacing a missing attribute value by a new value

### 8.17 Sample questions

(a) Short answer questions

1. Explain the concept of a decision tree with an example.
2. What are the different types of decision trees?
3. Define the entropy of a dataset.
4. Write a formula to compute the entropy of a two-class dataset.
5. Define information gain and Gini index.
6. Give the names of five different decision-tree algorithms.
7. Can decision tree be used for regression? If yes, explain how. If no, explain why.
8. What is the difference between classification and regression trees?

(b) Long answer questions

1. Explain classification tree using an example.
2. Consider the following set of training examples:

```
Instance Classification a 1 a 2
1 + T T
2 + T T
3 − T F
4 + F F
5 − F T
6 − F T
```
```
(a) What is the entropy of this collection of training examples with respect to the target
function “classification”?
(b) What is the information gain ofa 2 relative to these training examples?
```
3. Explain the ID3 algorithm for learning decision trees.
4. Explain CART algorithm.
5. What are issues in decision tree learning? How are they overcome?
6. Describe an algorithm to construct regression trees.
7. What do you mean by information gain and entropy? How is it used to build the decision
    trees? Illustrate using an example.
8. Use ID3 algorithm to construct a decision tree for the data in the following table.


#### CHAPTER 8. DECISION TREES 109

```
Instance no. Class label x 1 x 2
1 1 T T
2 1 T T
3 0 T F
4 1 F F
5 0 F T
6 0 F T
```
9. Use ID3 algorithm to construct a decision tree for the data in the following table.

```
Gender Car ownership Travel cost Income level Class
(mode of transportation)
Male 0 Cheap Low Bus
Male 1 Cheap Medium Bus
Female 1 Cheap Medium Train
Female 0 Cheap Low Bus
Male 1 Cheap Medium Bus
Male 0 Standard Medium Train
Female 1 Standard Medium Train
Female 1 Expensive High Car
Male 2 Expensive Medium Car
Female 2 Expensive High Car
```
10. Use ID3 algorithm to construct a decision tree for the data in the following table.

```
Age Competition Type Class (profit)
Old Yes Software Down
Old No Software Down
Old No Hardware Down
Mid Yes Software Down
Mid Yes Hardware Down
Mid No Hardware Up
Mid No Software Up
New Yes Software Up
New No Hardware Up
New No Software Up
```
11. Construct a decision tree for the following data.


#### CHAPTER 8. DECISION TREES 110

```
Class label (risk) Collateral Income Debt Credit history
high none low high bad
high none middle high unknown
moderate none middle low unknown
high none low low unknown
low none upper low unknown
low adequate upper low unknown
high none low low bad
moderate adequate upper low bad
low none upper low good
low adequate upper high good
high none low high good
moderate none middle high good
low none upper high good
high none middle high bad
```

Chapter 9

## Neural networks

### 9.1 Introduction

AnArtificial Neural Network(ANN) models the relationship between a set of input signals and an
output signal using a model derived from our understanding of how a biological brain responds to
stimuli from sensory inputs. Just as a brain uses a network of interconnected cells called neurons
to create a massive parallel processor, ANN uses a network of artificial neurons or nodes to solve
learning problems.

### 9.2 Biological motivation

Let us examine how a biological neuron functions. Figure 9.2 gives a schematic representation of
the functioning of a biological neuron.
In the cell, the incoming signals are received by the cell’sdendritesthrough a biochemical pro-
cess. The process allows the impulse to be weighted according to its relative importance or fre-
quency. As the cell body begins accumulating the incoming signals, a threshold is reached at which
the cell fires and the output signal is transmitted via an electrochemical process down theaxon. At
the axon’s terminals, the electric signal is again processed as a chemical signal to be passed to the
neighboring neurons across a tiny gap known as asynapse.^1
Biological learning systems are built of very complex webs of interconnected neurons. The hu-
man brain has an interconnected network of approximately 1011 neurons, each connected, on an
average, to 104 other neurons. Even though the neuron switching speeds are much slower than than

(^1) Neuron. (2018, February 15). In Wikipedia, The Free Encyclopedia. Retrieved 01:44, February 23, 2018.
Figure 9.1: Anatomy of a neuron

#### 111


#### CHAPTER 9. NEURAL NETWORKS 112

```
Figure 9.2: Flow of signals in a biological neuron
```
computer switching speeds, we are able to take complex decisions relatively quickly. Because of
this, it is believed that the information processing capabilities of biological neural systems is a con-
sequence of the ability of such systems to carry out a huge number of parallel processes distributed
over many neurons. The developments in ANN systems are motivated by the desire to implement
this kind of highly parallel computation using distributed representations.

### 9.3 Artificial neurons

Definition

Anartificial neuronis a mathematical function conceived as a model of biological neurons. Artificial
neurons are elementary units in an artificial neural network. The artificial neuron receives one or
more inputs (representing excitatory postsynaptic potentials and inhibitory postsynaptic potentials
at neural dendrites) and sums them to produce an output. Each input is separately weighted, and the
sum is passed through a function known as anactivation functionortransfer function.

Schematic representation of an artificial neuron

The diagram shown in Figure??gives a schematic representation of a model of an artificial neuron.
The notations in the diagram have the following meanings:

```
∑ f
```
#### :::

```
x 0 = 1
```
```
x 1 w 0
w 1
x 2 w 2
```
```
xn
```
```
wn
```
```
n
Q
i= 0
```
```
wixi f⎛
⎝
```
```
n
Q
i= 0
```
```
wixi
```
#### ⎞

#### ⎠

```
Output (y)
```
```
y=f
```
#### ⎛

#### ⎝

```
n
Q
i= 0
```
```
wixi
```
#### ⎞

#### ⎠

```
Figure 9.3: Schematic representation of an artificial neuron
```
```
x 1 ;x 2 ;:::xn∶ input signals
w 1 ;w 2 ;:::wn∶ weights associated with input signals
```

#### CHAPTER 9. NEURAL NETWORKS 113

```
x 0 ∶ input signal taking the constant value 1
w 0 ∶ weight associated withx 0 (called bias)
Q∶ indicates summation of input signals
f∶ function which produces the output
y∶ output signal
```
The functionfcan be expressed in the following form:

```
y=f
```
```
n
Q
i= 0
```
```
wixi (9.1)
```
Remarks

The small circles in the schematic representation of the artificial neuron shown in Figure 9.3 are
called thenodesof the neuron. The circles on the left side which receives the values ofx 0 ;x 1 ;:::;xn
are called theinput nodesand the circle on the right side which outputs the value ofyis called
output node. The squares represent the processes that are taking place before the result is outputted.
They need not be explicitly shown in the schematic representation. Figure 9.4 shows a simplified
representation of an artificial neuron.

#### :::

```
x 0 = 1
```
```
x 1 w 0
w 1
x 2 w 2
```
```
xn
```
```
wn
```
```
Output (y)
```
```
y=f
```
#### ⎛

#### ⎝

```
n
Q
i= 0
```
```
wixi
```
#### ⎞

#### ⎠

```
Figure 9.4: Simplified representation of an artificial neuron
```
### 9.4 Activation function

9.4.1 Definition

In an artificial neural network, the function which takes the incoming signals as input and produces
the output signal is known as theactivation function.

Remark

Eq.(9.1) represents the activation function of the ANN model shown in Figure??.

9.4.2 Some simple activation functions

The following are some of the simple activation functions.


#### CHAPTER 9. NEURAL NETWORKS 114

1. Threshold activation function

Thethreshold activation functionis defined by

```
f(x)=
```
#### ⎧⎪

#### ⎪

#### ⎨

#### ⎪⎪

#### ⎩

```
1 ifx> 0
− 1 ifx≤ 0
```
The graph of this function is shown in Figure 9.5.

```
x
```
#### 1

#### − 1

#### 0

```
Figure 9.5: Threshold activation function
```
2. Unit step functions

Sometimes, the threshold activation function is also defined as a unit step function in which case it
is called aunit-step activation function. This is defined as follows:

```
f(x)=
```
#### ⎧⎪

#### ⎪

#### ⎨

#### ⎪⎪

#### ⎩

```
1 ifx≥ 0
0 ifx< 0
```
The graph of this function is shown in Figure 9.6.

```
x
```
#### 1

#### − 1

#### 0

```
Figure 9.6: Unit step activation function
```
3. Sigmoid activation function (logistic function)

One of the must commonly used activation functions is the sigmoid activation function. It is defined
as follows:

```
f(x)=
```
#### 1

1 +e−x
The graph of the function is shown in Figure 9.7.

```
x
```
```
f(x)
1
```
#### 0

```
Figure 9.7: The sigmoid activation function
```

#### CHAPTER 9. NEURAL NETWORKS 115

4. Linear activation function

The linear activation function is defined by

```
F(x)=mx+c:
```
This defines a straight line in thexy-plane.

```
x
```
#### 1

#### − 1

#### 0

```
Figure 9.8: Linear activation function
```
5. Piecewise (or, saturated) linear activation function

This is defined by

```
f(x)=
```
#### ⎧⎪

#### ⎪⎪⎪

#### ⎪

#### ⎨

#### ⎪⎪⎪

#### ⎪⎪

#### ⎩

```
0 ifx<xmin
mx+c ifxmin≤x≤xmax
0 ifx>xmax
```
```
x
```
#### 1

#### − 1

#### 0

```
Figure 9.9: Piecewise linear activation function
```
6. Gaussian activation function

This is defined by

```
f(x)=
```
#### 1

#### 

#### √

#### 2 

```
e−
```
```
(x−)^2
2 ^2 :
```
```
x
```
#### 1

#### − 1

#### 0

```
Figure 9.10: Gaussian activation function
```

#### CHAPTER 9. NEURAL NETWORKS 116

7. Hyperbolic tangential activation function

This is defined by

```
f(x)=
```
```
ex−e−x
ex+e−x
```
#### :

```
x
```
#### 1

#### − 1

#### 0

```
Figure 9.11: Hyperbolic tangent activation function
```
### 9.5 Perceptron

The perceptron is a special type of artificial neuron in which thee activation function has a special
form.

9.5.1 Definition

A perceptron is an artificial neuron in which the activation function is the threshold function.
Consider an artificial neuron havingx 1 ,x 2 ,⋯,xnas the input signals andw 1 ,w 2 ,⋯,wnas the
associated weights. Letw 0 be some constant. The neuron is called a perceptron if the output of the
neuron is given by the following function:

```
o(x 1 ;x 2 ;:::;xn)=
```
#### ⎧⎪

#### ⎪

#### ⎨

#### ⎪⎪

#### ⎩

```
1 ifw 0 +w 1 x 1 + ⋯ +wnxn> 0
− 1 ifw 0 +w 1 x 1 + ⋯ +wnxn≤ 0
```
Figure 9.12 shows the schematic representation of a perceptron.

#### ∑^

#### :::

```
x 0 = 1
```
```
x 1 w 0
w 1
```
```
x 2 w 2
```
```
xn
```
```
wn
```
```
n
Q
i= 0
```
```
wixi
y=
```
#### ⎧⎪

#### ⎪⎪⎪

#### ⎨

#### ⎪⎪⎪

#### ⎪⎩

```
1 if
```
```
n
Q
i= 0
```
```
wixi> 0
```
```
− 1 otherwise
```
```
Output (y)
```
```
Figure 9.12: Schematic representation of a perceptrn
```

#### CHAPTER 9. NEURAL NETWORKS 117

Remarks

1. The quantity−w 0 can be looked upon as a “threshold” that should be crossed by the weighted
    sumw 1 x 1 + ⋯ +wnxnin order for the neuron to output a “ 1 ”.

9.5.2 Representations of boolean functions by perceptrons

In this section we examine whether simple boolean functions likex 1 ANDx 2 can be represented by
perceptrons. To be consistent with the conventions in the definition of a perceptron we assume that
the values− 1 and 1 represent the boolean constants “false” and “true” respectively.

9.5.3 Representation ofx 1 ANDx 2

Letx 1 andx 2 be two boolean variables. Then the boolean functionx 1 ANDx 2 is represented by
Table 9.1. It can be easily verified that the perceptron shown in Figure 9.13 represents the function

```
x 1 x 2 x 1 ANDx 2
− 1 − 1 − 1
− 1 1 − 1
1 − 1 − 1
1 1 1
```
```
Table 9.1: The boolean functionx 1 ANDx 2
```
x 1 ANDx 2.

#### ∑^

```
x 0 = 1
```
```
x 1
```
```
x 2
```
```
w 0 =− 0 : 8
```
```
w 1 = 0 : 5
```
```
w 3 = 0 : 5
```
```
3
Q
i= 0
```
```
wixi
y=
```
#### ⎧⎪

#### ⎪⎪⎪

#### ⎪

#### ⎨

#### ⎪⎪⎪

#### ⎪⎪

#### ⎩

```
1 if
```
```
3
Q
i= 0
```
```
wixi> 0
```
```
− 1 otherwise
```
```
Output (y)
```
```
Figure 9.13: Representation ofx 1 ANDx 2 by a perceptron
```
```
In the perceptron shown in Figure 9.13, the output is given by
```
```
y=
```
#### ⎧⎪

#### ⎪⎪⎪

#### ⎪

#### ⎨

#### ⎪⎪⎪

#### ⎪⎪

#### ⎩

```
1 if
```
```
3
Q
i= 0
```
```
wixi> 0
```
```
− 1 otherwise
```
#### =

#### ⎧⎪

#### ⎪

#### ⎨

#### ⎪⎪

#### ⎩

```
1 if− 0 : 8 + 0 : 5 x 1 + 0 : 5 x 2 > 0
− 1 otherwise
```
Representations of OR, NAND and NOR

The functionsx 1 ORx 2 ,x 1 NANDx 2 andx 1 NORx 2 can also be represented by perceptrons. Table
9.2 shows the values to be assigned to the weightsw 0 ;w 1 ;w 2 for getting these boolean functions.


#### CHAPTER 9. NEURAL NETWORKS 118

```
Boolean function w 0 w 1 w 2
x 1 ANDx 2 − 0 : 8 0 : 5 0 : 5
x 1 ORx 2 − 0 : 3 0 : 5 0 : 5
x 1 NANDx 2 0 : 8 − 0 : 5 − 0 : 5
x 1 NORx 2 0 : 3 − 0 : 5 − 0 : 5
```
```
Table 9.2: Representations of boolean functions by perceptrons
```
Remarks

Not all boolean functions can be represented by perceptrons. For example, the boolean function
x 1 XORx 2 cannot be represented by a perceptron. This means that we cannot assign values to
w 0 ;w 1 ;w 2 such that the expressionw 0 +w 1 x 1 +w 2 x 2 takes the values ofx 1 XORx 2 , and that this
is the case can be easily verified also.

9.5.4 Learning a perceptron

By “learning a perceptron” we mean the process of assigning values to the weights and the thresh-
old such that the perceptron produces correct output for each of the given training examples. The
following are two algorithms to solve this learning problem:

9.5.5 Perceptron learning algorithm

Definitions

In the algorithm, we use the following notations:

```
n : Number of input variables
y=f(z) : Output from the perceptron for an input
vectorz
D={(x 1 ;d 1 );:::;(xs;ds)} : Training set ofssamples
xj=(xj 0 ;xj 1 ;:::;xjn) : Then-dimensional input vector
dj : Desired output value of the perceptron for
the inputxj
xji : Value of thei-th feature of thej-th training
input vector
xj 0 : 1
wi : Weight of thei-th input variable
wi(t) : Weightiat thet-th iteration
```
Algorithm

```
Step 1. Initialize the weights and the threshold. Weights may be initialized to 0 or to a small
random value.
```
```
Step 2. For each examplejin the training setD, perform the following steps over the inputxj
and desired outputdj:
```
```
a) Calculate the actual output:
```
```
yj(t)=f[w 0 (t)xj 0 +w 1 (t)xj 1 +w 2 (t)xj 2 + ⋯ +wn(t)xjn]
```

#### CHAPTER 9. NEURAL NETWORKS 119

```
b) Update the weights:
```
```
wi(t+ 1 )=wi(t)+(dj−yj(t))xji
```
```
for all features 0 ≤i≤n.
```
```
Step 3. Step 2 is repeated until the iteration error^1 s∑sj= 1 Sdj−yj(t)Sis less than a user-specified
error threshold , or a predetermined number of iterations have been completed, wheres
is again the size of the sample set.
```
Remarks

The above algorithm can be applied only if the training examples arelinearly separable.

### 9.6 Artificial neural networks

Anartificial neural network(ANN) is a computing system inspired by the biological neural networks
that constitute animal brains. An ANN is based on a collection of connected units called artificial
neurons. Each connection between artificial neurons can transmit a signal from one to another. The
artificial neuron that receives the signal can process it and then signal artificial neurons connected to
it.
each connection between artificial neurons has a weight attached to it that get adjusted as learning
proceeds. Artificial neurons may have a threshold such that only if the aggregate signal crosses that
threshold the signal is sent. Artificial neurons are organized in layers. Different layers may perform
different kinds of transformations on their inputs. Signals travel from the input layer to the output
layer, possibly after traversing the layers multiple times.

### 9.7 Characteristics of an ANN

An ANN can be defined and implemented in several different ways. The way the following charac-
teristics are defined determines a particular variant of an ANN.

- The activation function
    This function defines how a neuron’s combined input signals are transformed into a single
    output signal to be broadcasted further in the network.
- The network topology (or architecture)
    This describes the number of neurons in the model as well as the number of layers and manner
    in which they are connected.
- The training algorithm
    This algorithm specifies how connection weights are set in order to inhibit or excite neurons
    in proportion to the input signal.

9.7.1 Activation functions

The activation function is the mechanism by which the artificial neuron processes incoming informa-
tion and passes it throughout the network. Just as the artificial neuron is modeled after the biological
version, so is the activation function modeled after nature’s design.
Letx 1 ,x 2 ,:::,xnbe the input signals,w 1 ,w 2 ,:::,wnbe the associated weights and−w 0 the
threshold. Let
x=w 0 +w 1 x 1 + ⋯ +wnxn:

The activation function is some function ofx. Some of the simplest and commonly used activations
are given in Section 9.4.


#### CHAPTER 9. NEURAL NETWORKS 120

9.7.2 Network topology

By “network topology” we mean the patterns and structures in the collection of interconnected
nodes. The topology determines the complexity of tasks that can be learned by the network. Gener-
ally, larger and more complex networks are capable of identifying more subtle patterns and complex
decision boundaries. However, the power of a network is not only a function of the network size,
but also the way units are arranged.
Different forms of forms of network architecture can be differentiated by the following charac-
teristics:

- The number of layers
- Whether information in the network is allowed to travel backward
- The number of nodes within each layer of the network
1. The number of layers

In an ANN, theinput nodesare those nodes which receive unprocessed signals directly from the
input data. Theoutput nodes(there may be more than one) are those nodes which generate the final
predicted values. Ahidden nodeis a node that processes the signals from the input nodes (or other
such nodes) prior to reaching the output nodes.
The nodes are arranged inlayers. The set of nodes which receive the unprocessed signals from
the input data constitute thefirst layerof nodes. The set of hidden nodes which receive the outputs
from the nodes in the first layer of nodes constitute thesecond layerof nodes. In a similar way we
can define the third, fourth, etc. layers. Figure 9.14 shows an ANN with only one layer of nodes.
Figure 9.15 shows an ANN with two layers.

#### :::

```
x 0
```
```
x 1 w 0
w 1
x 2 w 2
```
```
xn
```
```
wn
```
```
Output (y)
```
```
Input layer Output layer
```
```
Figure 9.14: An ANN with only one layer
```
2. The direction of information travel

Networks in which the input signal is fed continuously in one direction from connection to connec-
tion until it reaches the output layer are calledfeedforward networks. The network shown in Figure
9.15 is a feedforward network.
Networks which allows signals to travel in both directions using loops are calledrecurrent net-
works(or,feedback networks).
In spite of their potential, recurrent networks are still largely theoretical and are rarely used
in practice. On the other hand, feedforward networks have been extensively applied to real-world
problems. In fact, the multilayer feedforward network, sometimes called the Multilayer Perceptron
(MLP), is the de facto standard ANN topology. If someone mentions that they are fitting a neural
network, they are most likely referring to a MLP.


#### CHAPTER 9. NEURAL NETWORKS 121

```
Input
layer
```
```
Hidden
layer
```
```
Output
layer
```
```
x 0
```
```
x 1
```
```
x 2
```
#### ⋯

```
xn
```
```
Output
```
```
Figure 9.15: An ANN with two layers
```
3. The number of nodes in each layer

The number of input nodes is predetermined by the number of features in the input data. Similarly,
the number of output nodes is predetermined by the number of outcomes to be modeled or the
number of class levels in the outcome. However, the number of hidden nodes is left to the user to
decide prior to training the model. Unfortunately, there is no reliable rule to determine the number
of neurons in the hidden layer. The appropriate number depends on the number of input nodes, the
amount of training data, the amount of noisy data, and the complexity of the learning task, among
many other factors.

9.7.3 The training algorithm

There are two commonly used algorithms for learning a single perceptron, namely, the perceptron
rule and the delta rule. The former is used when the training data set is linearly separable and the
latter when the training data set is not linearly separable.
The algorithm which is now commonly used to train an ANN is known simply asbackpropaga-
tion.

9.7.4 The cost function

Definition

In a machine learning algorithm, thecost functionis a function that measures how well the algorithm
maps the target function that it is trying to guess or a function that determines how well the algorithm
performs in an optimization problem.

Remaarks

The cost function is also called theloss function, theobjective function, thescoring function, or the
error function.

Example

Letybe the the output variable. Lety 1 ;:::;ynbe the actual values ofyinnexamples and^y 1 ;:::;y^n
be the values predicted by an algorithm.


#### CHAPTER 9. NEURAL NETWORKS 122

```
Input
layer
```
```
Hidden
layer
```
```
Output
layer
```
```
x 0
```
```
x 1
```
```
x 2
```
#### ⋯

```
xn
```
```
Output 1
```
```
Output 2
```
```
(a) Network with one hidden layer and two output nodes
```
```
Input
layer
```
```
Hidden
layer 1
```
```
Hidden
layer 2
```
```
Output
layer
```
```
x 0
```
```
x 1
```
```
x 2
```
#### ⋯

```
xn
```
```
Output
```
```
(b) Network with two hidden layers
```
```
Figure 9.16: Examples of different topologies of networks
```
1. The sum of squares of the differences between the predicted and actual values ofy, denoted
    by SSE and defined below, can be taken as a cost function for the algorithm.

#### SSE=

```
n
Q
i= 1
```
```
(yi−^yi)^2 :
```
2. The mean of the sum of squares of the differences between the predicted and actual values of
    y, denoted by MSE and defined below, can be taken as a cost function for the algorithm.

#### MSE=

#### 1

```
n
```
```
n
Q
i= 1
```
```
(yi−y^i)^2 :
```
### 9.8 Backpropagation

The backpropagation algorithm was discovered in 1985-86. Here is an outline of the algorithm.


#### CHAPTER 9. NEURAL NETWORKS 123

```
Figure 9.17: A simplified model of the error surface showing the direction of gradient
```
9.8.1 Outline of the algorithm

1. Initially the weights are assigned at random.
2. Then the algorithm iterates through many cycles of two processes until a stopping criterion is
    reached. Each cycle is known as anepoch. Each epoch includes:

```
(a) Aforward phasein which the neurons are activated in sequence from the input layer to
the output layer, applying each neuron’s weights and activation function along the way.
Upon reaching the final layer, an output signal is produced.
(b) Abackward phasein which the network’s output signal resulting from the forward phase
is compared to the true target value in the training data. The difference between the
network’s output signal and the true value results in an error that is propagated backwards
in the network to modify the connection weights between neurons and reduce future
errors.
```
3. The technique used to determine how much a weight should be changed is known asgradient
    descent method. At every stage of the computation, the error is a function of the weights. If
    we plot the error against the wights, we get a higher dimensional analog of something like a
    curve or surface. At any point on this surface, the gradient suggests how steeply the error will
    be reduced or increased for a change in the weight. The algorithm will attempt to change the
    weights that result in the greatest reduction in error (see Figure 9.17).

9.8.2 Illustrative example

To illustrate the various steps in the backpropagation algorithm, we consider a small network with
two inputs, two outputs and one hidden layer as shown in Figure 9.18.^2
We assume that there are two observations:
Sample Input 1 Input 2 Output target 1 Output target 2
i 1 i 2 T 1 T 2
1 0.05 0.10 0.01 0.99
2 0.25 0.18 0.23 0.79

We are required to estimate the optimal values of the weightsw 1 ;:::;w 8 ;b 1 ;b 2. Hereb 1 andb 2 are
the biases. For simplicity, we have assigned the same biases to both nodes in the same layer.

Step 1. We initialise the connection weights to small random values. These initial weights are
shown in Figure 9.19.

(^2) Thanks to https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-
example/for this example.


#### CHAPTER 9. NEURAL NETWORKS 124

```
Input 1
```
```
Input 2
```
#### 1

```
Output 1
```
```
Output 2
```
#### 1

```
w 1
h 1
```
```
w 2
```
```
w 5
```
```
w 6
```
```
o 1
w 3
```
```
w 4 h^2
```
```
w 7
```
```
o 2
w 8
b 1
```
```
b 2
```
```
b 3
```
```
b 4
```
```
Figure 9.18: ANN for illustrating backpropagation algorithm
```
```
i 1 =: 05
```
```
i 2 =: 10
```
#### 1

#### T 1 =: 01

#### T 2 =: 99

#### 1

```
w 1 =: 15
h 1
```
```
w 2 =: 20
```
```
w 5 =: 40
o 1
```
```
w 6 =: 45
```
```
w 3 =: 25
```
```
h 2
w 4 =: 30
```
```
w 7 =: 50
```
```
w 8 =: 55
```
```
o 2
```
```
b 1 =: 35
b 2 =: 35
```
```
b 3 =: 60
b 4 =: 60
```
```
Figure 9.19: ANN for illustrating backpropagation algorithm with initial values for weights
```
Step 2. Present the first sample inputs and the corresponding output targets to the network. This is
shown in Figure 9.19.

Step 3. Pass the input values to the first layer (the layer with nodesh 1 andh 2 ).

Step 4. We calculate the outputs fromh 1 andh 2. We use the logistic activation function

```
f(x)=
```
#### 1

```
1 +e−x
```
#### :

```
outh 1 =f(w 1 ×i 1 +w 2 ×i 2 +b 1 × 1 )
=f( 0 : 15 × 0 : 05 + 0 : 20 × 0 : 10 + 0 : 35 × 1 )
=f( 0 : 3775 )
```
```
=
```
#### 1

```
1 +e−^0 :^3775
= 0 : 59327
outh 2 =f(w 3 ×i 1 +w 4 ×i 2 +b 2 × 1 )
=f( 0 : 25 × 0 : 05 + 0 : 30 × 0 : 10 + 0 : 35 × 1 )
=f( 0 : 3925 )
```
```
=
```
#### 1

```
1 +e−^0 :^3925
= 0 : 59689
```

#### CHAPTER 9. NEURAL NETWORKS 125

Step 5. We repeat this process for every layer. We get the outputs from the nodes in the output
layer as follows:

```
outo 1 =f(w 5 ×outh 1 +w 6 ×outh 2 +b 3 × 1 )
=f( 0 : 40 × 0 : 59327 + 0 : 45 × 0 : 59689 + 0 : 60 × 1 )
=f( 1 : 10591 )
```
```
=
```
#### 1

```
1 +e−^1 :^10591
= 0 : 75137
outo 2 =f(w 7 ×outh 1 +w 8 ×outh 2 +b 4 × 1 )
=f( 0 : 50 × 0 : 59327 + 0 : 55 × 0 : 59689 + 0 : 60 × 1 )
=f( 1 : 22492 )
```
```
=
```
#### 1

```
1 +e−^1 :^22492
= 0 : 77293
```
```
The sum of the squares of the output errors is given by
```
#### E=

#### 1

#### 2

```
(T 1 −outo 1 )^2 +
```
#### 1

#### 2

```
(T 2 −outo 2 )^2
```
```
=( 0 : 01 − 0 : 75137 )^2 +( 0 : 99 − 0 : 77293 )^2
= 0 : 298371
```
Step 6. We begin backward phase. We adjust the weights. We first adjust the weights leading to
the nodeso 1 ando 2 in the output layer and then the weights leading to the nodesh 1 andh 2
in the hidden layer. The adjusted values of the weightsw 1 ;:::;w 8 ;b 1 ;:::;b 4 are denoted
byw+ 1 ;:::;w+ 8 ;b+ 1 ;:::;b+ 4. The computations use a certain constantcalled thelearning
rate. In the following we have taken= 0 : 5.

```
(a) Computation of adjusted weights leading too 1 ando 2 :
```
```
o 1 =(T 1 −outo 1 )×outo 1 ×( 1 −outo 1 )
=( 0 : 01 − 0 : 75137 )× 0 : 75137 ×( 1 − 0 : 75137 )
=− 0 : 13850
w 5 +=w 5 +×o 1 ×outh 1
= 0 : 40 + 0 : 5 ×(− 0 : 13850 )× 0 : 59327
= 0 : 35892
w 6 +=w 6 +×o 1 ×outh 2
= 0 : 45 + 0 : 5 ×(− 0 : 13850 )× 0 : 59689
= 0 : 40867
b+ 3 =b 3 +×o 1 × 1
= 0 : 60 + 0 : 5 ×(− 0 : 13850 )× 1
= 0 : 53075
```
```
o 2 =(T 2 −outo 2 )×outo 2 ×( 1 −outo 2 )
=( 0 : 99 − 0 : 77293 )× 0 : 77293 ×( 1 − 0 : 77293 )
= 0 : 03810
w 7 +=w 7 +×o 2 ×outh 1
= 0 : 50 + 0 : 5 × 0 : 03810 × 0 : 59327
```

#### CHAPTER 9. NEURAL NETWORKS 126

#### = 0 : 51130

```
w 8 +=w 8 +×o 2 ×outh 2
= 0 : 55 + 0 : 5 × 0 : 03810 × 0 : 59689
= 0 : 56137
b+ 4 =b 4 +×o 2 × 1
= 0 : 60 + 0 : 5 × 0 : 03810 × 1
= 0 : 61905
```
```
(b) Computation of adjusted weights leading toh 1 andh 2 :
```
```
h 1 =(o 1 ×w 5 +o 2 ×w 7 )×outh 1 ×( 1 −outh 1 )
=(− 0 : 13850 × 0 : 40 + 0 : 03810 × 0 : 50 )× 0 : 59327 ×( 1 − 0 : 59327 )
=− 0 : 00877
w+ 1 =w 1 +×h 1 ×i 1
= 0 : 15 + 0 : 5 ×(− 0 : 00877 )× 0 : 05
= 0 : 14978
w+ 2 =w 2 +×h 1 ×i 2
= 0 : 20 + 0 : 5 ×(− 0 : 00877 )× 0 : 10
= 0 : 19956
b+ 1 =b 1 +×h 1 × 1
= 0 : 35 + 0 : 5 ×(− 0 : 00877 )× 1
= 0 : 34562
```
```
h 2 =(o 1 ×w 6 +o 2 ×w 8 )×outh 2 ×( 1 −outh 2 )
=((− 0 : 13850 )× 0 : 45 + 0 : 03810 × 0 : 55 )× 0 : 59689 ×( 1 − 0 : 59689 )
=− 0 : 00995
w+ 3 =w 3 +×h 2 ×i 1
= 0 : 25 + 0 : 5 ×(− 0 : 00995 )× 0 : 05
= 0 : 24975
w+ 4 =w 4 +×h 2 ×i 2
= 0 : 30 + 0 : 5 ×(− 0 : 00995 )× 0 : 10
= 0 : 29950
b+ 2 =b 2 +×h 2 × 1
= 0 : 35 + 0 : 5 ×(− 0 : 00995 )× 1
= 0 : 34503
```
Step 7. Now we set:

```
w 1 =w 1 +; w 2 =w+ 2 ; w 3 =w+ 3 ; w 4 =w 4 +
w 5 =w 5 +; w 6 =w+ 6 ; w 7 =w+ 7 ; w 8 =w 8 +
b 1 =b+ 1 ; b 2 =b+ 2 ; b 3 =b+ 3 ; b 4 =b+ 4
```
```
We choose the next sample input and the corresponding output targets to the network and
repeat Steps 2 to 6.
```
Step 8. The process in Step 7 is repeated until the root mean square of output errors is minimised.


#### CHAPTER 9. NEURAL NETWORKS 127

Remarks

1. The constant^12 is included in the expression forEso that the exponent is cancelled when we
    differentiate it. The result has been multiplied by a learning rate= 0 : 5 and so it doesnâA ̆Zt ́
    matter that we introduce the constant^12 inE.
2. In the above computations, the method used to calculate the adjusted weights is known as the
    delta rule.
3. The rule for computing the adjusted weights can be succinctly stated as follows. Letwbe a
    weight andw+its adjusted weight. LetEbe the the total sum of squares of errors. Thenw+
    is computed by
       w+=w−

#### @E

```
@w
```
#### :

```
Here@E@wis the gradient ofEwith respect tow; that is, the rate at whichEis changing with
respect tow. (The set of all such gradients specifies the direction in whichEis decreasing
the most rapidly, that is, the direction of quickest descent.) For example, it can be shown that
@E
@w 5
```
```
=−(T 1 −outo 1 )×outo 1 ×( 1 −outo 1 )×outh 1
```
```
=−o 1 ×outh 1
```
```
and so
```
```
w+ 5 =w 5 −
```
#### @E

```
@w 5
=w 5 +×o 1 ×outh 1
```
9.8.3 The algorithm

The backpropagation algorithm trains a given feed-forward multilayer neural network for a given set
of input patterns with known classifications. When each entry of the sample set is presented to the
network, the network examines its output response to the sample input pattern. The output response
is then compared to the known and desired output and the error value is calculated. Based on the
error, the connection weights are adjusted. The adjustments are based on the mean square error of
the output response to the sample input and it is known as thedelta learning rule. The set of these
sample patterns are repeatedly presented to the network until the error value is minimized.

Notations

Figures 9.20 and 9.21 show the various notations used in the algorithm.

```
M : Number of layers (excluding the input layer
which is assigned the layer number 0 )
Nj : Number of neurons (nodes) inj-th layer
Xp=(Xp 1 ;Xp 2 ;:::;XpN 0 ) :p-th training sample
Tp=(Tp 1 ;Tp 2 ;:::;TpNM) : Known output corresponding to
thep-th training sample
Op=(Op 1 ;Op 2 ;:::;OpNM) : Actual output by the network corresponding to
thep-th training sample
Yji : Output from thei-th neuron in layerj
Wjik : Connection weight fromk-th neuron in
layer(j− 1 )toi-th neuron in layerj
ji : Error value associated with thei-th neuron in layerj
```

#### CHAPTER 9. NEURAL NETWORKS 128

```
⋯ Tp 1 Op 1
```
```
⋯ Tp 2 Op 2
```
#### ⋯ ⋯ ⋯

```
⋯ TpN 0 OpN 0
```
```
j(layer #) j= 0 j= 1 j=M
```
```
Nj(# neurons)N 0 N 1 NM
```
```
Xp 1
```
```
Xp 2
```
```
XpN 0
```
```
Figure 9.20: Notations of backpropagation algorithm
```
```
ji Yij
```
#### :::

```
Y(j− 1 ) 1
```
```
Y(j− 1 ) 2 Wji 1
```
```
Wji 2
```
```
Y(j− 1 ) 3 Wji 3
```
```
Y(j− 1 )Nj− 1
```
```
WjiNj− 1
```
```
Yij=f∑
Nj− 1
k= 1 Y(j−^1 )kWjik
```
```
ij
```
```
Figure 9.21: Notations of backpropagation algorithm: Thei-th node in layerj
```
The algorithm

Step 1. Initialize connection weights into small random values.

Step 2. Present thepth sample input vector of pattern

```
Xp=(Xp 1 ;Xp 2 ;:::;XpN 0 )
```
```
and the corresponding output target
```
```
Tp=(Tp 1 ;Tp 2 ;:::;TpNM)
```
```
to the network.
```
Step 3. Pass the input values to the first layer, layer 1. For every input nodeiin layer 0, perform:
Y 0 i=Xpi:


#### CHAPTER 9. NEURAL NETWORKS 129

Step 4. For every neuroniin every layerj= 1 ; 2 ;:::;M, find the output from the neuron:

```
Yji=f∑
Nj− 1
k= 1 Y(j−^1 )kWjik;
```
```
where
f(x)=
```
#### 1

```
1 +exp(−x)
```
#### :

Step 5. Obtain output values. For every output nodeiin layerM, perform:

```
Opi=YMi:
```
Step 6. Calculate error valuejifor every neuroniin every layer in backward orderj=M;M−
1 ;:::; 2 ; 1 , from output to input layer, followed by weight adjustments. For the output
layer, the error value is:

```
Mi=YMi( 1 −YMi)(Tpi−YMi);
```
```
and for hidden layers:
```
```
ji=Yji( 1 −Yji)∑Nk=j 1 +^1 (j+ 1 )kW(j+ 1 )ki:
```
```
The weight adjustment can be done for every connection from neuronkin layer(j− 1 )to
every neuronjin every layeri:
```
```
Wjik+ =Wjik+jiYji;
```
```
whererepresents weight adjustment factor (called thelearning rate) normalized between
0 and 1.
```
Step 7. The actions in steps 2 through 6 will be repeated for every training sample patternp, and
repeated for these sets until the sum of the squares of output errors is minimized.

### 9.9 Introduction to deep learning

9.9.1 Definition

A neural network with multiple hidden layers is called aDeep Neural Network(DNN) and the
practice of training such network is referred to asdeep learning.

Remarks

In the terminology “deep learning”, the term “deep” is a technical term. It refers to the number of
layers in a neural network. Ashallow networkhas one so-called hidden layer, and a deep network
has more than one. Multiple hidden layers allow deep neural networks to learn features of the data
in a so-called feature hierarchy, because simple features recombine from one layer to the next, to
form more complex features. Networks with many layers pass input data (features) through more
mathematical operations than networks with few layers, and are therefore more computationally
intensive to train. Computational intensivity is one of the hallmarks of deep learning.
Figure 9.22 shows a shallow neural network and Figure 9.23 shows a deep neural network with
three hidden layers.


#### CHAPTER 9. NEURAL NETWORKS 130

```
Figure 9.22: A shallow neural network
```
```
Figure 9.23: A deep neural network with three hidden layers
```
9.9.2 Some applications

Deep learning applications are used in industries from automated driving to medical devices.

1. Automated driving:
    Automotive researchers are using deep learning to automatically detect objects such as stop
    signs and traffic lights. In addition, deep learning is used to detect pedestrians, which helps
    decrease accidents.
2. Aerospace and defense:
    Deep learning is used to identify objects from satellites that locate areas of interest, and iden-
    tify safe or unsafe zones for troops.
3. Medical research:
    Cancer researchers are using deep learning to automatically detect cancer cells. Teams at
    UCLA built an advanced microscope that yields a high-dimensional data set used to train a
    deep learning application to accurately identify cancer cells.
4. Industrial automation:
    Deep learning is helping to improve worker safety around heavy machinery by automatically
    detecting when people or objects are within an unsafe distance of machines.
5. Electronics:


#### CHAPTER 9. NEURAL NETWORKS 131

```
Deep learning is being used in automated hearing and speech translation. For example, home
assistance devices that respond to your voice and know your preferences are powered by deep
learning applications.
```
### 9.10 Sample questions

(a) Short answer questions

1. Explain the biological motivation for the formulation of the concept of artificial neural net-
    works.
2. With the aid of a diagram, explain the concept of an artificial neuron.
3. What is an activation function in an artificial neuron? Give some examples.
4. Define a perceptron.
5. Is neural network supervised or unsupervised learning? Why?
6. Is deep learning supervised or unsupervised? Why?
7. What is the basic idea of the backpropagation algorithm?
8. In the context of ANNs, what is meant by network topology?
9. Explain the different types of layers in an ANN.
10. What is the gradient descent method? How is used in the backpropagation algorithm?
11. A neuron with 4 inputs has the weights 1 ; 2 ; 3 ; 4 and bias 0. The activation function is linear,
say the functionf(x)= 2 x. If the inputs are 4 ; 8 ; 5 ; 6 , compute the output. Draw a diagram
representing the neuron.

(b) Long answer questions

1. Design a two layer network of perceptrons to implement A XOR B.
2. Explain the backpropagation algorithm.
3. Describe the perceptron learning algorithm.
4. What are the characteristics of an artificial neural networks.
5. Explain the concept of deep learning. Give some real life problems where this concept has
    been successfully applied.
6. Compute the output of the following neuron if the activation function is (i) the threshold
    function (ii) the sigmoid function (iii) the hyperbolic tangent function (assume the same bias
    0.5 for each node).

```
x 0 = 3 : 5
```
```
x 1 = 2 : 9
```
```
w 0 = 0 : 89
```
```
w 1 =− 2 : 07
```
```
x 2 = 1 : 2
```
```
w 2 = 0 : 08
```
```
Output (y)
```

#### CHAPTER 9. NEURAL NETWORKS 132

7. Which of the boolean functions AND, OR, XOR (or none of these) is represented by the
    following network of perceptrons (with unit step function as the activation function)?

```
x 1
```
```
w 1 = 1
```
```
x 2
```
```
w 2 = 1
```
```
w 3 =− 1
```
```
w 4 =− 1
```
```
w 5 = 3
```
```
b 1 =− 0 : 5
```
```
b 2 =− 0 : 5
```
```
b 4 = 0 : 5
```
```
b 3 ==− 1 : 5
```
```
Output (y)
```
8. Given the following network, compute the outputs fromo 1 ando 2 (assume that the activation
    function is the sigmoid function).

```
Inputi 1 =: 25
```
```
Inputi 2 =: 30
```
#### 1

```
Output 1
```
```
Output 2
```
#### 1

```
w 1 =: 17
h 1
```
```
w 2 =: 21
```
```
w 5 =: 52
o 1
```
```
w 6 =: 61
```
```
w 3 =: 18
```
```
h 2
w 4 =: 27
```
```
w 7 =: 55
```
```
w 8 =: 72
```
```
o 2
```
```
b 1 =: 12
b 2 =: 24
```
```
b 3 =: 48
b 4 =: 36
```
9. (Assignment question) Given the following data, use ANN with one hidden layer, appropriate
    initial weights and biases to compute the optimal values of the weights. Perform one iteration
    of the forward and phases of the backpropagation algorithm for each samples.

```
Sample Input 1 Input 2 Output target 1 Output target 2
1 1.20 2.30 0.53 0.76
2 0.23 0.37 1.17 2.09
```

Chapter 10

## Support vector machines

We begin this chapter by illustrating the basic concepts and terminology of the theory of support
vector machines by a simple example. We then introduce the necessary mathematical background,
which is essentially an introduction to finite dimensional vector spaces, for describing the general
concepts in the theory of support vector machines. The related algorithms without proofs are then
presented.

### 10.1 An example

10.1.1 Problem statement

Suppose we want to develop some criteria for determining the weather conditions under which tennis
can be played. To simplify the matters it has been decided to use the measures of temperature and
humidity as the critical parameters for the investigation. We have some data as given in Table 10.1
regarding the values of the parameters and the decisions taken as to whether to play tennis or not.
We are required to develop a criteria to know whether one would be playing tennis on a future date
if we know the values of the temperature and humidity of that date in advance.

10.1.2 Discussion and solution

We shall now see the various steps that lead to a solution of the problem using the ideas of support
vector machines.

```
temperature humidity play
85 85 no
60 70 yes
80 90 no
72 95 no
68 80 yes
74 73 yes
69 70 yes
75 85 no
83 78 no
```
```
Table 10.1: Example data with two class labels
```
#### 133


#### CHAPTER 10. SUPPORT VECTOR MACHINES 134

1. Two-class data set

This is our first observation regarding the data in Table 10.1. In Table 10.1, the data are classified
based on the values of the variable “play”. This variable has only two values or labels, namely “yes”
and ”no”. When there are only two class labels the data is said to be a “two-class data set”. So the
data in Table 10.1 is a two-class data set.

2. Scatter plot of the data

Since there are only two features or parameters, we may plot the values of one of the parameters, say
“temperature”, along the horizontal axis (that is, thex-axis) and the values of the other parameter
“humidity”, along the vertical axis (that is, they-axis). The data can be plotted in a coordinate plane
to get a scatter plot of the data. Figure 10.1 shows the scatter plot. In the figure the points which
correspond to the decision “yes” on playing tennis has been plotted as filled squares (◾) and which
correspond to the decision “no” has been marked as hollow circles (○).

Figure 10.1: Scatter plot of data in Table 10.1 (filled circles represent “yes” and unfilled circles
“no”)

3. A separating line

If we examine the plot in Figure 10.1, we can see that we can draw a straight line in the plane
separating the two types of points in the sense that all points plotted as filled squares are on one side
of the line and all points marked as hollow circles are on the other side of the line. Such a line is
called a “separating line” for the data. Figure 10.2 shows a separating line for the data in Table 10.1.
The equation of the separating line shown in Figure 10.2 is

```
5 x+ 2 y− 535 = 0 : (10.1)
```
It has the following property:

- If the data point with values(x′;y′)has the value “yes” for “play” (filled square), then

```
5 x′+ 2 y′− 535 < 0 : (10.2)
```
- If the data point with values(x;y)has the value “no” for “play” (hollow circle), then

```
5 x′+ 2 y′− 535 > 0 : (10.3)
```
If such a separating line exists for a given data then the data is said to be “linearly separable”.
Thus the data in table 10.1 is linearly separable. However note that not all data are linearly separable.


#### CHAPTER 10. SUPPORT VECTOR MACHINES 135

```
Figure 10.2: Scatter plot of data in Table 10.1 with a separating line
```
4. Several separating lines

Apparently, the conditions given in Eqs. (10.2) and (10.3) may be used as the criteria to know
whether one would be playing tennis on a future date if we know the values of the temperature
and humidity of that date in advance. But there are several separating lines and the problem of
determining which one to choose arises. Figure 10.3 shows two separating lines for the given data.

```
Figure 10.3: Two separating lines for the data in Table 10.1
```
4. Margin of a separating line

To choose the “best” separating line, we introduce the concept of the margin of a separating line.
Given a separating line for the data, we consider the perpendicular distances of the data points
from the separating line. Th double of the shortest perpendicular distance is called the “margin of the
separating line”. Figure??shows some of the perpendicular distances and the shortest perpendicular
distance for the data in Table 10.1 and for the separating line given by Eq. (10.1).

5. Maximal margin separating line

The “best” separating line is the one with the maximum margin.


#### CHAPTER 10. SUPPORT VECTOR MACHINES 136

```
Figure 10.4: Shortest perpendicular distance of a separating line from data points
```
The separating line with the maximum margin is called the “maximum margin line” or the “op-
timal separating line”. This line is also called the “support vector machine” for the data in Table
10.1.
Unfortunately, finding the equation of the maximum margin line is not a trivial problem. Figure
10.5 shows the maximum margin line for the data in Table 10.1. The equation of the maximum
margin line can be shown to be
7 x+ 6 y− 995 : 5 = 0 : (10.4)

```
Figure 10.5: Maximum margin line for data in Table 10.1
```
6. Support vectors

The data points which are closest to the maximum margin line are called the “support vectors”. The
support vectors are shown in Figure 10.6.

7. The required criterion

As per theory of support vector machines, the equation of the maximum margin line is used to
devise a criterion for taking a decision on whether to play tennis or not. Letx′andy′be the values


#### CHAPTER 10. SUPPORT VECTOR MACHINES 137

```
Figure 10.6: Support vectors for data in Table 10.1
```
of temperature and humidity on a given day. Then the decision as to whether play tennis on that day
is “yes” if
7 x+ 6 y− 995 : 5 < 0

and “no” if
7 x+ 6 y− 995 : 5 > 0 :

8. “Street” of maximum width separating “yes” points and “no” points

Considering Figure 10.6, we may draw a line through the support vectors 1 and 2 parallel to the
maximum margin line, and a line through support vector 3 parallel to the maximum margin line.
The two lines are shown as dashed lines in Figure 10.7. The region between these two dashed lines
can be thought of as a “road” or a “street” of maximum width that separates the “yes” data points
and the “no” data points.

Figure 10.7: Boundaries of “street” of maximum width separating “yes” points and “no” points in
Table 10.1


#### CHAPTER 10. SUPPORT VECTOR MACHINES 138

9. Final comments

```
i) Any line given an equation of the form
```
```
ax+by+c= 0
```
```
separates the coordinate plane into two halves. One half consists of all points for which
ax+by+c> 0 and the other half consists of all points for whichax+by+c< 0. Which half
is which depends the signs of the coefficientsa;b;c.
```
```
ii) Figure 10.8 shows the plot of the maximum margin line produced using the R programming
language.
```
Figure 10.8: Plot of the maximum margin line of data in Table 10.1 produced by the R programming
language

```
iii) In the sections below, we generalise the concepts introduced above to data sets having more
than two features.
```
### 10.2 Finite dimensional vector spaces

In Section 10.1 we have geometrically examined in detail the concepts of the theory of support
vector machines with an example having only two features. But, obviously, such a geometrical
approach is infeasible if there are more than two features. In such cases we have to resort to formal
algebraic/mathematical formalism to investigate the problem. The theory of what are known as
“finite dimensional vector spaces” provides such a formalism. We present below the absolutely
essential parts of this theory. Those who are interested in learning about the abstract concept of a
vector space may refer to any well written book on linear algebra.

10.2.1 Definition

We give the definition of a finite dimensional vector space here. We once again warn the reader
that we are introducing the terms with reference to a very special case of a finite dimensional vector


#### CHAPTER 10. SUPPORT VECTOR MACHINES 139

space and that all the terms given below have more general meanings.

Definition

Letnbe a positive integer. By an-dimensional vectorwe mean an orderedn-tuple of real numbers
of the form(x 1 ;x 2 ;:::;xn). We denote vectors byx⃗,⃗y, etc. In the vector⃗x=(x 1 ;x 2 ;:::;xn), the
numbersx 1 ;x 2 ;:::xnare called thecoordinatesor thecomponentsofx⃗. In the following, we call
real numbers asscalars.
The set of alln-dimensional vectors with the operations ofaddition of vectorsandmultiplication
of a vector by a scalarand with the definitions of thezero vectorand thenegative of a vectoras
defined below is an-dimensional vector space. It is denoted byRn.

1. Addition of vectors
    Let⃗x=(x 1 ;x 2 ;:::;xn)and⃗y=(y 1 ;y 2 ;:::;yn)be twon-dimensional vectors. The sum of
    ⃗xand⃗y, denoted byx⃗+y⃗, is defined by

```
⃗x+y⃗=(x 1 +y 1 ;x 2 +y 2 ;:::;xn+yn):
```
2. Multiplication by scalar
    Let be a scalar and⃗x=(x 1 ;x 2 ;:::;xn)be an-dimensional vector. The product ofx⃗by ,
    denoted by x⃗, is defined by

```
x⃗=(x 1 ;x 2 ;:::;xn):
```
```
When we write the product ofx⃗by , we always write the scalar on the left side of the
vector⃗xas we have done above.
```
3. The zero vector
    Then-dimensional vector( 0 ; 0 ;:::; 0 ), which has all components equal to 0 , is called the
    zero vector. It is also denoted by 0. From the context of the usage we can understand whether
    0 denotes the scalar 0 or the zero vector.
4. Negative of a vector
    Let⃗x=(x 1 ;x 2 ;:::;xn)be anyn-dimensional vector. The negative of⃗xis a vector denoted
    by−x⃗and is defined by
       −⃗x=(−x 1 ;−x 2 ;:::;−xn):
    We writex⃗+(−y⃗)asx⃗−⃗y.

10.2.2 Properties

Letnbe a positive integer. Let⃗x,y⃗,⃗zbe arbitrary vectors inRnand let;;
be arbitrary scalars.

1. Closure under addition:⃗x+y⃗is also an-dimensional vector.
2. Commutativity:x⃗+y⃗=⃗y+⃗x
3. Associativity:⃗x+(⃗y+z⃗)=(x⃗+y⃗)+z⃗
    (Because of this property, we can write the sums⃗x+(y⃗+⃗z)and(x⃗+y⃗)+⃗zin the form
    ⃗x+y⃗+z⃗.)
4. Existence of identity for addition:x⃗+ 0 =x⃗
5. Existence of inverse for addition:x⃗+(−x⃗)= 0
6. Closure under scalar multiplication: x⃗is also an-dimensional vector.


#### CHAPTER 10. SUPPORT VECTOR MACHINES 140

7. Compatibility of multiplication of a vector by a scalar with multiplication of scalars: ( ⃗x)=
    ( )x⃗
8. Distributivity of scalar multiplication over vector addition: (⃗x+y⃗)= ⃗x+ ⃗y
9. Distributivity of scalar multiplication over addition of scalars:( + )x⃗= ⃗x+ ⃗x
10. Existence of identity element for scalar multiplication: 1 x⃗=x⃗

Example of computation

Letn= 3. Letx⃗=(− 1 ; 2 ; 3 );y⃗=( 2 ; 0 ;− 1 );⃗z=( 1 ; 1 ; 0 ), = 2 ;=− 3 ;
= 4 and= 5. The
expression( x⃗+ y⃗+ z⃗)can be computed in several different ways. One of the methods is shown
below.

```
( ⃗x+ y⃗+ z⃗)= 5 ( 2 (− 1 ; 2 ; 3 )+(− 3 )( 2 ; 0 ;− 1 )+ 4 ( 1 ; 1 ; 0 ))
= 5 ((− 2 ; 4 ; 6 )+(− 6 ; 0 ; 3 )+( 4 ; 4 ; 0 ))
= 5 ((− 8 ; 4 ; 9 )+( 4 ; 4 ; 0 ))
= 5 (− 4 ; 8 ; 9 )
=(− 20 ; 40 ; 45 )
```
10.2.3 Norm and inner product

1. Norm
    The norm of then-dimensional vectorx⃗=(x 1 ;x 2 ;:::;xn), denoted bySSx⃗SS, is defined by

```
Yx⃗Y=
```
#### ¼

```
x^21 +x^22 + ⋯ +x^2 n:
```
2. Inner product
    The inner product of⃗x=(x 1 ;x 2 ;:::;xn)andy⃗=(y 1 ;y 2 ;:::;yn), denoted by⃗x **⋅** ⃗y, is defined
    by
       ⃗x **⋅** y⃗=x 1 y 1 +x 2 y 2 + ⋯ +xnyn:
    Note that we have
       Y⃗xY=

#### √

```
⃗x ⋅ x:⃗
```
3. Angle between two vectors
    The anglebetween two vectorsx⃗andy⃗is defined by

```
cos=
```
```
x⃗ ⋅ y⃗
Yx⃗YY⃗yY
```
#### :

4. Perpendicularity
    Two vectorsx⃗=(x 1 ;x 2 ;:::;xn)and⃗y=(y 1 ;y 2 ;:::;yn)are said to be perpendicular (or,
    orthogonal) if
       x⃗ **⋅** y⃗= 0 :


#### CHAPTER 10. SUPPORT VECTOR MACHINES 141

Example

Letn= 4 and let⃗x=(− 1 ; 2 ; 0 ; 3 )andy⃗=( 2 ; 3 ; 1 ;− 4 ).

```
Y⃗xY=
```
#### »

#### (− 1 )^2 + 22 + 02 + 32

#### =

#### √

#### 14

```
Yy⃗Y=
```
#### »

#### 22 + 32 + 12 +(− 4 )^2

#### =

#### √

#### 30

```
⃗x ⋅ ⃗y=(− 1 )× 2 + 2 × 3 + 0 × 1 + 3 ×(− 4 )
=− 8
```
```
cos=
```
#### − 8

#### √

#### 14

#### √

#### 30

#### =− 0 : 39036

```
= 112 : 98 degrees
```
Sincex⃗ **⋅** y⃗≠ 0 the vectorsx⃗andy⃗are not orthogonal.

### 10.3 Hyperplanes

Hyperplanes are certain subsets of finite dimensional vector spaces which are similar to straight lines
in planes and planes in three-dimensional spaces.

10.3.1 Definition

Consider then-dimensional vector spaceRn. The set of all vectors

```
⃗x=(x 1 ;x 2 ;:::;xn)
```
inRnwhose components satisfy an equation of the form

0 + 1 x 1 + 2 x 2 + ⋯ + (^) nxn= 0 ; (10.5)
where 0 ; 1 ; 2 ;:::;nare scalars, is called ahyperplanein the vector spaceRn.
Remarks 1
Letx⃗= (x 1 ;x 2 ;:::;xn)and ⃗= ( 1 ; 2 ;:::;n), then using the notation of inner product,
Eq.(10.5) can be written in the following form:
0 + ⃗ **⋅** ⃗x= 0 :
Remarks 2
The hyperplane inRndefined by Eq.(10.5) divides the spaceRninto two disjoint halves. One of
the two halves consists of all vectors⃗xfor which
0 + 1 x 1 + 2 x 2 + ⋯ + (^) nxn> 0
and the other half consists of all vectors⃗xfor which
0 + 1 x 1 + 2 x 2 + ⋯ + (^) nxn< 0 :


#### CHAPTER 10. SUPPORT VECTOR MACHINES 142

10.3.2 Special cases

Hyperplanes in 2 -dimensional vector spaces: Straight lines

Consider the 2 -dimensional vector spaceR^2. Vectors in this space are ordered pairs of the form
(x 1 ;x 2 ). Choosing appropriate coordinate axes, such a vector can be represented by a point with
coordinates⃗x=(x 1 ;x 2 )in the plane. So, the vector spaceR^2 can be identified with the set of points
in a plane. In this special case, the normYxYis the distance of the point(x 1 ;x 2 )in the plane from
the origin. The angle between the vectors⃗x=(x 1 ;x 2 )andy⃗=(y 1 ;y 2 )is the angle between the
lines joining the origin to the points(x 1 ;x 2 )and(y 1 ;y 2 ).
Consider the set of all vectors⃗x=(x 1 ;x 2 )inR^2 which satisfy the following equation:

```
0 + 1 x 1 + 2 x 2 = 0
```
where 0 + 1 ; 2 are scalars. From elementary analytical geometry we can see that the correspond-
ing set of points in the plane form a straight line in the plane. This straight line divides the plane
into two disjoint halves (see Figure 10.9). It can be proved that one of the two halves consists of all
points for which
0 + 1 x 1 + 2 x 2 > 0

and the other half consists of all points for which

```
0 + 1 x 1 + 2 x 2 < 0 :
```
```
x 1
```
```
x 2
```
#### O

```
Equation of line:
0 + 1 x 1 + 2 x 2 = 0
(assume 0 < 0 )
```
```
Half plane where
0 + 1 x 1 + 2 x 2 < 0
```
```
Half plane where
0 + 1 x 1 + 2 x 2 > 0
```
```
Figure 10.9: Half planes defined by a line
```
Hyperplanes in 3 -dimensional vector spaces: Planes

Consider the 3 -dimensional vector spaceR^3. Vectors in this space are ordered triples of the form
(x 1 ;x 2 ;x 3 ). Choosing appropriate coordinate axes, such a vector can be represented by a point
with coordinates⃗x=(x 1 ;x 2 ;x 3 )in the ordinary three-dimensional space. So, the vector spaceR^3
can be identified with the set of points in the three-dimensional space. As in the case ofR^2 , the
normYxYis the distance of the point(x 1 ;x 2 ;x 3 )from the origin. The angle between the vectors
⃗x=(x 1 ;x 2 ;x 3 )andy⃗=(y 1 ;y 2 ;y 3 )is the angle between the lines joining the origin to the points
(x 1 ;x 2 ;x 3 )and(y 1 ;y 2 ;y 3 ).
Consider the set of all vectors⃗x=(x 1 ;x 2 ;x 3 )inR^3 which satisfy the following equation:

```
0 + 1 x 1 + 2 x 2 + 3 x 3 = 0
```

#### CHAPTER 10. SUPPORT VECTOR MACHINES 143

where 0 ; 1 ; 2 ; 3 are scalars. From elementary analytical geometry we can see that the corre-
sponding set of points in space form a plane. This plane divides the space into two disjoint halves.
It can be proved that one of the two halves consists of all points for which

```
0 + 1 x 1 + 2 x 2 + 3 x 3 > 0
```
and the other half consists of all points for which

```
0 + 1 x 1 + 2 x 2 + 3 x 3 < 0 :
```
Geometry of hyperplanes inn-dimensional vector spaces

By analogy with a plane (which is a geometrical object having two dimensions) and the space of
our experience (which is a geometrical world having three dimensions) we imagine that there is a
geometrical world or object havingndimensions for any value ofn. We also imagine that the points
in this world can be represented by orderedntuples of the formx⃗=(x 1 ;x 2 ;:::;xn). We now
identify the set ofn-dimensional vectors with the points in this geometrical world ofn-dimensions.
Because of this identification, vectors in then-dimensional vector spaceRnare also referred as
points in an-dimensional space. The hyperplanes inRnare defined by analogy with the geometrical
straight lines and planes.

10.3.3 Distance of a hyperplane from a point

In two-dimensional space, that is, in a plane, using elementary analytical geometry, it can be shown
that the perpendicular distancePNof a pointP(x′ 1 ;y′ 1 )from a line

```
0 + 1 x 1 + 2 x 2 = 0
```
is given by

```
PN=
```
```
S 0 + 1 x′ 1 + 2 x′ 2 S
»
```
21 + (^22)

#### :

Similarly, in three-dimensional space, using elementary analytical geometry, it can be shown that
the perpendicular distancePNof a pointP(x′ 1 ;x′ 2 ;x′ 3 )from a plane

```
0 + 1 x 1 + 2 x 2 + 3 x 3 = 0
```
is given by (see Figure 10.10)

#### PN=

```
S 0 + 1 x′ 1 + 2 x′ 2 + 3 x′ 3 S
»
```
21 + 22 + (^23)

#### :

```
0 + 1 x 1 + 2 x 2 + 3 x 3 = 0
```
#### N

```
P(x′ 1 ;x′ 2 ;x′ 3 )
```
```
Figure 10.10: Perpendicular distance of a point from a plane
```
```
Motivated by these special cases, we introduce the following definition.
```

#### CHAPTER 10. SUPPORT VECTOR MACHINES 144

Definition

InRn, theperpendicular distancePNof a pointP(x′ 1 ;x′ 2 ;:::;x′n)from a hyperplane

0 + 1 x 1 + 2 x 2 +:::+ (^) nxn= 0
is given by
PN=
S 0 + 1 x′ 1 + 2 x′ 2 +:::+ (^) nx′nS
»
21 + 22 +:::+ (^2) n

#### : (10.6)

Remarks

Letx⃗′=(x′ 1 ;x′ 2 ;:::;x′n)and ⃗=( 1 ; 2 ;:::;n), then using the notations of inner product and
norm, Eq.(10.6) can be written in the following form:

#### PN=

```
S 0 + ⃗ ⋅ x⃗′S
Y⃗x′Y
```
#### :

### 10.4 Two-class data sets

In a machine learning problem, the variable being predicted is called theoutput variable, thetarget
variable, thedependent variableor theresponse. Atwo-class dataset is a data set in which the
target variable takes only one of two possible values only. If the target variable takes more than two
possible values, the data set is called amulti-class dataset.
In a two-class data set, the set of values of the target variable may be{“yes”, “no”}, or{“TRUE”, ”FALSE”},
or{ 0 ; 1 }, or{− 1 ;+ 1 }or any such similar set.
The methods of support vector machines were originally developed for classification problems
involving two-class data sets. So in this chapter we consider mainly two-class data sets.

### 10.5 Linearly separable data

10.5.1 Definitions

Consider a two-class data set havingnnumeric features and two possible class labels− 1 and+ 1.
Let the vectorx⃗=(x 1 ;:::;xn)represent the values of the features in one instance of the data set.
We say that the data set islinearly separableif we can find a hyperplane in then-dimensional vector
spaceRn, say

0 + 1 x 1 + 2 x 2 + ⋯ + (^) nxn= 0 (10.7)
having the following two properties:

1. For each instance⃗xwith class label− 1 we have

0 + 1 x 1 + 2 x 2 + ⋯ + (^) nxn< 0 :

2. For each instance⃗xwith class label+ 1 we have

0 + 1 x 1 + 2 x 2 + ⋯ + (^) nxn> 0 :
A hyperplane given by Eq.(10.7) having the two properties given above is called aseparating hy-
perplanefor the data set.
Remarks 1
If a data set with two class labels is linearly separable, then, in general, there will be several sepa-
rating hyperplanes for the data set. This is illustrated in the example below.


#### CHAPTER 10. SUPPORT VECTOR MACHINES 145

Remarks 2

Given a two-class data set, there is no simple method for determining whether the data set is linearly
separable. One of the efficient ways for doing this is to apply the methods of linear programming.
We omit the details.

10.5.2 Example

Example 1

We have seen in Section 10.1 that the data in Table 10.1 is linearly separable.

Example 2

Show that the data set given in Table 10.2 is not separable.

```
x y Class label
0 0 0
0 1 1
1 0 1
1 1 0
```
```
Table 10.2: Example of a two-class data that is not linearly separable
```
Solution

The scatterplot of data in TableTableVXOR shown in Figure 10.11 shows that the data is not linearly
separable.

```
Figure 10.11: Scatterplot of data in Table 10.2
```
### 10.6 Maximal margin hyperplanes

10.6.1 Definitions

Consider a linearly separable data set having two class labels “− 1 ” and “+ 1 ”. Consider a separating
hyperplaneHfor the data set.


#### CHAPTER 10. SUPPORT VECTOR MACHINES 146

1. Consider the perpendicular distances from the training instances to the separating hyperplane
    Hand consider the smallest such perpendicular distance. The double of this smallest distance
    is called themarginof the separating hyperplaneH.
2. The hyperplane for which the margin is the largest is called themaximal margin hyperplane
    (also calledmaximum margin hyperplane) or theoptimal separating hyperplane.
3. The maximal margin hyperplane is also called thesupport vector machinefor the data set.
4. The data points that lie closest to the maximal margin hyperplane are called thesupport vec-
    tors.

```
Figure 10.12: Maximal separating hyperplane, margin and support vectors
```
10.6.2 Special cases

To fix ideas, let us consider two special datasets in 2-dimensional space, namely, datasets having 2
and 3 examples.

Dataset with two examples

Consider the dataset in Table 10.3.

```
Example no. x 1 x 2 Class
1 2 1 + 1
2 4 3 − 1
```
```
Table 10.3: 2 -dimensional dataset with 2 examples
```
Geometrically it can be easily seen that the maximum margin hyperplane for this data is the
perpendicular bisector of the line segment joining the points( 2 ; 1 )and( 4 ; 3 )(see Figure 10.13).
This is true for any two-sample dataset in two-dimensional space.

Dataset with three examples

Consider a dataset with three examples from a two-dimensional space. Let these examples corre-
spond to the pointsA;B;Cin the coordinate plane. Two of these examples, sayBandC, must have
the same class label say+ 1 and the other pointAmust have a different class label, say− 1.
The maximal margin hyperplane of the dataset can be obtained as follows. Draw the line joining
BandCand draw the line throughAparallel toBC. The line midway between these two lines in
the maximal margin hyperplane of the three-sample dataset


#### CHAPTER 10. SUPPORT VECTOR MACHINES 147

```
x 1
```
```
x 2
```
#### A( 2 ; 1 )

#### B( 4 ; 3 )

```
( 3 ; 2 )Midpoint ofAB
```
#### ( 0 ; 0 )

```
Maximum margin hyperplane:
x 1 +x 2 − 5 = 0
```
```
Figure 10.13: Maximal margin hyperplane of a 2-sample set in 2-dimensional space
```
```
x 1
```
```
x 2
```
#### A( 2 ; 2 )

#### B( 4 ; 5 )

#### C( 7 ; 4 )

#### ( 0 ; 0 )

```
Maximal margin hyperplane
x 1 + 3 x 2 −^272 = 0
```
```
Figure 10.14: Maximal margin hyperplane of a 3-sample set in 2-dimensional space
```
### 10.7 Mathematical formulation of the SVM problem

The SVM problem is the problem of finding the equation of the SVM, that is, the maximal margin
hyperplane, given a linearly separable two-class data set. By the very definition of SVM, this is an
optimisation problem. The give below the mathematical formulation of this optimisation problem.

10.7.1 Notations and preliminaries

- Assume that we are given a two-class training dataset ofNpoints of the form

```
(x⃗ 1 ;y 1 );(x⃗ 2 ;y 2 );:::;(x⃗N;yN):
```
```
where theyi’s are either+ 1 or 1 (the class labels). Eachx⃗iis an-dimensional real vector.
```
- We assume that the dataset is linearly separable.
- Any hyperplane can be written as the set of pointsx⃗=(x 1 ;:::;xn)satisfying an equation of
    the form
       w⃗⋅⃗x−b= 0 :


#### CHAPTER 10. SUPPORT VECTOR MACHINES 148

- Since the training data is linearly separable, we can select two parallel hyperplanes that sep-
    arate the two classes of data, so that the distance between them is as large as possible. The
    maximum margin hyperplane is the hyperplane that lies halfway between them. It can be
    shown that these hyperplanes can be described by equations of the following forms:

```
w⃗⋅x⃗−b=+ 1 (10.8)
w⃗⋅x⃗−b=− 1 (10.9)
```
- For any point on or “above” the hyperplane Rq.(10.8), the class label is+ 1. This implies that

```
w⃗⋅x⃗i−b≥+ 1 ;ifyi=+ 1 (10.10)
```
```
Similarly, for any point on or “below” the hyperplane Eq.(10.9), the class label is− 1. This
implies that
w⃗⋅⃗xi−b≤− 1 ;ifyi=− 1 : (10.11)
```
- The two conditions in Eq.10.10 and Eq.10.11 can be written as a single condition as follows:

```
yi(w⃗⋅⃗xi−b)≥ 1 ; for all 1 ≤i≤N:
```
- Now, the distance between the two hyperplanes in Eq.(10.8) and Eq.(10.9) is
    2
Yw⃗Y:

```
So, to maximize the distance between the planes we have to minimizeYw⃗Y. Further we also
note thatYw⃗Yis minimum when^12 Yw⃗Y^2 is minimum. (The square of the norm is used to avoid
square-roots and the factor “^12 ” is introduced to simplify certain expressions.)
```
10.7.2 Formulation of the problem

Based on the above discussion, we now formulate the SVM problem as the following optimization
problem.

Problem

Given a two-class linearly separable dataset ofNpoints of the form

```
(x⃗ 1 ;y 1 );(⃗x 2 ;y 2 );:::;(⃗xN;yN):
```
where theyi’s are either+ 1 or 1 , find a vectorw⃗and a numberbwhich

```
minimize
```
#### 1

#### 2

```
Yw⃗Y^2
```
```
subject to yi(w⃗⋅⃗xi−b)≥ 1 ;fori= 1 ;:::N
```
10.7.3 The SVM classifier

The solution of the SYM problem gives us a claasifier for classifying unclassified data instances.
This is known as theSVM classifierfor a given dataset.

The classifier

Letw⃗=w⃗∗andb=b∗be a solution of the SVM problem. Letx⃗be an unclassified data instance.

- Assign the class label+ 1 to⃗xifw⃗∗ **⋅** x⃗−b∗> 0.
- Assign the class label− 1 to⃗xifw⃗∗ **⋅** x⃗−b∗< 0.


#### CHAPTER 10. SUPPORT VECTOR MACHINES 149

### 10.8 Solution of the SVM problem

The SVM optimization problem as formulated above is an example of a constrained optimization
problem. The general method for solving it is to convert it into a quadratic programming problem
and then apply the algorithms for solving quadratic programming problems. These methods yield
the following solution to the SVM problem. The details of these processes are beyond the scope of
these notes.

10.8.1 Solution

The vectorw⃗and the scalarbare given by

```
w⃗=
```
```
N
Q
i= 1
```
(^) iyi⃗xi (10.12)
b=

#### 1

#### 2

```
min
i∶yi=+ 1
```
```
(w⃗ ⋅ ⃗xi)+max
i∶yi=− 1
```
```
(w⃗ ⋅ ⃗xi) (10.13)
```
where ⃗=( 1 ; 2 ;:::;N)is a vector which maximizes

```
N
Q
i= 1
```
(^) i−

#### 1

#### 2

```
N
Q
i= 1 ;j= 1
```
(^) i (^) jyiyj(⃗xi **⋅** x⃗j)
subject to
N
Q
i= 1
(^) iyi= 0
(^) i> 0 fori= 1 ; 2 ;:::;N:
Remarks
It can be proved that an (^) iis nonzero only if⃗xilies on the two margin boundaries, that is, only if⃗xi
is a support vector. So, to specify a solution to the SVM problem, we need only specify the support
vectors⃗xiand the corresponding coefficients (^) iyi.
10.8.2 An algorithm to find the SVM classifier
The solution of the SVM problem given in Section??can be used to develop an algorithm to find a
SVM classifier for linearly separable two-class dataset. Here is an outline of such an algorithm.
Algorithm to find SVM classifier
Given a two-class linearly separable dataset ofNpoints of the form
(x⃗ 1 ;y 1 );(⃗x 2 ;y 2 );:::;(⃗xN;yN);
where theyi’s are either+ 1 or 1 :
Step1. Find ⃗=( 1 ; 2 ;:::;N)which maximizes

#### ( ⃗)=

```
N
Q
i= 1
```
(^) i−

#### 1

#### 2

```
N
Q
i= 1 ;j= 1
```
(^) i (^) jyiyj(⃗xi **⋅** x⃗j)
subject to
N
Q
i= 1
(^) iyi= 0
(^) i> 0 fori= 1 ; 2 ;:::;N:


#### CHAPTER 10. SUPPORT VECTOR MACHINES 150

Step2. Computew⃗=∑Ni= 1 iyix⃗i.

Step3. Computeb=^12 mini∶yi=+ 1 (w⃗ **⋅** x⃗i)+maxi∶yi=− 1 (w⃗ **⋅** ⃗xi).

Step4. The SVM classifier function is given by

```
f(x⃗)=w⃗ ⋅ x⃗−b (10.14)
```
where (^) iis nonzero only ifx⃗iis a support vector.
Remarks
There are specialised software packages for solving the SVM optimization problem. For example,
there is a special package calledsvmin the R programming language to solve such problems.
10.8.3 Illustrative example
Problem 1
Using the SVM algorithm, find the SVM classifier for the follwoing data.
Example no. x 1 x 2 Class
1 2 1 + 1
2 4 3 − 1
Solution
For this data we have:
N= 2
x⃗ 1 =( 2 ; 1 ); y 1 =+ 1
x⃗ 2 =( 4 ; 3 ); y 2 =− 1
⃗=( 1 ; 2 )
Step 1. We have:

#### ( ⃗)=

```
N
Q
i= 1
```
(^) i−

#### 1

#### 2

```
N
Q
i= 1 ;j= 1
```
(^) i (^) jyiyj(⃗xi **⋅** x⃗j)

#### =( 1 + 2 )−

#### 1

#### 2

```
[ 11 y 1 y 1 (x⃗ 1 ⋅ ⃗x 1 )+ 12 y 1 y 2 (x⃗ 1 ⋅ ⃗x 2 )+
```
```
21 y 2 y 1 (x⃗ 2 ⋅ ⃗x 1 )+ 22 y 2 y 2 (x⃗ 2 ⋅ ⃗x 2 )]
```
```
=( 1 + 2 )−
1
2
```
#### [ 12 (+ 1 )(+ 1 )( 2 × 2 + 1 × 1 )+ 12 (+ 1 )(− 1 )( 2 × 4 + 1 × 3 )+

#### 21 (− 1 )(+ 1 )( 4 × 2 + 3 × 1 )+ 22 (− 1 )(− 1 )( 4 × 4 + 3 × 3 )]

#### =( 1 + 2 )−

#### 1

#### 2

####  521 − 2212 + 2522 

```
N
Q
i= 1
```
(^) iyi= 1 y 1 + 2 y 2
= 1 − (^2)
We have to solve the following problem.


#### CHAPTER 10. SUPPORT VECTOR MACHINES 151

```
Problem
Find values of 1 and 2 which maximizes
```
#### ( ⃗)=( 1 + 2 )−

#### 1

#### 2

####  521 − 2212 + 2522 

```
subject to the conditions
1 − 2 = 0 ; 1 > 0 ; 2 > 0 :
```
```
Solution
To find the required values of 1 and 2 , we note that from the constraints we have 2 = 1 :
Using this in the expression forwe get
```
```
( ⃗)= 21 − 421 :
```
```
Forto be maximum we must have
```
```
d
d 1
```
#### = 2 − 81 = 0

```
that is
1 =
```
#### 1

#### 4

```
and so we also have
2 =
```
#### 1

#### 4

#### :

```
(For this value of 1 , clearlyd
```
(^2) f
d^21 <^0 andfis indeed maximum. Also we have^1 >^0 and
2 > 0 .)
Step 2. Now we have
w⃗=
N
Q
i= 1
(^) iyi⃗xi
= 1 y 1 x⃗ 1 + 2 y 2 ⃗x 2
=

#### 1

#### 4

#### (+ 1 )( 2 ; 1 )+

#### 1

#### 4

#### (− 1 )( 4 ; 3 )

#### =

#### 1

#### 4

#### (− 2 ;− 2 )

#### =(−^12 ;−^12 )

Step 3. Next we find

```
b=
```
#### 1

#### 2

```
min
i∶yi=+ 1
```
```
(w⃗ ⋅ ⃗xi)+max
i∶yi=− 1
```
```
(w⃗ ⋅ ⃗xi)
```
#### =

#### 1

#### 2

```
(w⃗ ⋅ x⃗ 1 )+(w⃗ ⋅ x⃗ 2 )
```
```
=
```
#### 1

#### 2

#### (−^14 × 2 −^12 × 1 )+(−^12 × 4 −^12 × 3 )

#### =

#### 1

#### 2

#### −^102 

#### =−

#### 5

#### 2


#### CHAPTER 10. SUPPORT VECTOR MACHINES 152

Step 4. Letx⃗=(x 1 ;x 2 ). The SVM classifier function is given by

```
f(x⃗)=w⃗ ⋅ x⃗−b
=(−^12 ;−^12 ) ⋅ (x 1 ;x 2 )−(−^52 )
```
```
=−
```
#### 1

#### 2

```
x 1 −
```
#### 1

#### 2

```
x 2 +
```
#### 5

#### 2

#### =−

#### 1

#### 2

```
(x 1 +x 2 − 5 )
```
Step 5. The equation of the maximal margin hyperplane is

```
f(x⃗)= 0
```
```
that is
−
```
#### 1

#### 2

```
(x 1 +x 2 − 5 )= 0
```
```
that is
x 1 +x 2 − 5 = 0 :
Note that this the equation of the perpendicular bisector of the line segment joining the
points( 2 ; 1 )and( 4 ; 3 )(see Figure 10.13).
```
Problem 2

Using the SVM algorithm, find the SVM classifier for the follwoing data.

```
Example no. x 1 x 2 Class
1 2 2 − 1
2 4 5 + 1
3 7 4 + 1
```
Solution

For this data we have:

```
N= 3
x⃗ 1 =( 2 ; 2 ); y 1 =− 1
x⃗ 2 =( 4 ; 5 ); y 2 =+ 1
x⃗ 3 =( 7 ; 4 ); y 3 =+ 1
⃗=( 1 ; 2 ; 3 )
x⃗=(x 1 ;x 2 )
```
Srep 1. We have

#### ( ⃗)=

```
N
Q
i= 1
```
#### 1 −

#### 1

#### 2

```
N
Q
i= 1 ;j= 1
```
(^) i (^) jyiyj(x⃗i **⋅** ⃗xj)

#### =

```
3
Q
i= 1
```
#### 1 −

#### 1

#### 2

```
3
Q
i= 1 ;j= 1
```
(^) i (^) jyiyj(x⃗i **⋅** ⃗xj)
We have
(x⃗ 1 **⋅** ⃗x 1 )= 08 ; (x⃗ 1 **⋅** ⃗x 2 )= 18 ; (x⃗ 1 **⋅** x⃗ 3 )= 22
(⃗x 2 **⋅** x⃗ 1 )= 18 ; (x⃗ 2 **⋅** x⃗ 2 )= 41 ; (x⃗ 2 **⋅** ⃗x 3 )= 48 ;
(x⃗ 3 **⋅** ⃗x 1 )= 22 ; (x⃗ 3 **⋅** ⃗x 2 )= 48 ; (x⃗ 3 **⋅** x⃗ 3 )= 65


#### CHAPTER 10. SUPPORT VECTOR MACHINES 153

```
Substituting these and simplifying we get
```
#### ( ⃗)=( 1 + 2 + 3 )−

#### 1

#### 2

#### [ 821 + 4122 + 6523 − 3612 − 4413 + 9623 ]

```
We also have
N
Q
i= 1
```
(^) iyi=− 1 + 2 + (^3)
Now we have to solve the following problem.
Problem
Find ⃗=( 1 ; 2 ; 3 )which maximizes

#### ( ⃗)=( 1 + 2 + 3 )−

#### 1

#### 2

#### [ 821 + 4122 + 6523 − 3612 − 4413 + 9623 ]

```
subject to the conditions
```
```
− 1 + 2 + 3 = 0 ; 1 > 0 ; 2 > 0 ; 3 > 0 :
```
```
Solution
From the constraints we have
1 = 2 + 3 :
Using this in the expression for( ⃗)and simplifying we get
```
#### ( ⃗)= 2 ( 2 + 3 )−

#### 1

#### 2

#### ( 1322 + 3223 + 2923 )

```
When( ⃗)is maximum we have
```
```
@
@ 2
```
#### = 0 ;

#### @

#### @ 3

#### = 0 (10.15)

```
that is
2 − 132 − 163 = 0 ; 2 − 162 − 293 = 0 :
Solving these equations we get
```
#### 2 =

#### 26

#### 121

#### ; 3 =−

#### 6

#### 121

```
Hence
1 =
```
#### 26

#### 121

#### −

#### 6

#### 121

#### =

#### 20

#### 121

#### :

```
(The conditions given in Eq.(??) are only necessary conditions for getting a maximum
value for( ⃗). It can be shown that the values for 2 and 3 obtained above do indeed
satisfy the sufficient conditions for yielding a maximum value of( ⃗).)
```
Srep 2. Now we have

```
w⃗=
```
```
N
Q
i= 1
```
(^) iyi⃗xi

#### =

#### 20

#### 121

#### (− 1 )( 2 ; 2 )+

#### 26

#### 121

#### (+ 1 )( 4 ; 5 )−

#### 6

#### 121

#### (+ 1 )( 7 ; 4 )

#### =( 112 ; 116 )


#### CHAPTER 10. SUPPORT VECTOR MACHINES 154

Srep 3. We have

```
b=
```
#### 1

#### 2

```
min
i∶yi=+ 1
(w⃗ ⋅ x⃗i)+max
i∶yi=− 1
(w⃗ ⋅ x⃗i)
```
#### =

#### 1

#### 2

```
min{(w⃗ ⋅ x⃗ 2 );(w⃗ ⋅ ⃗x 3 )}+max{(w⃗ ⋅ ⃗x 1 )}
```
#### =

#### 1

#### 2

```
min{^3811 ;^3811 }+max{^1611 }
```
#### =

#### 1

#### 2

#### 

#### 38

#### 11

#### +

#### 16

#### 11

#### 

#### =

#### 27

#### 11

Srep 4. The SVM classifier function is

```
f(⃗x)=w⃗ ⋅ x⃗−b
```
```
=
```
#### 2

#### 11

```
x 1 +
```
#### 6

#### 11

```
x 2 −
```
#### 27

#### 11

#### :

Srep 5. The equation of the maximal hyperplane is

```
f(x⃗)= 0
```
```
that is
2
11
```
```
x 1 +
```
#### 6

#### 11

```
x 2 −
```
#### 27

#### 11

#### = 0

```
that is
x 1 + 3 x 2 −
```
#### 27

#### 2

#### = 0 :

```
(See Figure 10.14.)
```
### 10.9 Soft margin hyperlanes

The algorithm for finding the SVM classifier will give give a solution only if the the given two-class
dataset is linearly separable. But, in real life problems, two-class datasets are only rarely linearly
separable. In such a case we introduce additional variables,i, calledslack variableswhich store
deviations from the margin. There are two types of deviation: An instance may lie on the wrong
side of the hyperplane and be misclassified. Or, it may be on the right side but may lie in the margin,
namely, not sufficiently away from the hyperplane (see Figure 10.15).
Ifi= 0 , thenx⃗iis correctly classified and there is no problem withx⃗i. If 0 <i< 1 thenx⃗iis
correctly classified but it is in the margin. Ifi> 1 ,⃗xiis misclassified. Th sum∑Ni= 1 iis defined
as thesoft errorand this is added as a penalty to the function to be minimized. We also introduce a
factorCto the soft error.
With these modifications, we now reformulate the SVM problem as follows (see Section 10.7.2
for the original formulation of the problem):

Reformulated problem

Given a two-class linearly separable dataset ofNpoints of the form

```
(x⃗ 1 ;y 1 );(⃗x 2 ;y 2 );:::;(⃗xN;yN):
```
where theyi’s are either+ 1 or 1 , find vectorsw⃗and⃗and a numberbwhich

```
minimize
```
#### 1

#### 2

```
Yw⃗Y^2 +C
```
```
N
Q
i= 1
```
```
i
```

#### CHAPTER 10. SUPPORT VECTOR MACHINES 155

```
Figure 10.15: Soft margin hyperplanes
```
```
subject to yi(w⃗⋅⃗xi−b)≥ 1 −i;fori= 1 ;:::N
i≥ 0 ;fori= 1 ;:::;N
```
Remarks

1. There are algorithms for solving the reformulated SVM problem given above. The details of
    these algorithms are beyond the scope of these notes.
2. The hyperplanes given by the equations

```
w⃗⋅x⃗i−b=+ 1 and w⃗⋅⃗xi−b=− 1
```
```
with the values ofw⃗andbobtained as solutions of the reformulated problem, are called the
soft margin hyperplanesfor the SVM problem.
```
### 10.10 Kernel functions

In the context of SVM’s, a kernel function is a function of the formK(x;⃗y⃗), wherex⃗andy⃗are
n-dimensional vectors, having a special property. These functions are used to obtain SVM-like
classifiers for two-class datasets which are not linearly separable.

10.10.1 Definition

Letx⃗andy⃗be arbitrary vectors in then-dimensional vector spaceRn. Letbe a mapping fromRn
to some vector space. A functionK(x;⃗y⃗)is called a kernel function if there is a functionsuch
thatK(⃗x;⃗y)=(x⃗) **⋅** (y⃗).

10.10.2 Examples

Example 1

Let

```
x⃗=(x 1 ;x 2 )∈R^2
y⃗=(y 1 ;y 2 )∈R^2
```
We define
K(⃗x;⃗y)=(x⃗ **⋅** y⃗)^2 :


#### CHAPTER 10. SUPPORT VECTOR MACHINES 156

We show that this is a kernel function. To do this, we note that

```
K(⃗x;⃗y)=(x⃗ ⋅ y⃗)^2
=(x 1 y 1 +x 2 y 2 )^2
=x^21 y^21 + 2 x 1 y 1 x 2 y 2 +x^22 y 22
```
Now we define

```
(x⃗)=(x^21 ;
```
#### √

```
2 x 1 x 2 ;x^22 )∈R^3
(y⃗)=(y 12 ;
```
#### √

```
2 y 1 y 2 ;y^22 )∈R^3
```
Then we have

```
(⃗x) ⋅ (⃗y)=x^21 y^21 +(
```
#### √

```
2 x 1 x 2 )(
```
#### √

```
2 y 1 y 2 )+x^22 y 22
=x^21 y^21 + 2 x 1 x 2 y 1 y 2 +x^22 y 22
=K(x;⃗y⃗)
```
This shows thatK(x;⃗y⃗)is indeed a kernel function.

Example 2

Let

```
x⃗=(x 1 ;x 2 )∈R^2
y⃗=(y 1 ;y 2 )∈R^2
```
We define
K(x;⃗y⃗)=(⃗x **⋅** ⃗y+)^2 :

We show that this is a kernel function. To do this, we note that

```
K(x;⃗y⃗)=(x⃗ ⋅ y⃗+)^2
=(x 1 y 1 +x 2 y 2 +)^2
=(x⃗) ⋅ (y⃗)
```
where
(x⃗)=(x^21 ;x^22 ;

#### √

```
2 x 1 x 2 ;
```
#### √

```
2 x 1 ;
```
#### √

```
2 x 2 ;
```
#### √

#### )∈R^6 :

This shows thatK(x;⃗y⃗)is indeed a kernel function.

10.10.3 Some important kernel functions

In the following we assume thatx⃗=(x 1 ;x 2 ;:::;xn)and⃗y=(y 1 ;y 2 ;:::;yn).

1. Homogeneous polynomial kernel

```
K(x;⃗y⃗)=(x⃗ ⋅ y⃗)d
wheredis some positive integer.
```
2. Non-homogeneous polynomial kernel

```
K(x;⃗y⃗)=(x⃗ ⋅ ⃗y+)d
wheredis some positive integer andis a real constant.
```

#### CHAPTER 10. SUPPORT VECTOR MACHINES 157

3. Radial basis function (RBF) kernel

```
K(x;⃗y⃗)=e−Y⃗x−⃗yY
```
(^2) ~ 2  2
This is also called the Gaussian radial function kernel.^1

4. Laplacian kernel function

```
K(x;⃗y⃗)=e−Yx⃗−⃗yY~
```
5. Hyperbolic tangent kernel function (Sigmoid kernel function)

```
K(x;⃗y⃗)=tanh( (x⃗ ⋅ y⃗)+c)
```
### 10.11 The kernel method (kernel trick)

10.11.1 Outline

1. Choose an appropriate kernel functionK(x;⃗y⃗).
2. Formulate and solve the optimization problem obtained by replacing each inner productx⃗ **⋅** y⃗
    byK(⃗x;⃗y)in the SVM optimization problem.
3. In the formulation of the classifier function for the SVM problem using the inner products of
    unclassified data⃗zand input vectorsx⃗i, replace each inner productz⃗ **⋅** x⃗iwithK(⃗z;x⃗i)to
    obtain the new classifier function.

10.11.2 Algorithm

Algorithm of the kernel method

Given a two-class linearly separable dataset ofNpoints of the form

```
(x⃗ 1 ;y 1 );(⃗x 2 ;y 2 );:::;(⃗xN;yN);
```
where theyi’s are either+ 1 or 1 and appropriate kernel functionK(x;⃗y⃗):

Step1. Find ⃗=( 1 ; 2 ;:::;N)which maximizes

```
N
Q
i= 1
```
(^) i−

#### 1

#### 2

```
N
Q
i= 1 ;j= 1
```
(^) i (^) jyiyjK(⃗xi;x⃗j)
subject to
N
Q
i= 1
(^) iyi= 0
(^) i> 0 fori= 1 ; 2 ;:::;N:
Step2. Computew⃗=∑Ni= 1 iyix⃗i.
Step3. Computeb=^12 mini∶yi=+ 1 K(w;⃗x⃗i)+maxi∶yi=− 1 K(w;⃗⃗xi).
Step4. The SVM classifier function is given byf(z⃗)=∑Ni= 1 iyiK(⃗xi;z⃗)+b:
(^1) To represent this kernel as an inner product, we need mapfromRninto an infinite-dimensional vector space. A
discussion of these ideas is beyond the scope of these notes.


#### CHAPTER 10. SUPPORT VECTOR MACHINES 158

### 10.12 Multiclass SVM’s

In machine learning, themulticlass classificationis the problem of classifying instances into one of
three or more classes. Classifying instances into one of the two classes is called binary classification.
Support vector machines can be constructed only when the dataset has only two class-labels and
is linearly separable. We have already discussed a method to extend the concept of SVM’s to the
case where the dataset is not linearly separable. In this section we consider how the SVM’s can be
used to obtain classifiers when there are more than two class labels. Two methods are generally used
to handle such cases known by the names ”One-against-all" and “one-against-one”.

10.12.1 “One-against-all” method

TheOne-Against-All(OAA) SVMs were first introduced by Vladimir Vapnik in 1995.

```
Figure 10.16: One-against all
```
Let there bepclass labels, say,c 1 ;c 2 ;:::;cp. We construct the followingptwo-class datasets
and obtain the corresponding SVM classifiers. First, we assign the class labels+ 1 to all instances
having class labelc 1 and the class label− 1 to all the remaining instances in the data set. Letf 1 (⃗x)
be the SVM classifier function for the resulting two-class dataset. Next, we assign the class labels
+ 1 to all instances having class labelc 2 and the class label− 1 to all the remaining instances in the
data set. Letf 2 (x⃗)be the SVM classifier function for the resulting two-class dataset. We continue
like this and generate SVM classifier functionsf 3 (x⃗),:::,fp(⃗x)
Two criteria have been developed to assign a class label to a test instancez⃗.

1. A data point⃗zwould be classified under a certain class if and only if that class’s SVM accepted
    it and all other classes’ SVMs rejected it. Thus⃗zwill be assignedciiffi(⃗z)> 0 andfj(⃗z)< 0
    for allj≠i.
2. ⃗zis the assigned the class labelciiffi(⃗z)has the highest value amongf 1 (z⃗);:::;fp(⃗z),
    regardless of sign.

Figure 10.16 illustrates the one-against-all method with three classes.

10.12.2 “One-against-one” method

In theone-against-one(OAO) (also calledone-vs-one(OVO)) strategy, a SVM classifier is con-
structed for each pair of classes. If there arepdifferent class labels, a total ofp(p− 1 )~ 2 classifiers
are constructed. An unknown instance is classified with the class getting the most votes. Ties are
broken arbitrarily.


#### CHAPTER 10. SUPPORT VECTOR MACHINES 159

```
Figure 10.17: One-against-one
```
For example, let there be three classes,A,BandC. In the OVO method we construct 3 ( 3 −
1 )~ 2 = 3 SVM binary classifiers. Now, if⃗zis to be classified, we apply each of the three classifiers
to⃗z. Let the three classifiers assign the classesA,B,Brespectively to⃗z. Since a label toz⃗is
assigned by the majority voting, in this example, we assign the class label ofBtoz⃗.
One-vs-one (OVO) strategy is not a particular feature of SVM. Indeed, OVO can be applied to
any binary classifier to solve multi-class classification problem.

### 10.13 Sample questions

(a) Short answer questions

1. Define an hyperplane in ann-dimensional space. What are the hyperplanes in 2 -dimensional
    and 3 -dimensional spaces?
2. Find the distance of the point( 1 ;− 2 ; 3 )from the hyperplane

```
3 x 1 − 4 x 2 + 12 x 3 − 1 = 0 :
```
3. What is a linearly separable dataset? Give an example. Give an example for a dataset which
    is not linearly separable.
4. What is meant by maximum margin hyperplane?
5. Define the support vector machine of a two-class dataset.
6. Define the support vectors of a two-class dataset.
7. What is a kernel function? Give an example.

(b) Long answer questions

1. State the mathematical formulation of the SVM problem. Give an outline of the method for
    solving the problem.
2. Explain the significance of soft margin hyperplanes and explain how they are computed.
3. Show that the function
    K(⃗x;⃗y)=(x⃗ **⋅** y⃗)^3
    is a kernel function.
4. What is meant by kernel trick in context of support vector machines? How is it used to find a
    SVM classifier.


#### CHAPTER 10. SUPPORT VECTOR MACHINES 160

5. Given the following dataset, using elementary geometry find the maximum margin hyperplane
    for the data. Verify the result by finding the same using the SVM algorithm.

```
Example x 1 x 2 Class label
1 2 1 − 1
2 4 5 + 1
3 3 6 + 1
```

Chapter 11

## Hidden Markov models

This chapter contains a brief introduction to hidden Markov models (HMM’s). The HMM is one
of the most important machine learning models in speech and language processing. To define it
properly, we need to first understand the concept of discrete Markov processes. So, we begin the
chapter with a description of Markov processes and then discuss HMM’s. The three basic problems
associated with a HMM are stated, but algorithms for their solutions are not given as they are beyond
the scope of these notes.

### 11.1 Discrete Markov processes: Examples

11.1.1 Example 1

Through this example we introduce the various elements that constitute a discrete homogeneous
Markov process.

1. System and states
    Let us consider a highly simplified model of the different states a stock-market is in, in a given
    week. We assume that there are only three possible states:

```
S 1 : Bull market trend
S 2 : Bear market trend
S 3 : Stagnant market trend
```
2. Transition probabilities
    Week after week, the stock-market moves from one state to another state. From previous data,
    it has been estimated that there are certain probabilities associated with these movements.
    These probabilities are called transition probabilities.
3. Markov assumption
    We assume that the following statement (called Markov assumption or Markov property) re-
    garding transition probabilities is true:
       - Let the weeks be counted as 1 ; 2 ;:::and let an arbitrary week be thet-th week. Then,
          the state in weekt+ 1 depends only on the state in weekt, regardless of the states in
          the previous weeks. This corresponds to saying that, given the present state, the future
          is independent of the past.
4. Homogeneity assumption
    To simplify the computations, we assume that the following property, called the homogeneity
    assumption, is also true.

#### 161


#### CHAPTER 11. HIDDEN MARKOV MODELS 162

- The probability that the stock market is in a particular state in a particular weekt+ 1
    given that it is in a particular state in weekt, is independent oft.
5. Representation of transition probabilitiesLet the probability that a bull week is followed
by another bull week be 90%, a bear week be 7.5%, and a stagnant week be 2.5%. Similarly,
let the probability that a bear week is followed by another bull week be 15%, bear week be
80% and a stagnant week be 5%. Finally, let the probability that a stagnant week be followed
by a bull week is 25%, a bear week be 25% and a stagnant week be 50%. The transition
probabilities can be represented in two ways:

```
(a) The states and the state transition probabilities can be represented diagrammatically as
in Figure 11.1.
```
```
Figure 11.1: A state diagram showing state transition probabilities
```
```
(b) The state transition probabilities can also be represented by a matrix called thestate
transition matrix. Let us label the states as “1=bull”, “2=bear” and “3=stagnant” and
consider the matrix
```
```
P=
```
#### ⎡⎢

#### ⎢⎢

#### ⎢⎢

#### ⎣

#### 0 :90 0:075 0: 025

#### 0 : 15 0 : 80 0 : 05

#### 0 : 25 0 : 50 0 : 25

#### ⎤⎥

#### ⎥⎥

#### ⎥⎥

#### ⎦

```
In this matrix, the element in thei-th row,j-th column represents the probability that the
market in stateiis followed by market in statej.
Note that in the state transition matrixP, the sum of the elements in every row is 1.
```
6. Initial probabilities
    The initial probabilities are the probabilities that the stock-market is in a particular state ini-
    tially. These are denoted by 1 ; 2 ; 3 : 1 is the probability that the stock-market is in bull
    state initially; similarly, 2 and 3. the values of these probabilities can be presented as a
    vector:

```
=
```
#### ⎡

#### ⎢⎢

#### ⎢⎢

#### ⎢

#### ⎣

####  1

####  2

####  3

#### ⎤

#### ⎥⎥

#### ⎥⎥

#### ⎥

#### ⎦

#### =

#### ⎡

#### ⎢⎢

#### ⎢⎢

#### ⎢

#### ⎣

#### 0 : 5

#### 0 : 3

#### 0 : 2

#### ⎤

#### ⎥⎥

#### ⎥⎥

#### ⎥

#### ⎦

7. The discrete Markov process
    The functioning of the stock-markets with the three statesS 1 ;S 2 ;S 3 with the assumption that
    the Markov property is true, the transition probabilities given by the matrixPand the initial


#### CHAPTER 11. HIDDEN MARKOV MODELS 163

```
probabilities given by the vectorconstitute a discrete Markov process. Since we also assume
the homogeneity property for the transition probabilities is true, it is a homogeneous discrete
Markov process.
```
Probabilities for future states

Consider the matrix:

#### TP=^0 :5 0:3 0:^2 

#### ⎡⎢

#### ⎢

#### ⎢⎢

#### ⎢⎣

#### 0 :90 0:075 0: 025

#### 0 : 15 0 : 80 0 : 05

#### 0 : 25 0 : 50 0 : 25

#### ⎤⎥

#### ⎥

#### ⎥⎥

#### ⎥⎦

#### = 0 :5450 0:3775 0: 0775 

The elements in this row vector represent the probabilities that the stock-market is in the bull state,
the bear state and the stagnant state respectively in the second week.
In general, the elements of the row vectorTPnrepresent the probabilities that the stock-market
is in the bull state, the bear state and the stagnant state respectively in the(n+ 1 )-th week.

11.1.2 Example 2

Consider a simplified model of weather. We assume that the weather conditions are observed once
a day at noon and it is recorded as in one of the following states:

```
S 1 : Rainy
S 2 : Cloudy
S 3 : Sunny
```
Assuming that the Markov property and the homogeneity property are true, we can write the state
transition probability matrixP. Let the matrix be

#### P=

#### ⎡⎢

#### ⎢⎢

#### ⎢⎢

#### ⎣

#### 0 :4 0:3 0: 3

#### 0 :2 0:6 0: 2

#### 0 :1 0:1 0: 8

#### ⎤⎥

#### ⎥⎥

#### ⎥⎥

#### ⎦

Let the initial probabilities be
= 0 :25 0:25 0: 50 

The changes in weather with the three satesS 1 ;S 2 ;S 3 satisfying the Markov property and the ho-
mogeneity property, the transition probability matrixPand the initial probabilities given bycon-
stitute a discrete homogeneous Markov process.

### 11.2 Discrete Markov processes: General case

A Markov process is a random process indexed by time, and with the property that the future is
independent of the past, given the present. The time space may be discrete taking the values 1 ; 2 ;:::
or continuous taking any nonnegative real number as a value. In these notes, we consider only
discrete time Markov processes.

1. System and states
    Consider a system that at any time is in one ofNdistinct states:

```
S 1 ;S 2 ;:::;SN
```
```
We denote the state at timetbyqtfort= 1 ; 2 ;:::. So,qt=Simeans that the system is in
stateSiat timet.
```

#### CHAPTER 11. HIDDEN MARKOV MODELS 164

2. Transition probabilities
    At regularly spaced discrete times, the system moves to a new state with a given probability,
    depending on the values of the previous states. These probabilities are called the transition
    probabilities.
3. Markov assumptions (Markov property)
    We assume the following called the Markov assumption or the Markov property:
       - The state at timet+ 1 depends only on state at timet, regardless of the states in the
          previous times. This corresponds to saying that, given the present state, the future is
          independent of the past.
4. Homogeneity property
    We assume that the following property, called the homogeneity property, is true.
       - We also assume that these transition probabilities are independent of time, that is, the
          probabilitiesP(qt+ 1 =SjSqt=Si)are constants and do not depend ont. We denote this
          probablity byaij:
             aij=P(qt+ 1 =SjSqt=Si):
          We immediately note that

```
aij≥ 0 and
```
```
N
Q
j= 1
```
```
aij= 1 for alli:
```
5. Representation of transition probabilities
    The transition probabilities can be represented in two ways:

```
(a) If the number of states is small, the state transition probabilities can be represented
diagrammatically as in Figure 11.1.
(b) The state transition probabilities can also be represented by a matrix called thestate
transition matrix.
```
#### A=

#### ⎡⎢

#### ⎢⎢

#### ⎢⎢

#### ⎢⎢

#### ⎢⎣

```
a 11 a 12 ::: a 1 N
a 21 a 22 ::: a 2 N
⋯
aN 1 aN 2 ::: aNN
```
#### ⎤⎥

#### ⎥⎥

#### ⎥⎥

#### ⎥⎥

#### ⎥⎦

```
In this matrix, the element in thei-th row,j-th column represents the probability that the
system in stateSimoves to stateSj. Note that in the state transition matrixA, the sum
of the elements in every row is 1.
```
6. Initial probabilities
    We define the initial probabilitiesiwhich is the probability that the first state in the sequence
    isSi:
       =P(q 1 =Si):
    We also write

#### =

#### ⎡

#### ⎢⎢

#### ⎢⎢

#### ⎢⎢

#### ⎢⎢

#### ⎣

####  1

####  2

#### ⋯

#### N

#### ⎤

#### ⎥⎥

#### ⎥⎥

#### ⎥⎥

#### ⎥⎥

#### ⎦

```
We must have
N
Q
i= 1
```
```
i= 1 :
```

#### CHAPTER 11. HIDDEN MARKOV MODELS 165

7. Discrete Markov process
    A system with the statesS 1 ;S 2 ;:::;SNsatisfying the Markov property is called a discrete
    Markov process. If it satisfies the homogeneity property, then it is called a homogeneous
    discrete Markov process.

11.2.1 Probability for an observation sequence

Observable Markov model

The discrete Markov process described in Section 11.2 is also called anobservable Markov modelor
observable discrete Markov process. It is so called because the state of the system at any timetcan
be directly observed. This is in contrast to models where the state of the system cannot be directly
observed. If the state of the system cannot be directly observed the system is called ahidden Markov
model.Such systems are considered in Section??.

Probability for an observation sequence

In an observable Markov model, the states are observable. At any timetwe knowqt, and as the
system moves from one state to another, we get an observation sequence that is a sequence of states.
The output of the process is the set of states at each instant of time where each state corresponds to
a physical observable event.
LetObe an arbitrary observation sequence of lengthT. Let us consider a particular observation
sequence
Q=(q 1 ;q 2 ;:::;qT):

Now, given the transition matrixAand the initial probabilitieswe can calculate the probability
P(O=Q)as follows.

```
P(O=Q)=P(q 1 )P(q 2 Sq 1 )P(q 3 Sq 2 ):::P(qTSqT− 1 )
=q 1 aq 1 q 2 aq 2 q 3 :::aqT− 1 qT
```
Here,q 1 is the probability that the first state isq 1 ,aq 1 q 2 is the probability of going fromq 1 toq 2 ,
and so on. We multiply these probabilities to get the probability of the whole sequence.

Example

Consider the discrete Markov process described in Section 11.1.1. Let us compute the probability
of having a bull week followed by a stagnant week followed by two bear weeks. In this case the
observation sequence is

```
Q=(bull;stagnant;bear;bear)
=(S 1 ;S 2 ;S 3 ;S 3 )
```
The required probability is

```
P(O=Q)=P(S 1 )P(S 2 SS 1 )P(S 3 SS 2 )P(S 3 SS 3 )
= 1 a 12 a 23 a 33
= 0 : 5 × 0 : 075 × 0 : 05 × 0 : 25
= 0 : 00046875
```
11.2.2 Learning the parameters

Consider a homogeneous discrete Markov process with transition matrixAand initial probability
vector.Aandare the parameters of the process. The following procedure may be applied to
learn these parameters.


#### CHAPTER 11. HIDDEN MARKOV MODELS 166

Step 1. ObtainKobservation sequences each of lengthT. Letqtkbe the observed state at timet
in thek-th observation sequence.

Step 2. Let^ibe the estimate of the initial probabilityi. Then

```
^i=
number of sequences starting withSi
total number of sequences
```
#### :

Step 3. Let^aijbe the estimate ofaij. Then

```
^aij=
```
```
number of transitions fromSitoSj
number of transitions fromSi
```
Example

Let there be a discrete Markov process with three statesS 1 ,S 2 andS 3. Suppose we have the
following 10 observation sequences each of length 5 :

```
O 1 ∶ S 1 S 2 S 1 S 1 S 1
O 2 ∶ S 2 S 1 S 1 S 3 S 1
O 3 ∶ S 3 S 1 S 3 S 2 S 2
O 4 ∶ S 1 S 3 S 3 S 1 S 1
O 5 ∶ S 3 S 2 S 1 S 1 S 3
O 6 ∶ S 3 S 1 S 1 S 2 S 1
O 7 ∶ S 1 S 1 S 2 S 3 S 2
O 8 ∶ S 2 S 3 S 1 S 2 S 2
O 9 ∶ S 3 S 2 S 1 S 1 S 2
O 10 ∶ S 1 S 2 S 2 S 1 S 1
```
We have:

```
^ 1 =
```
```
number of sequences starting withS 1
total number of sequences
```
#### =

#### 4

#### 10

#### ^ 2 =

```
number of sequences starting withS 2
total number of sequences
```
#### =

#### 2

#### 10

#### ^ 3 =

```
number of sequences starting withS 3
total number of sequences
```
#### =

#### 4

#### 10

Therefor

```
=
```
#### ⎡⎢

#### ⎢⎢

#### ⎢⎢

#### ⎣

#### 4 ~ 10

#### 2 ~ 10

#### 4 ~ 10

#### ⎤⎥

#### ⎥⎥

#### ⎥⎥

#### ⎦

We illustrate the computation ofaij’s with an example.

```
^a 21 =
```
```
number of transitions fromS 2 toS 1
number of transitions fromS 2
```
#### =

#### 6

#### 11

```
^a 22 =
```
```
number of transitions fromS 2 toS 2
number of transitions fromS 2
```
#### =

#### 3

#### 11

```
^a 23 =
```
```
number of transitions fromS 2 toS 3
number of transitions fromS 2
```
#### =

#### 2

#### 11

The remaining transition probabilities can be estimated in a similar way.

#### A^=

#### ⎡⎢

#### ⎢⎢

#### ⎢⎢

#### ⎣

#### 9 ~19 6~19 4~ 19

#### 6 ~11 3~11 2~ 11

#### 5 ~10 4~10 1~ 10

#### ⎤⎥

#### ⎥⎥

#### ⎥⎥

#### ⎦

#### :


#### CHAPTER 11. HIDDEN MARKOV MODELS 167

```
Figure 11.2: A two-coin model of an HMM
```
### 11.3 Hidden Markov models

11.3.1 Coin tossing example

Let us consider the following scenario:
Consider a room which is divided into two parts by a curtain through which we cannot see what
is happening on the other half of the room. Person A is sitting in one half and person B is sitting
in the other half. Person B is doing some coin tossing experiment, but she will not tell person A
anything about what she is doing. Person B will only announce the result of each coin flip. Let a
typical sequence of announcements be

```
O=O 1 O 2 ::: OT
=H H T H H T T T ::: H (say)
```
where as usualHstands for heads andTstands for tails. Person A wants to create a mathematical
model which explains this sequence of observation. Person A suspects that person B is announcing
the results based on the outcomes of some discrete Markov process. If that is true, then the Markov
process that is happening behind the curtain is hidden from the rest of the world and we are left with
ahidden Markov process. To verify whether actually a Markov process is happening is a daunting
task. Based on the observations likeOalone, we have to decide on the following:

- A Markov process has different states. What should the states in the process correspond to
    what is happening behind the curtain?
- How many states should be there?
- What should be the initial probabilities?
- What should be the transition probabilities?

Let us assume that person B is doing something like the following before announcing the outcomes.

1. Let person B be in possession of two biased coins (or, three coins, or any number of coins)
    and she is flipping these coins in some order. When flipping a particular coin, the system is
    in the state of that coin. So, each of these coins may be identified as a state and there are two
    states, sayS 1 andS 2.
2. The outcomes of the flips of the coins are the observations. These observations are represented
    by the observation symbols “H” (for “head”) and “T” (for “tail”).


#### CHAPTER 11. HIDDEN MARKOV MODELS 168

3. After flipping coin, one of the two coins should be flipped next. There must be some definite
    procedure for doing this. The procedure is some random process with definite probabilities
    for selecting the coins. These are the transition probabilities and they define the transition
    probability matrixA.
4. Since the coins are biased, there would be definite probabilities for getting “H” or “T” each
    time the coin is flipped. These probabilities are called the observation probabilities.
5. There must be some procedure for selecting the first coin. This is specified by the initial
    probabilities vector.

11.3.2 The urn and ball model

Again, consider a room which is divided into two parts by a curtain through which we cannot see
what is happening on the other half of the room. Person A is sitting in one half and person B is
sitting in the other half. Person B is doing some experiment, but she will not tell person A anything
about what she is doing. Person B will only announce the result of each experiment. Let a typical
sequence of announcements be

```
O=O 1 O 2 ::: OT
=“red”, “green”, “red”,... , “blue”
```
Person A wants to create a mathematical model which explains this sequence of observations.

Figure 11.3: AnN-state urn and ball model which illustrates the general case of a discrete symbol
HMM

Person A suspects that person B is announcing the results based on the outcomes of some discrete
Markov process. If that is true, then the Markov process that is happening behind the curtain is
hidden from the rest of the world and we are left with ahidden Markov process.
In this example, let us assume that person A suspects that something like the following is hap-
pening behind the curtain.
There areNlarge urns behind the curtain. Within each urn there are large number of coloured
balls. There areMdistinct colours of balls. Person B, according to some random process, chooses
an initial urn. From this urn a ball is chosen at random and the colour of the ball is announced.
The ball is then replaced in the urn. A new urn is then selected according to some random selection
process associated with the current urn and the ball selection process is repeated.
This process is a typical example of a hidden Markov process. Note the following:

1. Selection of an urn may be made to correspond to a state of the process. Then, there areN
    states in the process.


#### CHAPTER 11. HIDDEN MARKOV MODELS 169

2. The colours of the balls selected are the observations. The name of the colour may be referred
    to as the “observation symbol”. Hence, there areMobservation symbols in the process.
3. The random selection process associated with the current urn specifies the transition probabil-
    ities.
4. Each urn contains a mixture of balls of different colours. So, corresponding to each urn, there
    are definite probabilities for getting balls of different colours. These probabilities are called
    the observation probabilities.
5. The procedure for selecting the first urn provides the initial probabilities.

11.3.3 Hidden Markov model (HMM): The general case

A hidden Markov model (HMM) is characterized by the following:

1. The number of states in the model, sayN. Let the states beS 1 ;S 2 ;:::;SN.
2. The number of distinct observation symbols, sayM. Let the observation symbols bev 1 ;v 2 ;:::;vM.
    (The observation symbols correspond to the physical outputs of the system.)
3. The state transition probabilities specified by anN×NmatrixA=[aj]:

```
aij=P(qt+ 1 =SjSqt=Si);fori;j= 1 ; 2 ;:::;N:
```
```
whereqtis the state at timet.
```
4. The observation symbol probability distributionsbj(k)forj= 1 ;:::;Nandk= 1 ;:::;M.
    bj(k)is the probability that, at timet, the outcome is the symbolvkgiven that the system is
    in stateSj:
       bj(k)=P(vkattSqt=Sj):
    We denote byBtheN×Mmatrix whose element in thej-th rowk-column isbj(k).
5. The initial probabilities=[i]:

```
=P(q 1 =Si);fori= 1 ; 2 ;:::;N:
```
The values ofNandMare implicitly defined inA,Band. So, a HMM is completely defined by
the parameter set
=(A;B;):

### 11.4 Three basic problems of HMMs

Given the general model of HMM, there are three basic problems that must be solved for the model
to be useful for real-world applications. These problems are the following:

Problem 1. Evaluation problem

```
Given the observation sequence
```
```
O=O 1 O 2 ::: OT;
```
```
and a HMM model
=(A;B;)
how do we efficiently compute
P(OS);
the probability of the observation sequenceOgiven the model?
```

#### CHAPTER 11. HIDDEN MARKOV MODELS 170

Problem 2. Finding state sequence problem

```
Given the observation sequence
```
```
O=O 1 O 2 ::: OT;
```
```
and a HMM model
=(A;B;)
how do we find the the state sequence
```
```
Q=q 1 q 2 :::;qT
```
```
which has the highest probability of generatingO; that is, how do we findQ⋆that
maximizes the probabilityP(QSO;)?
```
Problem 3. Learning model parameters problem

```
Given a training setXobservation sequences, how do we learn the model
```
```
=(A;B;)
```
```
that maximizes the probability of generatingX; that is, how do we find⋆that maxi-
mizes the probability
P(XS):
```
11.4.1 Solutions of the basic problems

The details of the algorithms for solving these problems are beyond the scope of these notes. Prob-
lem 1 is solved using the Forwards-Backwards algorithms. Problem 2 is solved by the Viterbi
algorithm and posterior decoding. Finally, Problem 3 is solved by the Baum-Welch algorithm.^1

### 11.5 HMM application: Isolated word recognition

Most speech-recognition systems are classified as isolated or continuous. Isolated word recognition
requires a brief pause between each spoken word, whereas continuous speech recognition does not.
Speech-recognition systems can be further classified as speaker-dependent or speaker-independent.
A speaker-dependent system only recognizes speech from one particular speaker’s voice, whereas a
speaker-independent system can recognize speech from anybody.
In this section, we consider in an outline form how HMMs are used in building anisolated word
recogniser.

1. Assume that we have a vocabularyVof words to be recognised.
2. For each word in the vocabulary, there is a training set ofKoccurrences of each spoken word
    (spoken by 1 or more talkers) where each occurrence of the word constitute an observation
    sequence.
3. The observations are some appropriate representation of the characteristics of the word. These
    representations are obtained via some preprocessing of the speech signal like linear predictive
    coding (LPC).
4. For each wordv∈V, we build an HMM, say

```
v=(Av;Bv;v):
```
```
For this, we have to apply the algorithms for learning an HMM to estimate the parameters
(Av;Bv;v)that maximise the probability of generating the observations in the training set
ofKoccurrences of the wordv.
```
(^1) For a concise presentation of the algorithms, visit [http://www.shokhirev.com/](http://www.shokhirev.com/)
nikolai/abc/alg/hmm/hmm.html.


#### CHAPTER 11. HIDDEN MARKOV MODELS 171

```
Figure 11.4: Block diagram of an isolated word HMM recogniser
```
5. Now consider an unknown wordvwhich needs to be recognised. The following procedure is
    used to recognise the word.

```
(a) The speech signal corresponding to the wordwis subjected to preprocessing like LPC
and converted to the representation used in building the HMMs and the measurement of
the observation sequenceO=O 1 O 2 ::: OTis obtained.
(b) The probabilitiesP(OSv), for each wordv∈Vare calculated.
(c) Choose the wordvfor whichP(OSv)is highest:
```
```
v⋆=arg max
v∈V
```
```
P(OSv):
```
```
(d) The wordwis recognised as the wordv⋆.
```
### 11.6 Sample questions

(a) Short answer questions

1. What is the state transition matrix of a discrete Markov process?
2. What is the Markov property of a discrete Markov process?
3. Consider a Markov process with two states “Rainy” and “Dry” and the transition probabilities
    as shown in the following diagram.


#### CHAPTER 11. HIDDEN MARKOV MODELS 172

```
0.3 Rainy Dry 0.8
```
#### 0.7

#### 0.2

```
IfP(Rain)= 0 : 4 andP(Dry)= 0 : 6 compute the probability for the sequence “Rain, Rain,
Dry, Dry”.
```
(b) Long answer questions

1. Describe a discrete Markov process with an example.
2. Describe a hidden Markov model.
3. Explain how hidden Markov models are used in speech recognition.
4. What are the basic problems associated with a hidden Markov model.
5. Describe the urn and ball model of a hidden Markov model.
6. Describe the coin tossing model of a hidden Markov model.
7. Let there be a discrete Markov process with two statesS 1 andS 2. Given the following se-
    quences of observations of these states, estimate the initial probabilities and the transition
    probabilities of the process.

```
S 1 S 2 ; S 2 S 2 ; S 1 S 2 ; S 2 S 1 ; S 1 S 1 ; S 2 S 1 ; S 1 S 2 ; S 1 S 1 :
```

Chapter 12

## Combining multiple learners

In general there are several algorithms for learning the same task. Though these are generally suc-
cessful, no one single algorithm is always the most accurate. Now, we shall discuss models com-
posed of multiple learners that complement each other so that by combining them, we attain higher
accuracy.

### 12.1 Why combine many learners

There are several reasons why a single learner may not produce accurate results.

- Each learning algorithm carries with it a set of assumptions. This leads to error if the assump-
    tions do not hold. We cannot be fully sure whether the assumptions are true in a particular
    situation.
- Learning is an ill-posed problem. With finite data, each algorithm may converge to a different
    solution and may fail in certain circumstances.
- The performance of a learner may be fine-tuned to get the highest possible accuracy on a
    validation set. But this fine-tuning is a complex task and still there are instances on which
    even the best learner is not accurate enough.
- It has been proved that there is no single learning algorithm that always produces the most
    accurate output.

### 12.2 Ways to achieve diversity

When many learning algorithms are combined, the individual algorithms in the collection are called
thebase learnersof the collection.
When we generate multiple base-learners, we want them to be reasonably accurate but do not
require them to be very accurate individually. The base-learners are not chosen for their accuracy,
but for their simplicity. What we care for is the final accuracy when the base- learners are combined,
rather than the accuracies of the bas-learners we started from.
There are several different ways for selecting the base learners.

1. Use different learning algorithms
    There may be several learning algorithms for performing a given task. For example, for
    classification, one may choose the naive Bayes’ algorithm, or the decision tree algorithm or
    even the SVM algorithm.
    Different algorithms make different assumptions about the data and lead to different results.
    When we decide on a single algorithm, we give emphasis to a single method and ignore all
    others. Combining multiple learners based on multiple algorithms, we get better results.

#### 173


#### CHAPTER 12. COMBINING MULTIPLE LEARNERS 174

2. Use the same algorithm with different hyperparameters
    In machine learning, ahyperparameteris a parameter whose value is set before the learning
    process begins. By contrast, the values of other parameters are derived via training.
    The number of layers, the number of nodes in each layer and the initial weights are all hyper-
    parameters in an artificial neural network. When we train multiple base-learners with different
    hyperparameter values, we average over it and reduce variance, and therefore error.
3. Use different representations of the input object
    For example, in speech recognition, to recognize the uttered words, words may be represented
    by the acoustic input. Words can also be represented by video images of the speaker’s lips as
    the words are spoken.
    Different representations make different characteristics explicit allowing better identification.
    In many applications, there are multiple sources of information, and it is desirable to use all
    of these data to extract more information and achieve higher accuracy in prediction. We make
    separate predictions based on different sources using separate base-learners, then combine
    their predictions.
4. Use different training sets to train different base-learners
    - This can be done by drawing random training sets from the given sample; this is calledbagging.
    - The learners can be trained serially so that instances on which the preceding base-
       learners are not accurate are given more emphasis in training later base-learners; ex-
       amples areboostingandcascading.
    - The partitioning of the training sample can also be done based on locality in the input
       space so that each base-learner is trained on instances in a certain local part of the input
       space.
5. Multiexpert combination methods
    These base learners work in parallel. All of them are trained and then given an instance,
    they all give their decisions, and a separate combiner computes the final decision using their
    predictions. Examples include voting and its variants.
6. Multistage combination methods
    These methods use a serial approach where the next base-learner is trained with or tested on
    only the instances where the previous base-learners are not accurate enough.

### 12.3 Model combination schemes

12.3.1 Voting

This is the simplest procedure for combining the outcomes of several learning algorithms. Let us
examine some special cases of this scheme

1. Binary classification problem
    Consider a binary classification problem with class labels− 1 and+ 1. Let there beLbase
    learners and letxbe a test instance. Each of the base learners will assign a class label tox. If
    the class label assigned is+ 1 , we say that the learner votes for+ 1 and that the label+ 1 gets
    a vote. The number of votes obtained by the class labels when the different base learners are
    applied is counted. In the voting scheme for combining the learners, the label which gets the
    majority votes is assigned tox.


#### CHAPTER 12. COMBINING MULTIPLE LEARNERS 175

2. Multi-class classification problem
    Let there benclass labelsC 1 ;C 2 ;:::;Cn. Letxbe a test instance and let there beLbase
    learners. Here also, each of the base learners will assign a class label toxand when a class
    label is assigned a label, the label gets a vote. In the voting scheme, the class label which gets
    the maximum number of votes is assigned tox.
3. Regression
    ConsiderLbase learners for predicting the value of a variabley. Lety^ibe the output predicted
    by thei-th base learner. The final output is computed as

```
y=wiy^ 1 +w 2 y^ 2 + ⋯ +wLy^L
```
```
wherew 1 ;w 2 ;:::;wLare called the weights attached to the outputs of the various base learn-
ers and they must satisfy the following conditions:
```
```
wj≥ 0 forj= 1 ; 2 ;:::;L
w 1 +w 2 + ⋯ +wL= 1 :
```
```
This is theweighted voting scheme. Insimple voting, we take
```
```
wi=
```
#### 1

#### L

```
forj= 1 ; 2 ;:::;L:
```
12.3.2 Bagging

Bagging is a voting method whereby base-learners are made different by training them over slightly
different training sets.
GeneratingLslightly different samples from a given sample is done by bootstrap, where given a
training setXof sizeN, we drawNinstances randomly fromXwith replacement (see Section??).
Because sampling is done with replacement, it is possible that some instances are drawn more than
once and that certain instances are not drawn at all. When this is done to generateLsamplesXj,
j= 1 ;:::;L, these samples are similar because they are all drawn from the same original sample,
but they are also slightly different due to chance.
The base-learners are trained with theseLsamplesXj. A learning algorithm is anunstable
algorithmif small changes in the training set causes a large difference in the generated learner.
Bagging, short for bootstrap aggregating, uses bootstrap to generateLtraining sets, trainsLbase-
learners using an unstable learning procedure and then during testing, takes an average. Bagging
can be used both for classification and regression. In the case of regression, to be more robust, one
can take the median instead of the average when combining predictions.
Algorithms such as decision trees and multilayer perceptrons are unstable.

12.3.3 Boosting

In bagging, generating complementary base-learners is left to chance and to the unstability of the
learning method. In boosting, we actively try to generate complementary base-learners by training
the next learner on the mistakes of the previous learners. The original boosting algorithm combines
three weak learners to generate a strong learner. A weak learner has error probability less than
1/2, which makes it better than random guessing on a two-class problem, and a strong learner has
arbitrarily small error probability.

The boosting method

1. Letd 1 ;d 2 ;d 3 be three learning algorithms for a particular task. Let a large training setXbe
    given.
2. We randomly divideXinto three sets, sayX 1 ;X 2 ;X 3.


#### CHAPTER 12. COMBINING MULTIPLE LEARNERS 176

3. We useX 1 and traind 1.
4. We then takeX 2 and feed it tod 1.
5. We take all instances misclassified byd 1 and also as many instances on whichd 1 is correct
    fromX 2 , and these together form the training set ofd 2.
6. We then takeX 3 and feed it tod 1 andd 2.
7. The instances on whichd 1 andd 2 disagree form the training set ofd 3.
8. During testing, given an instance, we give it tod 1 andd 2 if they agree, that is the response;
    otherwise the response ofd 3 is taken as the output.

It has been shown that this overall system has reduced error rate, and the error rate can arbitrar-
ily be reduced by using such systems recursively. One disadvantage of the system is thaaaaaat it
requires a very large training sample. An improved algorithm known as AdaBoost (short for “adap-
tive boosting”), uses the same training set over and over and thus need not be large. AdaBoost can
also combine an arbitrary number of base-learners, not three.

### 12.4 Ensemble learning⋆.

The word “ensemble” literally means “a group of things or people acting or taken together as a
whole, especially a group of musicians who regularly play together.”
In machine learning, an ensemble learning method consists of the following two steps:

1. Create different models for solving a particular problem using a given data.
2. Combine the models created to produce improved results.

The different models may be chosen in many different ways:

- The models may be created using appropriate different algorithms likek-NN algorithm, Naive-
    Bayes algorithm, decision tree algorithm, etc.
- The models may be created by using the same algorithm but using different splits of the same
    dataset into training data and test data.
- The models may be created by assigning different initial values to the parameters in the algo-
    rithm as in ANN algorithms.

The models created in the ensemble learning methods are combined in several ways.

- Simple majority voting in classification problems: Every model makes a prediction (votes)
    for each test instance and the final output prediction is the one that receives more than half of
    the votes.
- Weighted majority voting in classification problem: In weighted voting we count the predic-
    tion of the better models multiple times. Finding a reasonable set of weights is up to us.
- Simple averaging in prediction problems: In simple averaging method, for every instance of
    test dataset, the average predictions are calculated.
- Weighted averaging in prediction problems: In this method, the prediction of each model is
    multiplied by the weight and then their average is calculated.

### 12.5 Random forest⋆

Arandom forestis an ensemble learning method where multiple decision trees are constructed and
then they are merged to get a more accurate prediction.


#### CHAPTER 12. COMBINING MULTIPLE LEARNERS 177

```
Figure 12.1: Example of random forest with majority voting
```
12.5.1 Algorithm

Here is an outline of the random forest algorithm.

1. The random forests algorithm generates many classification trees. Each tree is generated as
    follows:

```
(a) If the number of examples in the training set isN, take a sample ofNexamples at
random - but with replacement, from the original data. This sample will be the training
set for generating the tree.
(b) If there areMinput variables, a numbermis specified such that at each node,mvari-
ables are selected at random out of theMand the best split on thesemis used to split
the node. The value ofmis held constant during the generation of the various trees in
the forest.
(c) Each tree is grown to the largest extent possible.
```
2. To classify a new object from an input vector, put the input vector down each of the trees in
    the forest. Each tree gives a classification, and we say the tree “votes” for that class. The
    forest chooses the classification

12.5.2 Strengths and weaknesses

Strengths

The following are some of the important strengths of random forests.

- It runs efficiently on large data bases.
- It can handle thousands of input variables without variable deletion.
- It gives estimates of what variables are important in the classification.
- It has an effective method for estimating missing data and maintains accuracy when a large
    proportion of the data are missing.
- Generated forests can be saved for future use on other data.


#### CHAPTER 12. COMBINING MULTIPLE LEARNERS 178

- Prototypes are computed that give information about the relation between the variables and
    the classification.
- The capabilities of the above can be extended to unlabeled data, leading to unsupervised
    clustering, data views and outlier detection.
- It offers an experimental method for detecting variable interactions.
- Random forest run times are quite fast, and they are able to deal with unbalanced and missing
    data.
- They can handle binary features, categorical features, numerical features without any need for
    scaling.
- There are lots of excellent, free, and open-source implementations of the random forest algo-
    rithm. We can find a good implementation in almost all major ML libraries and toolkits.

Weaknesses

- A weakness of random forest algorithms is that when used for regression they cannot predict
    beyond the range in the training data, and that they may over-fit data sets that are particularly
    noisy.
- The sizes of the models created by random forests may be very large. It may take hundreds of
    megabytes of memory and may be slow to evaluate.
- Random forest models are black boxes that are very hard to interpret.

### 12.6 Sample questions

(a) Short answer questions

1. Explain the necessity of combining several algorithms for accomplishing a particular task.
2. What is a base learner? How do we select base learners?

(b) Long answer questions

1. Explain the following: (i) voting (ii) bagging (iii) boosting.
2. Explain what is meant by random forests.


Chapter 13

## Clustering methods

### 13.1 Clustering

Clusteringorcluster analysisis the task of grouping a set of objects in such a way that objects in the
same group (called a cluster) are more similar (in some sense) to each other than to those in other
groups (clusters).
Clustering is a main task of exploratory data mining and used in many fields, including machine
learning, pattern recognition, image analysis, information retrieval, bioinformatics, data compres-
sion, and computer graphics. It can be achieved by various algorithms that differ significantly in
their notion of what constitutes a cluster and how to efficiently find them. Popular notions of clus-
ters include groups with small distances between cluster members, dense areas of the data space,
etc.

13.1.1 Examples of data with natural clusters

In many applications, there will naturally be several groups or clusters in samples.

1. Consider the case of optical character recognition: There are two ways of writing the digit 7;
    the American writing is ‘7’, whereas the European writing style has a horizontal bar in the
    middle (something like 7−). In such a case, when the sample contains examples from both
    continents, the sample will contain two clusters or groups one corresponding to the American
    7 and the other corresponding to the European 7−.
2. In speech recognition, where the same word can be uttered in different ways, due to different
    pronunciation, accent, gender, age, and so forth, there is not a single, universal prototype. In
    a large sample of utterances of a specific word, All the different ways should be represented
    in the sample.

### 13.2 k-means clustering

13.2.1 Outline

Thek-means clustering algorithm is one of the simplest unsupervised learning algorithms for solving
the clustering problem.
Let it be required to classify a given data set into a certain number of clusters, say,kclusters.
We start by choosingkpoints arbitrarily as the “centres” of the clusters, one for each cluster. We
then associate each of the given data points with the nearest centre. We now take the averages of
the data points associated with a centre and replace the centre with the average, and this is done for
each of the centres. We repeat the process until the centres converge to some fixed points. The data
points nearest to the centres form the various clusters in the dataset. Each cluster is represented by
the associated centre.

#### 179


#### CHAPTER 13. CLUSTERING METHODS 180

13.2.2 Example

We illustrate the algorithm in the case where there are only two variables so that the data points
and cluster centres can be geometrically represented by points in a coordinate plane. The distance
between the points(x 1 ;x 2 )and(y 1 ;y 2 )will be calculated using the familiar distance formula of
elementary analytical geometry:

```
»
(x 1 −y 1 )^2 +(x 2 −y 2 )^2 :
```
Problem

Usek-means clustering algorithm to divide the following data into two clusters and also compute
the the representative data points for the clusters.

```
x 1 1 2 2 3 4 5
x 2 1 1 3 2 3 5
```
```
Table 13.1: Data fork-means algorithm example
```
Solution

```
x 1
```
```
x 2
```
#### 0 1 2 3 4 5

#### 1

#### 2

#### 3

#### 4

#### 5

```
Figure 13.1: Scatter diagram of data in Table 13.1
```
1. In the problem, the required number of clusters is 2 and we takek= 2.
2. We choose two points arbitrarily as the initial cluster centres. Let us choose arbitrarily (see
    Figure 13.2)
       ⃗v 1 =( 2 ; 1 ); ⃗v 2 =( 2 ; 3 ):
3. We compute the distances of the given data points from the cluster centers.


#### CHAPTER 13. CLUSTERING METHODS 181

```
x 1
```
```
x 2
```
#### 0 1 2 3 4 5

#### 1

#### 2

#### 3

#### 4

#### 5

```
v⃗ 1
```
```
v⃗ 2
```
```
Figure 13.2: Initial choice of cluster centres and the resulting clusters
```
```
⃗xi Data point Distance Distance Minimum Assigned
from⃗v 1 =( 2 ; 1 ) fromv⃗ 2 =( 2 ; 3 ) distance center
x⃗ 1 ( 1 ; 1 ) 1 2 : 24 1 ⃗v 1
x⃗ 2 ( 2 ; 1 ) 0 2 0 ⃗v 1
x⃗ 3 ( 2 ; 3 ) 2 0 0 ⃗v 2
x⃗ 4 ( 3 ; 2 ) 1 : 41 1 : 41 0 ⃗v 1
x⃗ 5 ( 4 ; 3 ) 2 : 82 2 2 ⃗v 2
x⃗ 6 ( 5 ; 5 ) 5 3 : 61 3 : 61 ⃗v 2
```
```
(The distances of⃗x 4 fromv⃗ 1 and⃗v 2 are equal. We have assigned⃗v 1 to⃗x 4 arbitrarily.)
This divides the data into two clusters as follows (see Figure 13.2):
Cluster 1:{⃗x 1 ;x⃗ 2 ;x⃗ 4 }represented by⃗v 1
Number of data points in Cluster 1:c 1 = 3.
Cluster 2 :{x⃗ 3 ;⃗x 5 ;x⃗ 6 }represented byv⃗ 2
Number of data points in Cluster 2:c 2 = 3.
```
4. The cluster centres are recalculated as follows:

```
⃗v 1 =
```
#### 1

```
c 1
```
```
(x⃗ 1 +x⃗ 2 +⃗x 4 )
```
#### =

#### 1

#### 3

```
(⃗x 1 +⃗x 2 +⃗x 4 )
```
```
=( 2 : 00 ; 1 : 33 )
```
```
⃗v 2 =
```
#### 1

```
c 2
```
```
(x⃗ 3 +x⃗ 5 +⃗x 6 )
```
#### =

#### 1

#### 3

```
(⃗x 3 +⃗x 5 +⃗x 6 )
```
```
=( 3 : 67 ; 3 : 67 )
```
5. We compute the distances of the given data points from the new cluster centers.


#### CHAPTER 13. CLUSTERING METHODS 182

```
⃗xi Data point Distance Distance Minimum Assigned
from⃗v 1 =( 2 ; 1 ) fromv⃗ 2 =( 2 ; 3 ) distance center
x⃗ 1 ( 1 ; 1 ) 1 : 05 3 : 77 1 : 05 ⃗v 1
x⃗ 2 ( 2 ; 1 ) 0 : 33 3 : 14 0 : 33 ⃗v 1
x⃗ 3 ( 2 ; 3 ) 1 : 67 1 : 80 1 : 67 ⃗v 1
x⃗ 4 ( 3 ; 2 ) 1 : 20 1 : 80 1 : 20 ⃗v 1
x⃗ 5 ( 4 ; 3 ) 2 : 60 0 : 75 0 : 75 ⃗v 2
x⃗ 6 ( 5 ; 5 ) 4 : 74 1 : 89 1 : 89 ⃗v 2
```
```
This divides the data into two clusters as follows (see Figure 13.4):
Cluster 1 :{x⃗ 1 ;⃗x 2 ;x⃗ 3 ;x⃗ 4 }represented by⃗v 1
Number of data points in Cluster 1:c 1 = 4.
Cluster 2 :{x⃗ 5 ;⃗x 6 }represented by⃗v 2
Number of data points in Cluster 1:c 2 = 2.
```
6. The cluster centres are recalculated as follows:

```
⃗v 1 ==
```
#### 1

```
c 1
```
```
(x⃗ 1 +x⃗ 2 +x⃗ 3 +⃗x 4 )
```
#### =

#### 1

#### 4

```
(x⃗ 1 +x⃗ 2 +x⃗ 3 +x⃗ 4 )
```
```
=( 2 : 00 ; 1 : 33 )
```
```
⃗v 2 =
```
#### 1

#### 2

```
(x⃗ 5 +x⃗ 6 )=( 3 : 67 ; 3 : 67 )
```
```
x 1
```
```
x 2
```
#### 0 1 2 3 4 5

#### 1

#### 2

#### 3

#### 4

#### 5

```
v⃗ 1
```
```
⃗v 2
```
```
Figure 13.3: Cluster centres after first iteration and the corresponding clusters
```
7. We compute the distances of the given data points from the new cluster centers.
    4.609772 3.905125 2.692582 2.500000 1.118034 1.118034


#### CHAPTER 13. CLUSTERING METHODS 183

```
⃗xi Data point Distance Distance Minimum Assigned
from⃗v 1 =( 2 ; 1 ) fromv⃗ 2 =( 2 ; 3 ) distance center
x⃗ 1 ( 1 ; 1 ) 1 : 25 4 : 61 1 : 25 ⃗v 1
x⃗ 2 ( 2 ; 1 ) 0 : 75 3 : 91 0 : 75 ⃗v 1
x⃗ 3 ( 2 ; 3 ) 1 : 25 2 : 69 1 : 25 ⃗v 1
x⃗ 4 ( 3 ; 2 ) 1 : 03 2 : 50 1 : 03 ⃗v 1
x⃗ 5 ( 4 ; 3 ) 2 : 36 1 : 12 1 : 12 ⃗v 2
x⃗ 6 ( 5 ; 5 ) 4 : 42 1 : 12 1 : 12 ⃗v 2
```
```
This divides the data into two clusters as follows (see Figure??):
Cluster 1 :{x⃗ 1 ;⃗x 2 ;x⃗ 3 ;x⃗ 4 }represented by⃗v 1
Number of data points in Cluster 1:c 1 = 4.
Cluster 2 :{x⃗ 5 ;⃗x 6 }represented by⃗v 2
Number of data points in Cluster 1:c 1 = 2.
```
8. The cluster centres are recalculated as follows:

```
⃗v 1 =
```
#### 1

```
c 1
```
```
(x⃗ 1 +x⃗ 2 +x⃗ 3 +⃗x 4 )
```
#### =

#### 1

#### 4

```
(x⃗ 1 +⃗x 2 +⃗x 3 +⃗x 4 )
```
```
=( 2 : 00 ; 1 : 75 )
```
```
⃗v 2 =
```
#### 1

```
c 2
```
```
(x⃗ 5 +x⃗ 6 )
```
#### =

#### 1

#### 2

```
(x⃗ 5 +⃗x 6 )
```
```
=( 4 : 00 ; 4 : 50 )
```
```
x 1
```
```
x 2
```
#### 0 1 2 3 4 5

#### 1

#### 2

#### 3

#### 4

#### 5

```
v⃗ 1
```
```
v⃗ 2
```
```
Figure 13.4: New cluster centres and the corresponding clusters
```
9. This divides the data into two clusters as follows (see Figure??):
    Cluster 1 :{x⃗ 1 ;⃗x 2 ;x⃗ 3 ;x⃗ 4 }represented by⃗v 1
    Cluster 2 :{x⃗ 5 ;⃗x 6 }represented by⃗v 2


#### CHAPTER 13. CLUSTERING METHODS 184

10. The cluster centres are recalculated as follows:

```
⃗v 1 =
```
#### 1

#### 4

```
(x⃗ 1 +x⃗ 2 +x⃗ 3 +x⃗ 4 )=( 2 : 00 ; 1 : 75 )
```
```
⃗v 2 =
```
#### 1

#### 2

```
(x⃗ 5 +x⃗ 6 )=( 4 : 00 ; 4 : 50 )
```
```
We note that these are identical to the cluster centres calculated in Step 8. So there will be no
reassignment of data points to different clusters and hence the computations are stopped here.
```
11. Conclusion: Thekmeans clustering algorithm withk= 2 applied to the dataset in Table 13.1
    yields the following clusters and the associated cluster centres:
    Cluster 1 :{x⃗ 1 ;⃗x 2 ;x⃗ 3 ;x⃗ 4 }represented by⃗v 1 =( 2 : 00 ; 1 : 75 )
    Cluster 2 :{x⃗ 5 ;⃗x 6 }represented by⃗v 2 =( 2 : 00 ; 4 : 75 )

13.2.3 The algorithm

Notations

We assume that each data point is an-dimensional vector:

```
x⃗=(x 1 ;x 2 ;:::;xn):
```
The distance between two data points

```
⃗x=(x 1 ;x 2 ;:::;xn)
```
and
⃗y=(y 1 ;y 2 ;:::;xn)

is defined as
SSx⃗−y⃗SS=

#### »

```
(x 1 −y 1 )^2 + ⋯(xn−yn)^2 :
```
LetX={x⃗ 1 ;:::;⃗xN}be the set of data points,V={⃗v 1 ;:::;⃗vk}be the set of centres andcifor
i= 1 ;:::;kbe the number of data points in thei-th cluster

Basic idea

What the algorithm aims to achieve is to find a partition the setXintokmutually disjoint subsets
S={S 1 ;S 2 ;:::;Sk}and a set of data pointsVwhich minimizes the following within-cluster sum
of errors:
k
Q
i= 1

#### Q

```
⃗x∈Si
```
```
SSx⃗−⃗viSS^2
```
Algorithm

Step 1. Randomly selectkcluster centersv⃗ 1 ;:::;⃗vk.

Step 2. Calculate the distance between each data point⃗xiand each cluster center⃗vj.

Step 3. For eachj= 1 ; 2 ;:::;N, assign the data pointx⃗jto the cluster center⃗vifor which the
distanceSSx⃗j−⃗viSSis minimum. Letx⃗i 1 ,x⃗i 2 ,:::,x⃗icibe the data points assigned to⃗vi.

Step 4. Recalculate the cluster centres using

```
⃗vi=
```
#### 1

```
ci
```
```
(⃗xi 1 + ⋯ +x⃗ici); i= 1 ; 2 ;:::;k:
```
Step 5. Recalculate the distance between each data point and newly obtained cluster centers.

Step 6. If no data point was reassigned then stop. Otherwise repeat from Step 3.


#### CHAPTER 13. CLUSTERING METHODS 185

Some methods for initialisation

The following are some of the methods for choosing the initialvi’s.

- Randomly take somekdata points as the initialvi’s.
- Calculate the mean of all data and add small random vectors to the mean to get thekinitial
    vi’s.
- Calculate the principal component, divide its range intokequal intervals, partition the data
    intokgroups, and then take the means of these groups as the initial centres.

13.2.4 Disadvantages

Even though thek-means algorithm is fast, robust and easy to understand, there are several disad-
vantages to the algorithm.

- The learning algorithm requires apriori specification of the number of cluster centers.
- The final cluster centres depend on the initialvi’s.
- With different representation of data we get different results (data represented in form of
    cartesian co-ordinates and polar co-ordinates will give different results).
- Euclidean distance measures can unequally weight underlying factors.
- The learning algorithm provides the local optima of the squared error function.
- Randomly choosing of the initial cluster centres may not lead to a fruitful result.
- The algorithm cannot be applied to categorical data.

13.2.5 Application: Image segmentation and compression

Image segmentation

The goal of segmentation is to partition an image into regions each of which has a reasonably
homogeneous visual appearance or which corresponds to objects or parts of objects. Each pixel in
an image is a point in a 3-dimensional space comprising the intensities of the red, blue, and green
channels. A segmentation algorithm simply treats each pixel in the image as a separate data point.
For any value ofk, each pixel is replaced by the pixel vector with the(R;G;B)intensity triplet
given by the centrekto which that pixel has been assigned. For a given value ofk, the algorithm
is representing the image using a palette of onlykcolours. It should be emphasized that this use of
k-means is a very crude approach to image segmentation. The image segmentation problem is in
general extremely difficult.

Data compression

We can also the clustering algorithm to perform data compression. There are two types of data
compression:lossless data compression, in which the goal is to be able to reconstruct the original
data exactly from the compressed representation, andlossy data compression, in which we accept
some errors in the reconstruction in return for higher levels of compression than can be achieved in
the lossless case.
We can apply thek-means algorithm to the problem of lossy data compression as follows. For
each of theNdata points, we store only the identity of the cluster to which it is assigned. We also
store the values of thekcluster centresk, which requires much less data, provided we choose
kmuch smaller thanN. Each data point is then approximated by its nearest centrek. New data
points can similarly be compressed by first finding the nearestkand then storing the labelkinstead
of the original data vector. This framework is often calledvector quantization, and the vectors Îijk
are calledcode-book vectors.


#### CHAPTER 13. CLUSTERING METHODS 186

### 13.3 Multi-modal distributions

13.3.1 Definitions

1. In statistics, aunimodal distributionis a continuous probability distribution with only one
    mode (or “peak”).
    A random variable having the normal distribution is a unimodal distribution. Similarly, the
    t-distribution and the chi-squared distribution are also unimodal distributions.

```
Unimodal Bimodal Multimodal
```
```
Figure 13.5: Probability distributions
```
2. Abimodal distributionis a continuous probability distribution with two different modes. The
    modes appear as distinct peaks in the graph of the probability density function.
3. Amultimodal distributionis a continuous probability distribution with two or more modes.

### 13.4 Mixture of normal distributions

13.4.1 Bimodal mixture

Consider the following functions which are probability density functions of normally distributed
random variables.

```
f 1 (x)=
```
#### 1

####  1

#### √

#### 2 

```
e
```
```
−(x−^1 )
2
2 ^21 (13.1)
```
```
f 2 (x)=
```
#### 1

####  2

#### √

#### 2 

```
e
```
```
−(x−^2 )
2
2 ^22 (13.2)
```
Now consider the following function:

```
f(x)= 1 f 1 (x)+ 2 f 2 (x) (13.3)
```
where 1 and 2 are some constants satisfying the relation

```
 1 + 2 = 1 : (13.4)
```
It can be shown that the function given in Eq.(13.3) together with Eq.(13.4) defines a probability
density function. It can also be shown that the graph of this function has two peaks. Hence this
function defines a bimodal distribution. This distribution is called a mixture of the normal distribu-
tions defined by Eqs.(13.1) and (13.2). We may mix more than two normal distributions.


#### CHAPTER 13. CLUSTERING METHODS 187

13.4.2 Definition

Consider the followingkprobability density functions:

```
fi(x)=
```
#### 1

```
i
```
#### √

#### 2 

```
e
```
```
−(x−i)
2
2 ^2 i ; i= 1 ; 2 ;:::;k: (13.5)
```
Let 1 ; 2 ;:::;kbe constants such that

```
i≥ 0 ; i= 1 ; 2 ;:::;k (13.6)
 1 + 2 + ⋯ +k= 1 : (13.7)
```
Then the random variableXwhose probability density function is

```
f(x)=f 1 (x)+f 2 (x)+ ⋯ +fk(x); (13.8)
```
is said to be amixture of theknormal distributionshaving the probability density functions defined
in Eq.(13.5).

A natural example

As a natural example for such mixtures of normal populations, we consider the probability distribu-
tion of heights of people in a region. This is a mixture of two normal distributions: the distribution
of heights of males and the distribution of heights of females. Given only the height data and not
the gender assignments for each data point, the distribution of all heights would follow the weighted
sum of two normal distributions.

13.4.3 Example for mixture of two normal distributions

Data and histogram

Consider the 100 observations of some attributeXgiven in Table 13.2.

```
[1] 5.39 1.30 2.95 2.16 2.37 2.33 4.76 2.99 1.71 2.41
[11] 2.71 2.79 0.54 1.37 5.16 1.22 1.58 4.34 3.83 3.44
[21] 3.68 5.03 0.92 2.57 1.97 2.17 5.02 2.73 1.63 3.09
[31] 4.05 3.76 3.13 6.50 5.10 3.62 3.14 2.36 2.73 4.08
[41] 3.28 2.28 1.52 3.86 2.10 0.86 2.94 2.18 3.39 2.55
[51] 3.23 3.30 2.16 3.86 1.92 2.55 4.33 0.86 2.68 2.24
[61] 2.82 3.63 2.84 3.82 2.49 3.25 2.39 3.18 6.35 4.16
[71] 6.68 5.26 8.00 6.27 7.98 6.50 6.56 8.50 7.48 6.42
[81] 5.99 7.44 6.96 7.10 8.48 6.99 7.29 6.87 6.71 7.99
[91] 8.19 8.28 6.98 7.43 8.33 5.65 8.96 7.36 5.24 7.30
```
```
Table 13.2: A set of 100 observations of a numeric attributeX
```
To make some sense of this set of observations, let us construct the frequency table for the data
as in Table 13.3.

```
Range 0-1 1-2 2-3 3-4 4-5 5-6 6-7 7 -8 8-9 9-10
Frequency 4 9 26 18 6 9 12 9 7 0
Relative
frequency 0.04 0.09 0.26 0.18 0.06 0.09 0.12 0.09 0.07 0.00
```
```
Table 13.3: Frequency table of data in Table 13.2
```

#### CHAPTER 13. CLUSTERING METHODS 188

Figure 13.6 shows the histogram of the relative frequencies. Notice that the histogram has two
“peaks”, one nearx= 2 : 5 and one nearx= 6 : 5. So, the graph of the probability density function of
the attributeXmust have two peaks. Recall that the graph of the probability density function of a
random variable having the normal distribution has only one peak.

Probability distribution

The data in Table 13.2 was generated using the R programming language. It is a true “mixture” of
the values two normally distributed random variables. 70% of the observations are random values
of a normally distributed random variable with 1 = 3 and 1 = 1 : 20 and 30% of the observations
are values of a normally distributed random variable with 2 = 7 and 2 = 0 : 87. The weight for the
first normal distribution is 1 =70%= 0 : 7 and that for the second distribution is 2 =30%= 0 : 3.
The probability density function for the mixed distribution is

```
f(x)= 0 : 7 ×
```
#### 1

#### 1 : 20

#### √

#### 2 

```
e−(x−^3 )
```
(^2) ~( 2 × 1 : 202 )
+ 0 : 3 ×

#### 1

#### 0 : 87

#### √

#### 2 

```
e−(x−^7 )
```
(^2) ~( 2 × 0 : 872 )
: (13.9)
Figure 13.6 also shows the curve defined by Eq.(13.9) superimposed on the histogram of the relative
frequency distribution.
Figure 13.6: Graph of pdf defined by Eq.(13.9) superimposed on the histogram of the data in Table
13.3

### 13.5 Mixtures in terms of latent variables

Consider the mixture ofknormal distributions defined by Eqs.(13.5) – (13.8).
Let us define ak-dimensional random variable

```
Z⃗=(z 1 ;z 2 ;:::;zk)
```

#### CHAPTER 13. CLUSTERING METHODS 189

where eachz 1 is either 0 or 1 and a 1 appears only at one place; that is,

```
zi∈{ 0 ; 1 }andz 1 +z 2 + ⋯ +zk= 0 :
```
We also assume that
P(zk= 1 )=k:

The probability function ofZ⃗can be written in the form

P(Z⃗)= 1 z^1 z 22 :::kzk:
Now, suppose we have a set of observations{x 1 ;x 2 ;:::;xN}. Suppose that, in some way, we
can associate a value of the random variableZ⃗, sayZ⃗i, with each valuexiand think of the given set
of observations as a set of ordered pairs

```
{(x 1 ;Z⃗ 1 );(x 2 ;Z⃗ 2 );:::;(xN;Z⃗N)}:
```
Here, only thexi-s are known; theZ⃗i-s are unknown. Let us further assume that the conditional
probability distributionp(xSZ⃗)be given by

```
p(xSZ⃗)=[f 1 (x)]z^1 × ⋯ ×[fk(x)]zk:
```
Then the marginal distribution ofxis given by

```
p(x)=Q
Z⃗
```
```
p(Z⃗)P(xSZ⃗)
```
```
= 1 f 1 (x)+ ⋯ +kfk(x): (13.10)
```
The right hand side of Eq.(13.10) is the probability density function of a mixture ofknormal distri-
butions with weights 1 ;:::;k.
Thus, a mixture of normal distributions is the marginal distribution of a bivariate distribution
(x;Z⃗)whereZ⃗is an unobserved or latent variable.

### 13.6 Expectation-maximisation algorithm

The maximum likelihood estimation method (MLE) is a method for estimating the parameters of a
statistical model, given observations (see Section 6.5 for details). The method attempts to find the
parameter values that maximize the likelihood function, or equivalently the log-likelihood function,
given the observations.
Theexpectation-maximisation algorithm(sometimes abbreviated as theEM algorithm) is used
to find maximum likelihood estimates of the parameters of a statistical model in cases where the
equations cannot be solved directly. These models generally involve latent or unobserved variables
in addition to unknown parameters and known data observations. For example, a Gaussian mixture
model can be described by assuming that each observed data point has a corresponding unobserved
data point, or latent variable, specifying the mixture component to which each data point belongs.
The EM Algorithm is not really an algorithm. Rather it is a general procedure to create algo-
rithms for specific MLE problems. The complete details of this general procedure are beyond the
scope of this book. However, we present below a minimal outline of the algorithm

Outline of EM algorithm

Step 1. Initialise the parametersto be estimated.

Step 2. Expectation step (E-step)

```
Take the expected value of the complete data given the observation and the current param-
eter estimate, say,^j. This is a function ofand^j, say,Q(;^j).
```
Step 3. Maximization step (M-step)

```
Find the valuesthat maximizes the functionQ(;^j).
```
Step 4. Repeat Steps 1 and 2 until the parameter values or the likelihood function converge.


#### CHAPTER 13. CLUSTERING METHODS 190

### 13.7 The EM algorithm for Gaussian mixtures

In the case of Gaussian mixture problems, because of the nature of the function, finding a maximum
likelihood estimate by taking the derivatives of the log-likelihood function with respect to all the
parameters and simultaneously solving the resulting equations is nearly impossible. So we apply the
EM algorithm to solve the problem.
As already indicated, the EM algorithm is a general procedure for estimating the parameters
in a statistical model. This algorithm can be adapted to develop an algorithm for estimating the
parameters in a Gaussian mixture model. The adapted EM algorithm has been explained below.
(The details of how the EM algorithm can be adapted to estimate the parameters in a Gaussian
mixture model are also beyond the scope of this book. For details on these matters, one may refer to
[1]).

Problem

Suppose we are given a set ofNobservations

```
{x 1 ;x 2 ;:::;xN}
```
of a numeric variableX. LetXbe a mix ofknormal distributions and let the probability density
function ofXbe
f(x)= 1 f 1 (x)+ ⋯ +kfk(x)

where

```
i≥ 0 ; i= 1 ; 2 ;:::;k
i+ ⋯ +k= 1
```
```
fi(x)=
```
#### 1

```
i
```
#### √

#### 2 

```
e
```
```
−(x−i)
2
2 ^2 i ; i= 1 ; 2 ;:::;k:
```
Estimate the parameters 1 ;:::;k, 1 ;:::;kand 1 :::;k.

Log-likelihood function

Letdenote the set of parametersi;i;i(i= 1 ;:::;k). The log-likelihood function for the above
problem is given below:

```
logL()=logf(x 1 )+ ⋯ +logf(xN)
```
#### =

```
N
Q
i= 1
```
```
log
```
#### ⎛

#### ⎝

####  1

####  1

#### √

#### 2 

```
e
```
```
−(xi−^1 )
2
2 ^21 + ⋯ + k
k
```
#### √

#### 2 

```
e
```
```
−(xi−k)
```
```
2
2 ^2 k ⎞
⎠
```
#### (13.11)

The algorithm

Step 1. Initialise the meansi’s, the variances^2 i’s and the mixing coefficientsi’s.

Step 2. Calculate the following forn= 1 ;:::;Nandi= 1 ;:::;k:

(^) in=
ifi(xn)
 1 f 1 (xn)+ ⋯ +kfk(xn)
Ni= (^) i 1 + ⋯ + (^) iN
Step 3. Recalculate the parameters using the following:
(new)i =

#### 1

```
Ni
```
( (^) i 1 x 1 + ⋯ (^) iNxN)


#### CHAPTER 13. CLUSTERING METHODS 191

```
i^2 (new)=
```
#### 1

```
Ni
```
 (^) i 1 (x 1 −(new)i )^2 + ⋯ + (^) iN(x 1 −(new)i )^2 
i(new)=
Ni
N
Step 4. Evaluate the log-likelihood function given in Eq.(13.11) and check for convergence of ei-
ther the parameters or the log-likelihood function. If the convergence criterion is not satis-
fied, return to Step 2.

### 13.8 Hierarchical clustering

Hierarchical clustering(also called hierarchical cluster analysis or HCA) is a method of cluster
analysis which seeks to build a hierarchy of clusters (or groups) in a given dataset. The hierarchical
clustering produces clusters in which the clusters at each level of the hierarchy are created by merg-
ing clusters at the next lower level. At the lowest level, each cluster contains a single observation.
At the highest level there is only one cluster containing all of the data.
The decision regarding whether two clusters are to be merged or not is taken based on themea-
sure of dissimilaritybetween the clusters. The distance between two clusters is usually taken as the
measure of dissimilarity between the clusters.
In Section??, we shall see various methods for measuring the distance between two clusters.

13.8.1 Dendrograms

Hierarchical clustering can be represented by a rooted binary tree. The nodes of the trees represent
groups or clusters. The root node represents the entire data set. The terminal nodes each represent
one of the individual observations (singleton clusters). Each nonterminal node has two daughter
nodes.
The distance between merged clusters is monotone increasing with the level of the merger. The
height of each node above the level of the terminal nodes in the tree is proportional to the value of
the distance between its two daughters (see Figure 13.9).
Adendrogramis a tree diagram used to illustrate the arrangement of the clusters produced by
hierarchical clustering.
The dendrogram may be drawn with the root node at the top and the branches growing vertically
downwards (see Figure 13.8(a)). It may also be drawn with the root node at the left and the branches
growing horizontally rightwards (see Figure 13.8(b)). In some contexts, the opposite directions may
also be more appropriate.
Dendrograms are commonly used in computational biology to illustrate the clustering of genes
or samples.

Example

Figure 13.7 is a dendrogram of the dataset{a;b;c;d;e}. Note that the root node represents the en-
tire dataset and the terminal nodes represent the individual observations. However, the dendrograms
are presented in a simplified format in which only the terminal nodes (that is, the nodes represent-
ing the singleton clusters) are explicitly displayed. Figure 13.8 shows the simplified format of the
dendrogram in Figure 13.7.
Figure 13.9 shows the distances of the clusters at the various levels. Note that the clusters are at
4 levels. The distance between the clusters{a}and{b}is 15, between{c}and{d}is 7.5, between
{c;d}and{e}is 15 and between{a;b}and{c;d;e}is 25.

13.8.2 Methods for hierarchical clustering

There are two methods for the hierarchical clustering of a dataset. These are known as theagglom-
erative method(or the bottom-up method) and thedivisive method(or, the top-down method).


#### CHAPTER 13. CLUSTERING METHODS 192

```
a b c d e
```
```
a;b
```
```
c;d
```
```
c;d;e
```
```
a;b;c;d;e
```
```
Figure 13.7: A dendrogram of the dataset{a;b;c;d;e}
```
```
a b c d e a
```
```
b
```
```
c
```
```
d
```
```
e
```
```
(a) (b)
```
```
Figure 13.8: Different ways of drawing dendrogram
```
```
Distance
```
#### 0

#### 5

#### 10

#### 15

#### 20

#### 25

```
a b c d e Level 1
```
```
Level 2
```
```
Level 3
```
```
Level 4
```
Figure 13.9: A dendrogram of the dataset{a;b;c;d;e}showing the distances (heights) of the clus-
ters at different levels

Agglomerative method

In the agglomerative we start at the bottom and at each level recursively merge a selected pair of
clusters into a single cluster. This produces a grouping at the next higher level with one less cluster.
If there areNobservations in the dataset, there will beN− 1 levels in the hierarchy. The pair chosen
for merging consist of the two groups with the smallest “intergroup dissimilarity”.
For example, the hierarchical clustering shown in Figure 13.7 can be constructed by the agglom-
erative method as shown in Figure 13.10. Each nonterminal node has two daughter nodes. The
daughters represent the two groups that were merged to form the parent.


#### CHAPTER 13. CLUSTERING METHODS 193

```
a b c d e
```
```
Step 1
```
```
a b c d e
```
```
a;b
```
```
Step 2
```
```
a b c d e
```
```
a;b c;d
```
```
Step 3
```
```
a b c d e
```
```
a;b
```
```
c;d
```
```
c;d;e
```
```
Step 4
```
```
a b c d e
```
```
a;b
```
```
c;d
```
```
c;d;e
```
```
a;b;c;d;e
```
```
Step 5
```
```
Figure 13.10: Hierarchical clustering using agglomerative method
```

#### CHAPTER 13. CLUSTERING METHODS 194

Divisive method

The divisive method starts at the top and at each level recursively split one of the existing clusters at
that level into two new clusters. If there areNobservations in the dataset, there the divisive method
also will produceN− 1 levels in the hierarchy. The split is chosen to produce two new groups with
the largest “between-group dissimilarity”.
For example, the hierarchical clustering shown in Figure 13.7 can be constructed by the divi-
sive method as shown in Figure 13.11. Each nonterminal node has two daughter nodes. The two
daughters represent the two groups resulting from the split of the parent.

### 13.9 Measures of dissimilarity

In order to decide which clusters should be combined (for agglomerative), or where a cluster should
be split (for divisive), a measure of dissimilarity between sets of observations is required. In most
methods of hierarchical clustering, the dissimilarity between two groups of observations is measured
by using an appropriate measure of distance between the groups of observations. The distance
between two groups of observations is defined in terms of the distance between two observations.
There are several ways in which the distance between two observations can be defined and also there
are also several ways in which the distance between two groups of observations can be defined.

13.9.1 Measures of distance between data points

Numeric data

We assume that each observation or data point is an-dimensional vector. Let⃗x=(x 1 ;:::;xn)
and⃗y=(y 1 ;:::;yn)be two observations. Then the following are the commonly used measures of
distances in the hierarchical clustering of numeric data.

```
Name Formula
Euclidean distance SSx⃗−y⃗SS 2 =
```
#### »

```
(x 1 −y 1 )^2 + ⋯ +(xn−yn)^2
Squared Euclidean distance SSx⃗−y⃗SS^22 =(x 1 −y 1 )^2 + ⋯ +(xn−yn)^2
Manhattan distance SSx⃗−y⃗SS 1 =Sx 1 −y 1 S+ ⋯ +Sxn−ynS
Maximum distance SSx⃗−y⃗SS∞=max{Sx 1 −y 1 S;:::;Sxn−ynS}
```
Non-numeric data

For text or other non-numeric data, metrics such as the Levenshtein distance are often used.
TheLevenshtein distanceis a measure of the ”distance” between two words. The Levenshtein
distance between two words is the minimum number of single-character edits (insertions, deletions
or substitutions) required to change one word into the other.
For example, the Levenshtein distance between “kitten” and “sitting” is 3, since the following
three edits change one into the other, and there is no way to do it with fewer than three edits:

```
kitten→sitten (substitution of “s” for “k”)
sitten→sittin (substitution of “i” for “e”)
sittin→sitting (insertion of‘g” at the end)
```
13.9.2 Measures of distance between groups of data points

LetAandBbe two groups of observations and letxandybe arbitrary data points inAandB
respectively. Suppose we have chosen some formula, say Euclidean distance formula, to measure
the distance between data points. Letd(x;y)denote the distance betweenxandy. We denote by


#### CHAPTER 13. CLUSTERING METHODS 195

```
a;b;c;d;e
```
```
Step 1
```
```
a;b;c;d;e
```
```
a;b c;d;e
```
```
Step 2
```
```
a b
```
```
a;b;c;d;e
```
```
a;b c;d;e
```
```
Step 3
```
```
a b e
```
```
a;b;c;d;e
```
```
a;b c;d;e
```
```
c;d
```
```
Step 4
```
```
a b c d e
```
```
a;b
```
```
c;d
```
```
c;d;e
```
```
a;b;c;d;e
```
```
Step 5
```
```
Figure 13.11: Hierarchical clustering using divisive method
```

#### CHAPTER 13. CLUSTERING METHODS 196

d(A;B)the distance between the groupsAandB. The following are some of the different methods
in whichd(A;B)is defined.

1. d(A;B)=max{d(x;y)∶x∈A;y∈B}.
    Agglomerative hierarchical clustering using this measure of dissimilarity is known ascomplete-
    linkage clustering. The method is also known asfarthest neighbour clustering.

```
a
```
```
b c
```
```
d
```
```
e
```
#### A

#### B

```
Figure 13.12: Length of the solid line “ae” ismax{d(x;y)∶x∈A;y∈B}
```
2. d(A;B)=min{d(x;y)∶x∈A;y∈B}.
    Agglomerative hierarchical clustering using this measure of dissimilarity is known assingle-
    linkage clustering. The method is also known asnearest neighbour clustering.

```
a
```
```
b c
```
```
d
```
```
e
```
#### A

#### B

```
Figure 13.13: Length of the solid line “bc” ismin{d(x;y)∶x∈A;y∈B}
```
3. d(A;B)=

#### 1

#### SAS SBS

#### Q

```
x∈A;y∈B
```
```
d(x;y)whereSAS,SBSare respectively the number of elements in
```
```
AandB.
Agglomerative hierarchical clustering using this measure of dissimilarity is known asmean
or average linkage clustering. It is also known as UPGMA (Unweighted Pair Group Method
with Arithmetic Mean).
```
### 13.10 Algorithm for agglomerative hierarchical clustering

Given a set ofNitems to be clustered and anN×Ndistance matrix, required to construct a
hierarchical clustering of the data using the agglomerative method.

Step 1. Start by assigning each item to its own cluster, so that we haveNclusters, each containing
just one item. Let the distances between the clusters equal the distances between the items
they contain.


#### CHAPTER 13. CLUSTERING METHODS 197

Step 2. Find the closest pair of clusters and merge them into a single cluster, so that now we have
one less cluster.

Step 3. Compute distances between the new cluster and each of the old clusters.

Step 4. Repeat Steps 2 and 3 until all items are clustered into a single cluster of sizeN.

13.10.1 Example

Problem 1

Given the dataset{a;b;c;d;e}and the following distance matrix, construct a dendrogram by complete-
linkage hierarchical clustering using the agglomerative method.

```
a b c d e
a 0 9 3 6 11
b 9 0 7 5 10
c 3 7 0 9 2
d 6 5 9 0 8
e 11 10 2 8 0
```
```
Table 13.4: Example for distance matrix
```
Solution

The complete-linkage clustering uses the “maximum formula”, that is, the following formula to
compute the distance between two clustersAandB:

```
d(A;B)=max{d(x;y)∶x∈A;y∈B}
```
1. Dataset :{a;b;c;d;e}.
    Initial clustering (singleton sets)C 1 :{a},{b},{c},{d},{e}.
2. The following table gives the distances between the various clusters inC 1 :

```
{a} {b} {c} {d} {e}
{a} 0 9 3 6 11
{b} 9 0 7 5 10
{c} 3 7 0 9 2
{d} 6 5 9 0 8
{e} 11 10 2 8 0
```
```
In the above table, the minimum distance is the distance between the clusters{c}and{e}.
Also
d({c};{e})= 2 :
```
```
We merge{c}and{e}to form the cluster{c;e}.
The new set of clustersC 2 :{a},{b},{d},{c;e}.
```
3. Let us compute the distance of{c;e}from other clusters.
    d({c;e};{a})=max{d(c;a);d(e;a)}=max{ 3 ; 11 }= 11 :
    d({c;e};{b})=max{d(c;b);d(e;b)}=max{ 7 ; 10 }= 10 :
    d({c;e};{d})=max{d(c;d);d(e;d)}=max{ 9 ; 8 }= 9 :
    The following table gives the distances between the various clusters inC 2.


#### CHAPTER 13. CLUSTERING METHODS 198

```
{a} {b} {d} {c;e}
{a} 0 9 6 11
{b} 9 0 5 10
{d} 6 5 0 9
{c;e} 11 10 9 0
```
```
In the above table, the minimum distance is the distance between the clusters{b}and{d}.
Also
d({b};{d})= 5 :
```
```
We merge{b}and{d}to form the cluster{b;d}.
The new set of clustersC 3 :{a},{b;d},{c;e}.
```
4. Let us compute the distance of{b;d}from other clusters.
    d({b;d};{a})=max{d(b;a);d(d;a)}=max{ 9 ; 6 }= 9 :
    d({b;d};{c;e})=max{d(b;c);d(b;e);d(d;c);d(d;e)}=max{ 7 ; 10 ; 9 ; 8 }= 10 :
    The following table gives the distances between the various clusters inC 3.

```
{a} {b;d} {c;e}
{a} 0 9 11
{b;d} 9 0 10
{c;e} 11 10 0
```
```
In the above table, the minimum distance is the distance between the clusters{a}and{b;d}.
Also
d({a};{b;d})= 9 :
```
```
We merge{a}and{b;d}to form the cluster{a;b;d}.
The new set of clustersC 4 :{a;b;d},{c;e}
```
5. Only two clusters are left. We merge them form a single cluster containing all data points. We
    have

```
d({a;b;d};{c;e})=max{d(a;c);d(a;e);d(b;c);d(b;e);d(d;c);d(d;e)}
=max{ 3 ; 11 ; 7 ; 10 ; 9 ; 8 }
= 11
```
6. Figure 13.14 shows the dendrogram of the hierarchical clustering.

Problem 2

Given the dataset{a;b;c;d;e}and the distance matrix given in Table 13.4, construct a dendrogram
by single-linkage hierarchical clustering using the agglomerative method.

Solution

The complete-linkage clustering uses the “maximum formula”, that is, the following formula to
compute the distance between two clustersAandB:

```
d(A;B)=min{d(x;y)∶x∈A;y∈B}
```
1. Dataset :{a;b;c;d;e}.
    Initial clustering (singleton sets)C 1 :{a},{b},{c},{d},{e}.


#### CHAPTER 13. CLUSTERING METHODS 199

```
Distance
```
#### 0

#### 2

#### 4

#### 6

#### 8

#### 10

```
a b d c e
```
```
Figure 13.14: Dendrogram for the data given in Table 13.4 (complete linkage clustering)
```
2. The following table gives the distances between the various clusters inC 1 :

```
{a} {b} {c} {d} {e}
{a} 0 9 3 6 11
{b} 9 0 7 5 10
{c} 3 7 0 9 2
{d} 6 5 9 0 8
{e} 11 10 2 8 0
```
```
In the above table, the minimum distance is the distance between the clusters{c}and{e}.
Also
d({c};{e})= 2 :
```
```
We merge{c}and{e}to form the cluster{c;e}.
The new set of clustersC 2 :{a},{b},{d},{c;e}.
```
3. Let us compute the distance of{c;e}from other clusters.
    d({c;e};{a})=min{d(c;a);d(e;a)}=max{ 3 ; 11 }= 3 :
    d({c;e};{b})=min{d(c;b);d(e;b)}=max{ 7 ; 10 }= 7 :
    d({c;e};{d})=min{d(c;d);d(e;d)}=max{ 9 ; 8 }= 8 :
    The following table gives the distances between the various clusters inC 2.

```
{a} {b} {d} {c;e}
{a} 0 9 6 3
{b} 9 0 5 7
{d} 6 5 0 8
{c;e} 3 7 8 0
```
```
In the above table, the minimum distance is the distance between the clusters{a}and{c;e}.
Also
d({a};{c;e})= 3 :
```
```
We merge{a}and{c;e}to form the cluster{a;c;e}.
The new set of clustersC 3 :{a;c;e},{b},{d}.
```

#### CHAPTER 13. CLUSTERING METHODS 200

4. Let us compute the distance of{a;c;e}from other clusters.
    d({a;c;e};{b})=min{d(a;b);d(c;b);d(e;b)}={ 9 ; 7 ; 10 }= 7
    d({a;c;e};{d})=min{d(a;d);d(c;d);d(e;d)}={ 6 ; 9 ; 8 }= 6
    The following table gives the distances between the various clusters inC 3.

```
{a;c;e} {b} {d}
{a;c;e} 0 7 6
{b} 7 0 5
{d} 6 5 0
```
```
In the above table, the minimum distance is between{b}and{d}. Also
```
```
d({b};{d})= 5 :
```
```
We merge{b}and{d}to form the cluster{b;d}.
The new set of clustersC 4 :{a;c;e},{b;d}
```
5. Only two clusters are left. We merge them form a single cluster containing all data points. We
    have

```
d({a;c;e};{b;d})=min{d(a;b);d(a;d);d(c;b);d(c;d);d(e;b);d(e;d)}
=min{ 9 ; 6 ; 7 ; 9 ; 10 ; 8 }
= 6
```
6. Figure 13.15 shows the dendrogram of the hierarchical clustering.

```
Distance
```
#### 0

#### 1

#### 2

#### 3

#### 4

#### 5

#### 6

```
a c e b d
```
```
Figure 13.15: Dendrogram for the data given in Table 13.4 (single linkage clustering)
```
### 13.11 Algorithm for divisive hierarchical clustering

Divisive clustering algorithms begin with the entire data set as a single cluster, and recursively divide
one of the existing clusters into two daughter clusters at each iteration in a top-down fashion. To
apply this procedure, we need a separate algorithm to divide a given dataset into two clusters.

- The divisive algorithm may be implemented by using thek-means algorithm withk= 2 to
    perform the splits at each iteration. However, it would not necessarily produce a splitting
    sequence that possesses the monotonicity property required for dendrogram representation.


#### CHAPTER 13. CLUSTERING METHODS 201

13.11.1 DIANA (DIvisive ANAlysis)

DIANA is a divisive hierarchical clustering technique. Here is an outline of the algorithm.

Step 1. Suppose that clusterClis going to be split into clustersCiandCj.

Step 2. LetCi=ClandCj=∅.

Step 3. For each objectx∈Ci:

```
(a) For the first iteration, compute the average distance ofxto all other objects.
(b) For the remaining iterations, compute
```
```
Dx=average{d(x;y)∶y∈Ci}−average{d(x;y)∶y∈Cj}:
```
```
x
```
```
Ci
```
```
Cj
```
```
Figure 13.16:Dx= (average of dashed lines)−(average of solid lines)
```
Step 4. (a) For the first iteration, move the object with the maximum average distance toCj.

```
(b) For the remaining iterations, find an objectxinCifor whichDxis the largest. If
Dx> 0 then movextoCj.
```
Step 5. Repeat Steps 3(b) and 4(b) until all differencesDxare negative. ThenClis split intoCiand
Cj.

Step 6. Select the smaller cluster with the largest diameter. (The diameter of a cluster is the largest
dissimilarity between any two of its objects.) Then divide this cluster, following Steps 1-5.

Step 7. Repeat Step 6 until all clusters contain only a single object.

13.11.2 Example

Problem

Given the dataset{a;b;c;d;e}and the distance matrix in Table 13.4, construct a dendrogram by the
divisive analysis algorithm.

Solution

1. We have, initially
    Cl={a;b;c;d;e}
2. We write
    Ci=Cl; Cj=∅:
3. Division into clusters


#### CHAPTER 13. CLUSTERING METHODS 202

```
(a) Initial iteration
Let us calculate the average dissimilarities of the objects inCiwith the other objects in
Ci.
Average dissimilarity ofa
```
#### =

#### 1

#### 4

```
(d(a;b)+d(a;c)+d(a;e))=
```
#### 1

#### 4

#### ( 9 + 3 + 6 + 11 )= 7 : 25

```
Similarly we have :
Average dissimilarity ofb= 7 : 75
Average dissimilarity ofc= 5 : 25
Average dissimilarity ofd= 7 : 00
Average dissimilarity ofe= 7 : 75
The highest average distance is 7 : 75 and there are two corresponding objects. We choose
one of them,b, arbitrarily. We movebtoCj.
We now have
Ci={a;c;d;e}; Cj=∅ ∪{b}={b}:
```
```
(b) Remaining iterations
(i) 2-nd iteration.
```
```
Da=
```
#### 1

#### 3

```
(d(a;c)+d(a;d)+d(a;e))−
```
#### 1

#### 1

```
(d(a;b))=
```
#### 20

#### 3

#### − 9 =− 2 : 33

```
Dc=
```
#### 1

#### 3

```
(d(c;a)+d(c;d)+d(c;e))−
```
#### 1

#### 1

```
(d(c;b))=
```
#### 14

#### 3

#### − 7 =− 2 : 33

```
Dd=
```
#### 1

#### 3

```
(d(d;a)+d(d;c)+d(d;e))−
```
#### 1

#### 1

```
(d(c;b))=
```
#### 23

#### 3

#### − 7 = 0 : 67

```
De=
```
#### 1

#### 3

```
(d(e;a)+d(e;c)+d(e;d))−
```
#### 1

#### 1

```
(d(e;b))=
```
#### 21

#### 3

#### − 7 = 0

```
Ddis the largest andDd> 0. So we move,dtoCj.
We now have
Ci={a;c;e}; Cj={b}∪{d}={b;d}:
(ii) 3-rd iteration
```
```
Da=
```
#### 1

#### 2

```
(d(a;c)+d(a;e))−
```
#### 1

#### 2

```
(d(a;b)+d(a;d))=
```
#### 14

#### 2

#### −

#### 15

#### 2

#### =− 0 : 5

```
Dc=
```
#### 1

#### 2

```
(d(c;a)+d(c;e))−
```
#### 1

#### 2

```
(d(c;b)+d(c;d))=
```
#### 5

#### 2

#### −

#### 16

#### 2

#### =− 13 : 5

```
De=
```
#### 1

#### 2

```
(d(e;a)+d(e;c))−
```
#### 1

#### 2

```
(d(e;b)+d(e;d))=
```
#### 13

#### 2

#### −

#### 18

#### 2

#### =− 2 : 5

```
All are negative. So we stop and form the clustersCiandCj.
```
4. To divide,CiandCj, we compute their diameters.

```
diameter(Ci)=max{d(a;c);d(a;e);d(c;e)}
=max{ 3 ; 11 ; 2 }
= 11
diameter(Cj)=max{d(b;d)}
= 5
```
```
The cluster with the largest diameter isCi. So we now splitCi.
We repeat the process by takingCl={a;c;e}. The remaining computations are left as an
exercise to the reader.
```

#### CHAPTER 13. CLUSTERING METHODS 203

### 13.12 Density-based clustering

In density-based clustering, clusters are defined as areas of higher density than the remainder of the
data set. Objects in these sparse areas - that are required to separate clusters - are usually considered
to be noise and border points. The most popular density based clustering method is DBSCAN
(Density-Based Spatial Clustering of Applications with Noise).

```
Figure 13.17: Clusters of points and noise points not belonging to any of those clusters
```
13.12.1 Density

We introduce some terminology and notations.

- Let(epsilon) be some constant distance. Letpbe an arbitrary data point. The-neighbourhood
    ofpis the set
       N(p)={q∶d(p;q)<}
- We choose some numberm 0 to define points of “high density”: We say that a pointpis point
    ofhigh densityifN(p)contains at leastm 0 points.
- We define a pointpas acore pointifN(p)has more thanm 0 points.
- We define a pointpas aborder pointifN(p)has fewer thanm 0 points, but is in the-
    neighbourhood of a core point.
- A point which is neither a core point nor a border point is called anoise point.

```
p p p q r q
```
```
(a) (b) (c) (d)
```
```
Figure 13.18: Withm 0 = 4 : (a)pa point of high density (b)pa core point (c)pa border point
(d)ra noise point
```
- An objectqisdirectly density-reachablefrom objectpifpis a core object andqis inN(p).
- An objectqisindirectly density-reachablefrom an objectpif there is a finite set of objects
    p 1 ;:::;prsuch thatp 1 is directly density-reachable formp,p 2 is directly density reachable
    fromp 1 , etc.,qis directly density-reachable formpr.


#### CHAPTER 13. CLUSTERING METHODS 204

```
p q p p 1 p 2 p 3 q
```
```
(a) (b)
```
```
Figure 13.19: Withm 0 = 4 : (a)qis directly density-reachable fromp(b)qis indirectly
density-reachable fromp
```
13.12.2 DBSCAN algorithm

LetX={x 1 ;x 2 ;:::;xn}be the set of data points. DBSCAN requires two parameters:(eps) and
the minimum number of points required to form a cluster (m 0 ).

Step 1. Start with an arbitrary starting pointpthat has not been visited.

Step 2. Extract the-neighborhoodN(p)ofp.

Step 3. If the number of points inN(p)is not greater thanm 0 then the pointpis labeled as noise
(later this point can become the part of the cluster).

Step 4. If the number of points inN(p)is greater thanm 0 then the pointpis a core point and is
marked as visited. Select a newcluster-idand mark all objects inN(p)with this cluster-id.

Step 5. If a point is found to be a part of the cluster then its-neighborhood is also the part of the
cluster and the above procedure from step 2 is repeated for all-neighborhood points. This
is repeated until all points in the cluster are determined.

Step 6. A new unvisited point is retrieved and processed, leading to the discovery of a further
cluster or noise.

Step 7. This process continues until all points are marked as visited.

### 13.13 Sample questions

(a) Short answer questions

1. What is clustering?
2. Is clustering supervised learning? Why?
3. Explain some applications of thek-means algorithm.
4. Explain how clustering technique is used in image segmentation problem.
5. Explain how clustering technique used in data compression.
6. What is meant by the mixture of two normal distributions?
7. Explain hierarchical clustering.
8. What is a dendrogram? Give an example.
9. Is hierarchical clustering unsupervised learning? Why?
10. Describe the two methods for hierarchical clustering.


#### CHAPTER 13. CLUSTERING METHODS 205

11. In a clustering problem, what does the measure of dissimilarity measure? Give some examples
    of measures of dissimilarity.
12. Explain the different types of linkages in clustering.
13. In the context of density-based clustering, define high density point, core point, border point
    and noise point.
14. What is agglomerative hierarchical clustering?

(b) Long answer questions

1. Applyk-means algorithm for given data withk= 3. UseC 1 ( 2 ),C 2 ( 16 )andC 3 ( 38 )as initial
    centers. Data:
       2 ; 4 ; 6 ; 3 ; 31 ; 12 ; 15 ; 16 ; 38 ; 35 ; 14 ; 21 ; 3 ; 25 ; 30
2. Explain K-means algorithm and group the points (1, 0, 1), (1, 1, 0), (0, 0, 1) and (1, 1, 1) using
    K-means algorithm.
3. Applying thek-means algorithm, find two clusters in the following data.

```
x 185 170 168 179 182 188 180 180 183 180 180 177
y 72 56 60 68 72 77 71 70 84 88 67 76
```
4. Usek-means algorithm to find 2 clusters in the following data:

```
No. 1 2 3 4 5 6 7
x 1 1.0 1.5 3.0 5.0 3.5 4.5 3.5
x 2 1.0 2.0 4.0 7.0 5.0 5.0 4.5
```
5. Give a general outline of the expectation-maximization algorithm.
6. Describe EM algorithm for Gaussian mixtures.
7. Describe an algorithm for agglomerative hierarchical clustering.
8. Given the following distance matrix, construct the dendrogram using agglomerative clustering
    with single linkage, complete linkage and average linkage.

```
A B C D E
A 0 1 2 2 3
B 1 0 2 4 3
C 2 2 0 1 5
D 2 4 1 0 3
E 3 3 5 3 0
```
9. Describe an algorithm for divisive hierarchical clustering.
10. For the data in Question 8, construct a dendrogram using DIANA algorithm.
11. Describe the DBSCAN algorithm for clustering.


## Bibliography

```
[1] Christopher M. Bishop,Pattern Recognition and Machine Learning, Springer, 2006.
```
```
[2] Ethem Alpaydin, Introduction to Machine Learning, The MIT Press, Cambridge, Mas-
sachusetts, 2004.
```
```
[3] Margaret H. Dunham,Data Mining: Introductory and Advanced Topics, Pearson, 2006.
```
```
[4] Mitchell T.,Machine Learning, McGraw Hill.
```
```
[5] Ryszard S. Michalski, Jaime G. Carbonell, and Tom M. Mitchell,Machine Learning : An
Artificial Intelligence Approach, Tioga Publishing Company.
```
```
[6] Michael J. Kearns and Umesh V. Vazirani,An Introduction to Computational Learning Theory,
The MIT Press, Cambridge, Massachusetts, 1994.
```
```
[7] D. H. Wolpert, W. G. Macready (1997), “No Free Lunch Theorems for Optimization”, IEEE
Transactions on Evolutionary Computation 1, 67.
```
#### 206


## Index

5-by-2 cross-validation, 50

abstraction, 3
accuracy, 54
activation function, 113
Gaussian -, 115
hyperbolic -, 116
linear -, 115
threshold -, 114
unit step -, 114
agglomerative method, 192
algorithm
backpropagation -, 123
backward selection -, 37
Baum-Welch, 170
C4.5 -, 105
DBSCAN -, 204
decision tree -, 95
DIANA -, 201
forward selection -, 36
Forwards-Backwards, 170
ID3 -, 96
kernel method -, 157
naive Bayes -, 65
PCA -, 40
perceptron learning -, 118
random forest -, 177
SVM -, 149
Viterbi -, 170
ANN, 119
Arthur Samuel, 1
artificial neural networks, 119
association rule, 6
attribute, 4
axis-aligned rectangle, 18
axon, 111

backpropagation algorithm, 123
backward phase, 123
backward selection, 37
Basic problems of HMM’s, 169
Baum-Welch algorithm, 170
Bayes’ theorem, 62
bias, 23
bimodal mixture, 186

```
binary classification, 15
bootstrap, 51
bootstrap sampling, 51
bootstrapping, 51
border point, 203
```
```
C4.5 algorithm, 105
CART algorithm, 105
classification, 7
classification tree, 84
cluster analysis, 179
clustering, 179
complete-linkage -, 196
density-based -, 203
farthest neighbour -, 196
hierarchical -, 191
k-means -, 179
nearest neighbour -, 196
single-linkage -, 196
complete-linkage clustering, 196
compression, 8
computational learning theory, 31
concept class, 31
conditional probability, 61
confusion matrix, 52
consistent, 16
construction of tree, 85
core point, 203
cost function, 121
covariance matrix, 40
cross-validation, 25, 49
5-by-2 -, 50
hold-out -, 49
K-fold -, 49
leave-one-out -, 50
```
```
data
categorical -, 5
nominal -, 5
numeric - , 5
ordinal -, 5
data compression, 8, 185
data storage, 2
DBSCAN algorithm, 204
decision tree, 83
```
#### 207


#### INDEX 208

decision tree algorithm, 95
deep learning, 129
deep neural network, 129
delta learning rule, 127
dendrogram, 191
denrite, 111
density-based clustering, 203
DIANA, 201
dichotomy, 27
dimensionality reduction, 35
directly-density reachable, 203
discrete Markov process, 165
discriminant, 9
dissimilarity, 192
DIvisive ANAlysis, 201
divisive method, 194

E-step, 189
eigenvalue, 40
eigenvector, 41
EM algorithm, 189
ensemble learning, 176
entropy, 89
epoch, 123
error rate, 54
evaluation, 3
event
independent -, 61
example, 4
expectation step, 189
expectation-maximization algorithm, 189
experience
learning from -, 1

face recognition, 8
false negative, 51
false positive, 51
false positive rate, 55
farthest neighbour clustering, 196
feature, 4
feature extraction, 35
feature selection, 35
feedforward network, 120
first layer, 120
first principal component, 41
forward phase, 123
forward selection, 36
Forwards-Backwards algorithms, 170
FPR, 55

Gaussian activation function, 115
Gaussian mixture, 190
genralisation, 3
Gini index, 94
Gini split index, 94

```
gradient descent method, 123
```
```
hidden Markov model, 169
hidden node, 120
hierarchical clustering, 191
high density point, 203
HMM, 169
basic problems, 169
coin tossing example, 167
Evaluation problem, 169
learning parameter problem, 170
state sequence problem, 170
urn and ball model, 168
holdout method, 49
homogeneity property, 164
hyperplane, 141
hypothesis, 15
hypothesis space, 16
```
```
ID3 algorithm, 96
image segmentation, 185
independent
mutually -, 61
pairwise -, 61
independent event, 61
indirectly density-reachable, 203
inductive bias, 23
information gain, 92
initial probability, 164
inner product, 140
input feature, 15
input node, 120
input representation, 15
instance, 4
instance space, 29
internal node, 83
isolated word recognition, 170
```
```
K-fold cross-validation, 49
k-means clustering, 179
kernel
Gaussian -, 157
homogeneous polynomial -, 156
Laplacian -, 157
non-homogeneous polynomial -, 156
radial basis function -, 157
kernel function, 155
kernel method, 157
kernel method algorithm, 157
knowledge extraction, 8
```
```
Laplacian kernel, 157
latent variable, 188
layer in networks, 120
leaf node, 83
```

#### INDEX 209

learner, 2
learning, 1
reinforcement -, 13
supervised -, 11
unsupervised - , 12
learning associations, 6
learning program, 2
learning theory, 31
leave-one-out, 50
length of an instance, 32
Levenshtein distance, 194
likelihood, 63
linear activation function, 115
linear regression, 73
linearly separable data, 144
logistic function, 114
logistic regression, 73

M-step, 189
machine learning, 1
definition of -, 1
machine learning program, 2
Markov property, 164
maximal margin hyperplane, 145
maximisation step, 189
maximum margin hyperplane, 145
mean squared error, 35
measure of dissimilarity, 194
misclassification rate, 36
mixture of distributions, 186
model, 1
model selection, 23
more general than, 18
more specific than, 18
multiclass SVM, 158
multimodal distribution, 186
multiple class, 22
multiple linear regression, 78
multiple regression, 73

naive Bayes algorithm, 65
nearest neighbour clustering, 196
negative example, 15
neighbourhood, 203
network topology, 119
neural networks, 119
neuron
artificial -, 112
biological -, 111
no-free lunch theorem, 48
noise, 22
noise point, 203
norm, 140

observable Markov model, 165

```
Occam’s razor, 24
OLS method, 74
one-against-all, 22
one-against-all method, 158
one-against-one, 23
one-against-one method, 158
optical character recognition, 8
optimal separating hyperplane, 146
ordinary least square, 74
orthogonality, 140
output node, 120
overfitting, 24
```
```
PAC learnability, 31
PAC learning, 31
PCA, 38
PCA algorithm, 40
perceptron, 116
perceptron learning algorithm, 118
performance measure, 1
perpendicular distance, 144
perpendicularity, 140
polynomial kernel, 156
polynomial regression, 73
positive example, 15
precision, 53
principal component, 41
principal component analysis, 38
probability
conditional -, 61
posterior -, 63
prior -, 62
probably approximately correct learning, 31
```
```
radial basis function kernel, 157
random forest, 176
random forest algorithm, 177
random performance, 55
RDF kernel, 157
recall, 53
Receiver Operating Characteristic, 54
record, 4
recurrent network, 120
regression, 10
logistic -, 73
multiple , 73
polynomial -, 73
simple linear -, 73
regression function, 10
regression problem, 72
regression tree, 84, 101
reinforcement learning, 13
ROC, 54
ROC curve, 56
```

#### INDEX 210

ROC space, 55

saturated linear function, 115
scalar, 139
sensitivity, 54
separating line, 134
shallow network, 129
shattering, 28
sigmoid function, 114
simple linear regression, 73
single-linkage clustering, 196
size of a concept, 32
slack variable, 154
soft margin hyperplane, 154
specificity, 54
speech recognition, 8
storage, 2
strictly more general than, 18
strictly more specific than, 18
subset selection, 36
supervised learning, 11
support vector, 146
support vector machine, 146
SVM, 146
SVM algorithm, 149
SVM classifier, 148
synapse, 111

threshold function, 114
TPR, 55
training, 3
transition probability, 164
tree, 83
classification -, 84
regression -, 84
true negative, 51
true positive, 51
true positive rate, 55
two-class data set, 144

underfitting, 24
unimodal distribution, 186
unit of observation, 4
unit step function, 114
unsupervised learning, 12

validation set, 25
Vapnik-Chervonenkis dimension, 29
variable, 4
VC dimension, 29
vector space, 138
finite dimensional -, 138
version space, 19
Viterbi algorithm, 170

```
weighted least squares, 75
word recognition, 170
```
```
zero vector, 139
```


