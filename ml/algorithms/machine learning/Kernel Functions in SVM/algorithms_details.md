# Major Kernel Functions in Support Vector Machine (SVM) 
**Kernel Function** is a method used to take data as input and transform it into the required form of processing data. “Kernel” is used due to a set of mathematical functions used in Support Vector Machine providing the window to manipulate the data. So, Kernel Function generally transforms the training set of data so that a non-linear decision surface is able to transform to a linear equation in a higher number of dimension spaces. Basically, It returns the inner product between two points in a standard feature dimension.   
**Standard Kernel Function Equation :**    
![K (\bar{x}) = 1, if ||\bar{x}|| <= 1 ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-a862304d8b797326ff0502a70dec9993_l3.png "Rendered by QuickLaTeX.com")  
![K (\bar{x}) = 0, Otherwise ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-4f0382bdef579dde1a847cb25893ab59_l3.png "Rendered by QuickLaTeX.com")  
**Major Kernel Functions :-**   
For Implementing Kernel Functions, first of all, we have to install the “scikit-learn” library using the command prompt terminal:   
 

```
    pip install scikit-learn
```


*   **Gaussian Kernel:** It is used to perform transformation when there is no prior knowledge about data.

![K (x, y) = e ^ - (\frac{||x - y||^2} {2 \sigma^2}) ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-d7e7a69d8a516e737400e1735001c5fe_l3.png "Rendered by QuickLaTeX.com")  
 

*   **Gaussian Kernel Radial Basis Function (RBF):** Same as above kernel function, adding radial basis method to improve the transformation.

  
![K (x, y) = e ^ - (\gamma{||x - y||^2}) ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-486d573e512569e1b017e352f98cb278_l3.png "Rendered by QuickLaTeX.com")  
![K (x, x1) + K (x, x2) (Simplified - Formula) ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-a7ebed26a84a274239c3a1e94821bbce_l3.png "Rendered by QuickLaTeX.com")  
![K (x, x1) + K (x, x2) > 0 (Green) ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-5bad2b7743405cf90885ae9d78f02671_l3.png "Rendered by QuickLaTeX.com")  
![K (x, x1) + K (x, x2) = 0 (Red) ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-835887610f3ccde9d4d7623024d02e7b_l3.png "Rendered by QuickLaTeX.com")  
 

![](https://media.geeksforgeeks.org/wp-content/uploads/20200515140553/kernel.jpg)

**Gaussian Kernel Graph**

  
**Code:** 

python3
-------

`from` `sklearn.svm` `import` `SVC`

`classifier` `=` `SVC(kernel` `=``'rbf'``, random_state` `=` `0``)`

`classifier.fit(x_train, y_train)`

*   **Sigmoid Kernel:** this function is equivalent to a two-layer, perceptron model of the neural network, which is used as an activation function for artificial neurons.

![K (x, y) = tanh (\gamma.{x^T y}+{r}) ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-4a5b8fb12ec6a016f21ec597af935205_l3.png "Rendered by QuickLaTeX.com")  
 

![](https://media.geeksforgeeks.org/wp-content/uploads/20200515150022/sigmoid.jpg)

**Sigmoid Kernel Graph**

  
**Code:**   
 

python3
-------

`from` `sklearn.svm` `import` `SVC`

`classifier` `=` `SVC(kernel` `=``'sigmoid'``)`

`classifier.fit(x_train, y_train)`

*   **Polynomial Kernel:** It represents the similarity of vectors in the training set of data in a feature space over polynomials of the original variables used in the kernel.

![K (x, y) = tanh (\gamma.{x^T y}+{r})^d, \gamma>0 ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-5ff556b17c4ddba95f4dd9e186cde03e_l3.png "Rendered by QuickLaTeX.com")  
 

![](https://media.geeksforgeeks.org/wp-content/uploads/20200515150200/polynomial.jpg)

**Polynomial Kernel Graph**

  
**Code:**

python3
-------

`from` `sklearn.svm` `import` `SVC`

`classifier` `=` `SVC(kernel` `=``'poly'``, degree` `=` `4``)`

`classifier.fit(x_train, y_train)`

*   [**Linear Kernel:**](https://www.geeksforgeeks.org/creating-linear-kernel-svm-in-python/) used when data is linearly separable.

**Code:** 

python3
-------

`from` `sklearn.svm` `import` `SVC`

`classifier` `=` `SVC(kernel` `=``'linear'``)`

`classifier.fit(x_train, y_train)`

  
  


