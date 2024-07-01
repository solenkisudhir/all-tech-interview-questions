

### Deep Learning Activation Functions

#### 1. **Sigmoid Activation Function**
- **Definition and Purpose:** The sigmoid function maps input values to a range between 0 and 1. It's often used in the output layer of binary classification problems.
- **Mathematical Formulation:** \( \sigma(x) = \frac{1}{1 + e^{-x}} \)
- **Advantages:** Smooth gradient, outputs probabilities, and works well for binary classification.
- **Disadvantages:** Vanishing gradient problem, outputs not zero-centered.
- **Example (Python):**
  ```python
  import numpy as np

  def sigmoid(x):
      return 1 / (1 + np.exp(-x))
  
  x = np.array([-1.0, 0.0, 1.0])
  print(sigmoid(x))
  ```

#### 2. **Hyperbolic Tangent (tanh) Activation Function**
- **Definition and Purpose:** The tanh function maps input values to a range between -1 and 1. It's often used in hidden layers of neural networks.
- **Mathematical Formulation:** \( \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \)
- **Advantages:** Outputs zero-centered, stronger gradients than sigmoid.
- **Disadvantages:** Vanishing gradient problem.
- **Example (Python):**
  ```python
  def tanh(x):
      return np.tanh(x)
  
  print(tanh(x))
  ```

#### 3. **Rectified Linear Unit (ReLU) Activation Function**
- **Definition and Purpose:** The ReLU function outputs the input directly if it is positive; otherwise, it outputs zero. It's widely used in hidden layers of neural networks.
- **Mathematical Formulation:** \( \text{ReLU}(x) = \max(0, x) \)
- **Advantages:** Computationally efficient, mitigates vanishing gradient problem.
- **Disadvantages:** Dying ReLU problem (neurons can "die" during training).
- **Example (Python):**
  ```python
  def relu(x):
      return np.maximum(0, x)
  
  print(relu(x))
  ```

#### 4. **Leaky ReLU Activation Function**
- **Definition and Purpose:** The Leaky ReLU function allows a small, non-zero gradient when the input is negative, preventing neurons from "dying."
- **Mathematical Formulation:** \( \text{Leaky ReLU}(x) = \max(0.01x, x) \)
- **Advantages:** Prevents dying ReLU problem, retains benefits of ReLU.
- **Disadvantages:** Introduces a small slope for negative inputs, which may still cause issues.
- **Example (Python):**
  ```python
  def leaky_relu(x, alpha=0.01):
      return np.where(x > 0, x, alpha * x)
  
  print(leaky_relu(x))
  ```

#### 5. **Parametric ReLU (PReLU) Activation Function**
- **Definition and Purpose:** PReLU is a variant of Leaky ReLU where the slope of the negative part is learned during training.
- **Mathematical Formulation:** \( \text{PReLU}(x) = \max(\alpha x, x) \), where \(\alpha\) is a learned parameter.
- **Advantages:** Adaptable to data, mitigates dying ReLU problem.
- **Disadvantages:** Increased computational complexity due to learning additional parameters.
- **Example (Python):**
  ```python
  def prelu(x, alpha):
      return np.where(x > 0, x, alpha * x)
  
  alpha = 0.1  # Example learned parameter
  print(prelu(x, alpha))
  ```

#### 6. **Exponential Linear Unit (ELU) Activation Function**
- **Definition and Purpose:** ELU is similar to ReLU but tends to converge faster and produce more accurate results because it has a smooth curve for negative values.
- **Mathematical Formulation:** \( \text{ELU}(x) = \begin{cases} 
x & \text{if } x > 0 \\
\alpha (e^x - 1) & \text{if } x \leq 0 
\end{cases} \)
- **Advantages:** Smooth gradient, zero-centered, alleviates vanishing gradient problem.
- **Disadvantages:** More computationally expensive than ReLU.
- **Example (Python):**
  ```python
  def elu(x, alpha=1.0):
      return np.where(x > 0, x, alpha * (np.exp(x) - 1))
  
  print(elu(x))
  ```

#### 7. **Swish Activation Function**
- **Definition and Purpose:** Swish is a smooth, non-monotonic function that can outperform ReLU on deep networks.
- **Mathematical Formulation:** \( \text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}} \)
- **Advantages:** Smooth gradient, improved performance on deep networks.
- **Disadvantages:** More computationally expensive.
- **Example (Python):**
  ```python
  def swish(x):
      return x * sigmoid(x)
  
  print(swish(x))
  ```

#### 8. **Softmax Activation Function**
- **Definition and Purpose:** The softmax function is used in the output layer of a neural network for multi-class classification. It converts logits into probabilities.
- **Mathematical Formulation:** \( \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}} \)
- **Advantages:** Outputs probabilities, useful for multi-class classification.
- **Disadvantages:** Sensitive to outliers, can saturate.
- **Example (Python):**
  ```python
  def softmax(x):
      exp_x = np.exp(x - np.max(x))  # stability improvement
      return exp_x / np.sum(exp_x, axis=0)
  
  print(softmax(x))
  ```

### Summary:
- **Sigmoid and tanh**: Useful for binary classification and hidden layers, but suffer from vanishing gradient.
- **ReLU and its variants (Leaky ReLU, PReLU, ELU)**: Popular for hidden layers due to their ability to handle the vanishing gradient problem.
- **Swish**: A newer activation function showing promise in deep networks.
- **Softmax**: Essential for multi-class classification, converting logits to probabilities.

Understanding these activation functions and their characteristics is crucial for building and optimizing deep neural networks. Each function has its own strengths and weaknesses, and the choice of activation function can significantly impact the performance of a model.
