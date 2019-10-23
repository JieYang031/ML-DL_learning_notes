### Optimization methods in Andrew Ng Course:
* Gradient descent with Momentum
* RMSprop
* Adam

Common used method for gradient descent (make it faster and fit for the **mini-batch** usage).

#### 1. Gradient descent with Momentum 

> for each iteration i  
V_dw = Beta * V_dw + (1 - Beta) * dw  
V_db = Beta * V_db + (1 - Beta) * db  
W = W - learning_rate * V_dw    
B = B - learning_rate * V_db  

It helps the gradient descent faster and more efficient.  
It reduces the change of one direction and increase the change of another direction. In this way, it can reduce the change of useless direction and increase the change of useful direction.  

Notes:  
1. Tune hyperparameter alpha (the learning_rate) and beta, usually Beta = 0.9.  
2. Usually do not bias correction. If do, use the V_dw/(1-beta_t) for initial phase.  
3. Initialize V_dw and V_db as vectorized 0, the same shape as dw and db.  

Such as the image below, that by increasing the red arrow towards center of the contour and decreasing the other red arrow can help the gradient descent faster.  
![GD file](/img/GD.pdf) 

##### 2. RMSprop

> for each iteration i . 
S_dw = Beta2 * S_dw + (1 - Beta2) * dw^2    
S_db = beta2 * S_db + (1 - Beta2) * db^2  
W = W - learning_rate * sb/sqrt(S_dw)  
B = B - learning_rate * db/sqrt(S_db)  

In order to make sure S_dw or S_db not equal to zero, can add a small error such as 10e-8


#### 3. Adam Optimization
Adam = adaptive momentum estimation.

> for each iteration i:   
V_dw = beta1 * V_dw + (1 - beta1) * dw   
V_db = beta1 * V_db + (1 - beta1) * db   
S_dw = beta2 * S_dw + (1- beta2) * square (dw)   
S_db = beta2 * S_db + (1- beta2) * square (db)   
V_dw_corrected = V_dw/(1 - beta1^t)   
V_db_corrected = V_db/ (1- beta1^t)   
S_dw_corrected = S_dw/(1 - beta2^t)   
S_db_corrected = S_db/(1 - beta2^t)   
W = W - alpha (learning rate) * V_dw_corrected / sqrt (S_dw_corrected + sigma)   
B = B - alpha (learning rate) * V_db_corrected / sqrt (S_db_corrected + sigma)   

Hyperparameters and their choices:   
Alpha: learning rate, need to be tuned   
Beta1: default = 0.9   
Beta2: default = 0.99   
Sigma: default = 10e-8   

#### Final Notes:
There is another method we commonly used in GD (gradient descent) is called **Rprop** or resilient backpropagation.  
1. for full-batch optimization.
2. RMSprop can be treated as the adaptation of Rprop for min-batch learning.
3. Rprop is fist-order optimization algorithm. It consider the sign of partial derivative.

``` python
for t in range(num_iterations):
  dw[t] = compute_gradient(x, y)  
  if dw[t] * dw[t-1] >0:
    step_size = min(step_size * increasingFactor, step_size_max)
  else if dw[t] * dw[t-1] <0:
    step_size = man(step_size * decreasingFactor, step_size_min)
  
  w[t] = w[t] - sign(dw[t]) * step_size
```
