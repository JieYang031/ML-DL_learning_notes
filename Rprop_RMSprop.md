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
