## Summarization of Deep learning model building  


### 1. Data set processing

input: data set, output: train/training dev/dev/test  

* Small data set (~2000): train (60%)/ dev(20%)/test (20%) 
* Large data set (~10,000): increase sample size in train, decrease size in dev/test sets.   

But make sure dev/test sets coming from the same distribution.

1. For **mismatch data distribution** between the training and development/testing sets:  
For example: data available is different from data targeted.   
Train set need to separate _one more small set for variance evaluation_, called training dev set.    

2. For **mislabeled samples** (ground truth is wrong):  
Need to distinguish between _random mislabeled_ or _systematically error_.   
If manually adjusted, make sure dev and test sets are still representing the future data set and have same distribution.

### 2. Train to batch 

input: train set, output: batch or mini-batch  

If train set is big, train samples usually shuffled and partitioned into small **mini-batches**.   
For each mini-batch, one loop is executed as one batch situation. Batch norm is also applied within current running batch/mini-batch. The size of mini-batch need to be determined here, usually is the power of 2. Fit the computer memory.
Once all samples runned, one epoch is there. Some models applied thousand epoches to train NN longer time.

### 3. Build model

input: batch or mini-batch, output: model with parameters  

1. Input data normalization.   
For image data, flatten data, normalize by number of pixels, and convert softmax labels if necessary.

2. NN architecture.   
Number of layers needed (can be tuned), number of units for each hidden layer (can be tuned), activation function for each layer (may need longer time to find the best combination).
3. Initialize parameters.   
Three initializations: zeros, randoms, He, Xavier. 
Usually weights prefer He while the intercept (b) is fine with zeros. Make sure the variable dimension is correct. Weights: small, random, and variance controlled.

4. Forward propagation.   
Dropout needs to be applied here.   
Batch norm should be applied here on z, before activation function: to reduce the effect of any changes of early layer to the next layer. 

5. Compute cost.   
Sometimes, combined with the forward propagation together. L2-regularization needs to be applied here, lambda which control the regularization needs to be tuned.

6. Backward propagation. 
Dropout, L2 regularization, Early stopping need to be applied here. 

7. Update parameters.   
Learning rate need to be tuned, learning rate decay can be applied to avoid big steps in later phase.   
**Optimization algorithm** can be applied here to improve the gradient descent process, including: momentum, RMSprop, and Adam. For each optimizer, need to initialize the optimization variables. Those variables also need to be updated along the layers. Can be initialized as zero.

8. Predict model

### 4. With build model, error analysis.

Determine which is the most significant problem needs to be solved: **bias or variance**.
High bias: train longer, bigger network, new NN architecture (including activation function, # hidden units, and et al.)
High variance: more data, NN architecture.
