# Notes for CS231n_Stanford

## Loss Function
batch normalization is one way of regularization?
dropout

### hinge loss vs softmax loss 

hinge loss: look at the margin of the scores of correct class and scores of incorrect class
softmax loss: compute the probability distribution and look at the minus log probability 


gradient descent algorithm: 
- learning rate - hyper parameter is always the first thing to check?


## Backpropagation

Chain rule

when calcuate gradients, magnitude matters 

* The gradient of a vector is always going to be the same size as the original vector, and each element is gradient is going to mean how much of that particular element is going to affect final output of function
