# An example code for forward-propagating a single neuron might look as follows:

class Neuron(object):
    #...
    def forward(self, inputs):
        """ assume inputs and weights are 1-D numpy arrays and bias is a number """
        cell_body_sum = np.sum(inputs * self.weights) + self.biasfiring_rate = 1.0 / (1.0 + math.exp(-cell_body_sum)) #sigmoid activation fuction
        return firing_rate

# Example feed-forward computation

'''The full forward pass of this 3-layer neural network is then simply three matrix multiplications, interwoven with the application of the activation function:'''

# forward-pass of a 3-layer neural network:
f = lambda x: 1.0/(1.0 + np.exp(-x)) # activation function (use sigmoid)
x = np.random.randn(3,1) #random input vector of three numbers (3x1)
h1 = f(np.dot(W1, x) + b1) #calcuate first hidden layer activation
h2 = f(np.dot(W2, h1) + b2) # calcuate second hidden layer activaions
out = np.dot(W3, h2) + b3  # outpunteuron (1x1)


