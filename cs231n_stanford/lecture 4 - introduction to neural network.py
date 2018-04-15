# backprop for this neuron in code

w = [2,-3.-3] # assume some random weights and data
x = [-1,-2]

# forward pass
dot = w[0] * x[0] + w[1] * x[1] + w[2]
f = 1.0 / (1 + math.exp(-dot)) # sigmoid function

# backward pass through the neuron (backprogagation)
ddot = (1-f) * f # gradient on dot variable, using the sigmoid gradient derivation
dx =[w[0] * ddot, w[1] * ddot] #backprop into x
dw = [x[0] * ddot, x[1] * ddot, 1.0 * ddot] # backrpop into y

# backward pass through the neuron (backpropagation)
ddot = (1 - f) * f # gradient on dot variable, using the sigmoid gradient derivation
dx = [w[0] * ddot, w[1] * ddot] # backprop into x
dw = [x[0] * ddot, x[1] * ddot, 1.0 * ddot] # backprop into w
# we're done! we have the gradients on the inputs to the circui



# backprop in practice: Staged computation

x = 3
y = -4

# forward pass
sigy = 1.0 / (1 + math.exp(-y)) # sigmoid in numerator
num = x + sigy # numerator
sigx = 1.0 / (1 + math.exp(-x)) # sigmoid in dneomniator
xpy  = x + y
xpyspr = xpy **2
den = sigx + xpysqr # denominator
invden = 1.0 / den
f = num * invden # done

'''every single piece in our backprop will invovlve computiog the local gradient of that expression, and chaining it with the gradient 
on that expression with multiplication.'''

# backprop f = num * invden
dnum = invden # gradient on numerator
dinvden = num 
# backprop invden = 1.0 / den
dden = (-1.0 / (den ** 2)) * dinvden
# backprop den = sigx + xpysqr
dsigx = (1) * dden
dxpysqr = (1) * dden
# backprop xpysqr = xpy**2
dxpy = (2 * xpy) * dxpysqr
# backprop xpy = x + y
dx = (1) * dxpy
dy = (1) * dxpy
# backprop sigx = 1.0 / (1 + math.exp(-x))
dx += ((1-sigx) * sigx) * dsigx
# backprop num = x + sigy
dx += (1) * dnum
dsigy = (1) * dnum
# backprop sigy = 1.0 / (1 + math.exp(-y))
dy += （（1-sigy) * sigy) * dsigy
# done! phew


# Gradients for vectorized operations

Matrix-Matrix multiply gradient
# forward pass
W = np.random.randn(5,10)
X = np.random.randn(10,3)
D = W.dot(X)

# now suppose we had the gradient on D from above in the circuit

dD = np.random.randn(*D.shape)
dW = dD.dot(X.T)
dX = W.T.dot(dD)



# forward pass and backward pass API

#Graph (or Net) object (rough psuedo code)

class ComputationalGraph(object):
    #...
    def forward(inputs):
        # 1. [pass inputs to input gates...]
        # 2. forward the computational graph:
        for gate in self.graph.nodes_topologically_sorted():
            gate.forward()
        return loss # the final gate in the graph outputs the loss

    def backward():
        for gate in reversed(self.graph.nodes_topologically_sorted()):
            gate.backward() # little piece of backprop (chain rule applied)

        return inputs_gradients





class MultiplyGate(object):
    def forward(x,y):
        z = x * y
        self.x = x # must keep these around!
        self.y = y
        return z
    
    def backward(dz):
        dx = self.y * dz #[dz/dx * dL/dz]
        dy = self.x * dz #[dz/dy * dL/dz]
        return [dx, dy]



# Example feed-foward computation of a neural network

class Neuron:
    # ...
    def neuron_tick(inputs):
        """ assume inputs and weights are 1-D numpy arrays and bias is a number """
        cell_body_sum = np.sum(inputs * self.weights) + self.bias
        firing_rate = 1.0 / (1.0 + math.exp(-cell_body_sum))
        return firing_rate



# forward-pass of a 3-layer neural network:

f = lambda x: 1.0/(1.0 + np.exp(-x)) #activation function (sue sigmoid)
x = np.random.randn(3,1) # random input vector of three numbers (3x1)
h1 = f(np.dot(W1, x + b1)) # calcuate first hidden layer activations (4x1)
h2 = f(np.dot(W2, h1 + b2)) # calcuate second hidden layer activations (4x1)
out = np.dot(W3, h2) + b3 #output neuron (1x1)