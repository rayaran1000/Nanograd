import random
from nanograd import Item

class Neuron:

    """Defines single neuron of the MLP"""

    def __init__(self,nin): # nin -> number of inputs
        self.w = [Item(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Item(random.uniform(-1,1))

    def __call__(self,x): # x -> input
        out = sum([wi * xi for wi,xi in zip(self.w,x)], self.b)
        out_act = out.tanh()
        return out_act

    def parameters(self):
        return self.w + [self.b]

class Layer:

    """Defines a single layer in a Neural Network"""

    def __init__(self,nin,nout): 
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self,x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [parameters for neuron in self.neurons for parameters in neuron.parameters()]

class MLP:
  
    """Defines a Multi layered Perceptron network"""

    def __init__(self,nin,nout):
        total_inputs = [nin] + nout
        self.layers = [Layer(total_inputs[i],total_inputs[i+1]) for i in range(len(nout))]

    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
            return x

    def parameters(self):
        return [parameter for layer in self.layers for parameter in layer.parameters()]

