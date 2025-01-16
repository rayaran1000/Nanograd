# Nanograd

A tiny Autograd engine (inspired from Inspired from Andrej Karpathy's Micrograd). Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API. Both are tiny, with about 100 and 50 lines of code respectively. The DAG only operates over scalar values, so e.g. we chop up each neuron into all of its individual tiny adds and multiplies. Added some additional activation functions as well.


# Installation
```bash
pip install nanograd
```

### Example usage

Below is a slightly contrived example showing a number of possible supported operations:

```python
from nanograd import Item

a = Item(-4.0)
b = Item(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') 
g.backward()
print(f'{a.grad:.4f}') # the numerical value of dg/da
print(f'{b.grad:.4f}') # the numerical value of dg/db
```

### Training a neural net

The notebook `Notebook tutorial.ipynb` provides a full demo of training an 2-layer neural network (MLP). This is achieved by initializing a neural net from `nanograd.nn` module.