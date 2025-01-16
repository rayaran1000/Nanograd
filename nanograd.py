import math

class Item:
    """Stores scaler values object along with child nodes, operators and gradient"""

    def __init__(self,data,child_nodes=(),operator='',label=''):
        self.data = data
        self.child_nodes = set(child_nodes)
        self.operator = operator
        self.label = label
        self.grad = 0.0
        self.backprop = lambda : None

    def __repr__(self):
        return f"Item(data={self.data})"

    def __add__(self,other):
        other = other if isinstance(other,Item) else Item(other)
        out = Item((self.data + other.data),(self,other),'+')

        def backprop():
          self.grad += 1.0 * out.grad
          other.grad += 1.0 * out.grad

        out.backprop = backprop
        return out

    def __radd__(self,other):
        return self + other

    def __mul__(self,other):
        other = other if isinstance(other,Item) else Item(other)
        out =  Item((self.data * other.data),(self,other),'*')

        def backprop():
          self.grad += other.data
          other.grad += self.data

        out.backprop = backprop
        return out

    def __rmul__(self,other):
        return self * other

    def __neg__(self):
        return -1 * self

    def __sub__(self,other):
        other = other if isinstance(other,Item) else Item(other)
        return self + (-other)

    def __rsub__(self,other):
        return (-self) + other

    def __pow__(self,power):
        assert isinstance(power,(int,float))  # Power only supports ints and floats, so assert checks that(if fails, raises as assertion error)
        out = Item(self.data ** power,(self,),f'pow of {power}')

        def backprop():
          self.grad = power * (self ** (power - 1))*out.grad

        out.backprop = backprop
        return out

    def __truediv__(self,other):
        return self * (other**-1)

    def exp(self):
      out =  Item(math.exp(self.data),(self,),'exp')

      def backprop():
        self.grad += math.exp(self.data) * out.data

      out.backprop = backprop
      return out

    def tanh(self):
      x = self.data
      t = ((math.exp(2*x) - 1) / (math.exp(2*x) + 1))
      out = Item(t,(self,),'tanh') # (self,) -> its a tuple of 1 object

      def backprop():
        self.grad += (1 - (t**2)) * out.grad

      out.backprop = backprop
      return out

    def sigmoid(self):
      t = (math.exp(self.data)) / (1 + math.exp(self.data))
      out = Item(t,(self,),'sigmoid')

      def backprop():
        self.grad += t*(1-t)*out.grad

      out.backprop = backprop
      return out

    def relu(self):
      t = max(0,self.data)
      out = Item(t,(self,),'relu')

      def backprop():
        if t == 0:
          self.grad = 0
        else:
          self.grad += 1.0 * out.grad
      out.backprop = backprop
      return out

    def backward(self):
      topo = []
      visited = set()
      def build_topo(v):
        if v not in visited:
          visited.add(v)
          for child in v.child_nodes:
            build_topo(child)
          topo.append(v)
      build_topo(self)

      self.grad = 1.0
      for node in reversed(topo):
        node.backprop()