
import math

class Valor:
  def __init__(self, valor, _valores=(), _op='', etiqueta=''):
    self.valor = valor
    self.grad = 0.0 # gradiente
    self._backward = lambda: None # el valor predeterminado es nulo
    self._previos = set(_valores)
    self._op = _op
    self.etiqueta = etiqueta

  def __repr__(self):
    return f'Valor({self.valor})'

  def __add__(self, otro): # self + otro
    otro = otro if isinstance(otro, Valor) else Valor(otro)
    resultado = Valor(self.valor + otro.valor, (self, otro), '+')

    def _backward():
      self.grad += 1.0 * resultado.grad
      otro.grad += 1.0 * resultado.grad
    resultado._backward = _backward

    return resultado

  def __radd__(self, otro):
    return self + otro

  def __mul__(self, otro): # self * otro
    otro = otro if isinstance(otro, Valor) else Valor(otro)
    resultado = Valor(self.valor * otro.valor, (self, otro), '*')

    def _backward():
      self.grad += otro.valor * resultado.grad
      otro.grad += self.valor * resultado.grad
    resultado._backward = _backward

    return resultado

  def __pow__(self, otro): # self ** otro
    assert isinstance(otro, (int, float))
    resultado = Valor(self.valor**otro, (self,), f'**{otro}')

    def _backward():
      self.grad += otro * (self.valor ** (otro - 1)) * resultado.grad
    resultado._backward = _backward

    return resultado

  def __rmul__(self, otro): # self * otro
    return self * otro

  def __truediv__(self, otro): # self / otro
    return self * otro**-1

  def __neg__(self):
    return self * -1

  def __sub__(self, otro): # self - otro
    return self + (-otro)

  def __rsub__(self, otro):
    return otro + (-self)

  def __rtruediv__(self, otro):
    return otro * self**-1

  def tanh(self): # tangente hiperbólica
    x = self.valor
    t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
    resultado = Valor(t, (self, ), 'tanh')

    def _backward():
      self.grad += (1 - t**2) * resultado.grad #según la fórmula 1-tanh**2x
      
    resultado._backward = _backward
    
    return resultado

  def exp(self):
    x = self.valor
    resultado = Valor(math.exp(x), (self, ), 'exp')

    def _backward():
      self.grad += resultado.valor * resultado.grad
    resultado._backward = _backward

    return resultado

  def backward(self):
    topo = [] # haremos un ordenamiento topológico de los valores (los pondremos en orden 'cronológico')
    visitados = set()
    def construir_topo(v):
      if v not in visitados:
        visitados.add(v)
        for valores in v._previos:
          construir_topo(valores)
        topo.append(v)
    construir_topo(self)

    self.grad = 1.0 # asignamos valor de 1 para inicializar la propagación

    for nodo in reversed(topo): # comenzamos desde adelante hacia atrás
      nodo._backward()

  #def __repr__(self):
    return f'(valor={self.valor}, gradiente={self.grad})'
