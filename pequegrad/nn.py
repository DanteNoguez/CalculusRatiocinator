import random
from pequegrad.engine import Valor

class Neurona:
  def __init__(self, nin):
    self.w = [Valor(random.uniform(-1, 1)) for n in range(nin)]
    self.b = Valor(random.uniform(-1,1))

  def __call__(self, x):
    # w * x + b
    activacion = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
    resultado = activacion.tanh()
    return resultado

  def parametros(self):
    return self.w + [self.b]

class Capa:
  def __init__(self, nin, nout):
    self.neuronas = [Neurona(nin) for n in range(nout)]

  def __call__(self, x):
    resultado = [neurona(x) for neurona in self.neuronas]
    return resultado[0] if len(resultado) == 1 else resultado

  def parametros(self):
    return [parametro for n in self.neuronas for parametro in n.parametros()]

class MLP: # Perceptrón multicapa
  def __init__(self, nin, nouts):
    tamaño = [nin] + nouts
    self.capas = [Capa(tamaño[i], tamaño[i+1]) for i in range(len(nouts))]

  def __call__(self, x):
    for capa in self.capas:
      x = capa(x)
    return x

  def parametros(self):
    return [parametro for capa in self.capas for parametro in capa.parametros()]
