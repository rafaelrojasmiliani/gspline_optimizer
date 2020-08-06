

import numpy as np

class cCostraintsContainer(list):
  """
    This a class intended to store different kinds of constraints for
    optimization problems using numpy. The idea is to allow
    to program each constraint separately as a single class which
    implement the following methods:
      - value
      - first derivative (Jacobian)
      - __len__ number of equations that constitute the constraint.
    This class consents to add constraints to a list and  implements
    the construction of the Jacobian and the value vector by adding
    the single values and Jacobians of each single constraint.
  """
  def __init__(self,**kwargs):
    self.n_ = kwargs['_n']
    pass

  def allocateBuffers(self):
    """
      Allocate the memory buffers used to store the value of the constraint and
      the jacobian/gradient/derivative of the constraints.
    """
    self.jacbuff = np.zeros((len(self),self.n_))
    self.valbuff = np.zeros((len(self),))

  def __len__(self):
    """
      return the number of contraints
    """
    return sum([len(c) for c in self])

  def jacobian(self,x):
    self.jacbuff.fill(0.0)
    i0 = 0
    i1 = 0
    for c in self:
      i1 =  i0+len(c)
      c.diff_1(x,self.jacbuff[i0:i1,:])
      i0 += len(c)
    return self.jacbuff

  def value(self,x):
    self.valbuff.fill(0.0)
    i0 = 0
    i1 = 0
    for c in self:
      i1 =  i0+len(c)
      c.value(x,self.valbuff[i0:i1])
      i0 += len(c)
    return self.valbuff
