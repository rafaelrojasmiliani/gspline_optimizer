
import numpy as np
from ..mathtools import pwP6_diff
from constraints import cBilateralConstraint


class cdtConstraint(cBilateralConstraint):
  """
  Given a piecewise 5th order polynomial defined by N 5th order polynomials in
  N intervals with a canonical description in x it computes the dot product
  between the jumps in the 5th derivative and the velocity at the boundaries
  between intervals. Each polynomial is supposed to be at least 4th order
  continuous.
  """
  def __init__(self,_N,_dim):
    """
      Parameters:
      ----------
        _N: unsigned integer
          Number of intervals where the polynomial is defined.
        _dim: unsigned integer
          Dimension of the ambient space.
    """
    self.N_ = _N
    self.dim_   = _dim      # dimension of the ambient space 
    self.qd5R_ = np.zeros((_dim,))
    self.qd5L_ = np.zeros((_dim,))
    self.qd_ = np.zeros((_dim,))
    super(cdtConstraint,self).__init__(len_=_N-1)

  def value(self,x,res):
    """
      Returns the value of the constraint: the dot product between the jump in
      the 5th order derivative and the velocity at every time ti, i.e. the
      boundary between the intervals where the piecewise polynomial is defined.

      Parameters:
      ----------
        x: np.array, float
          Canonical representation of the piecewise 5th order polynomial.
        res: np.array, float
          Buffer to store the result.
    """
    tiarray = []
    ti=0.0
    for tau in x[-self.N_:-1]:
      ti += tau
      tiarray.append(ti)
    yiarray = [] # components of the curve
    for idim in range(0,self.dim_):
      # yi contains the coefficients of the polynomials describing
      # the idim component of q(t) (self.N_ 5th order polynomials)
      yi = [x[idim*6+i*6*self.dim_:idim*6+i*6*self.dim_+6] for i in range(0,self.N_)]
      yi = np.hstack(yi)
      # xi is the canonical description of the one-dimensional curve q_i(t)
      xi = np.hstack([yi,x[-self.N_:]])
      yiarray.append(xi)

    y=np.vstack(yiarray)
    
    
    for k,ti in enumerate(tiarray):
      for i in range(0,self.dim_):
        self.qd5R_[i]=  pwP6_diff(ti-1.0e-6,y[i,:],self.N_,5)
        self.qd5L_[i]=  pwP6_diff(ti+1.0e-6,y[i,:],self.N_,5)
        self.qd_[i]=  pwP6_diff(ti,y[i,:],self.N_,1)

      res[k] = (self.qd5R_-self.qd5L_).dot(self.qd_)

