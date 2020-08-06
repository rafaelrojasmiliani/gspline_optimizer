
from constraints import cBilateralConstraint,cAffineBilateralConstraint
from mathtools import getDmatLeg, getPLeg
from numpy.linalg import matrix_power
import numpy as np

import sys 

  

class ccStationarity_dt(cBilateralConstraint):
  """
      Class for the scalar constraint generated
      by the stationarity of the variation of
      the time for the minimization of the
      jerk problem.
  """
  def __init__(self,**kwargs):
    """
      Paramenters
      -----------
        N: unsigned int
          Number of intervals
    """

    self.N=N = kwargs['N']
    kwargs['len_']=N-1
    self.pdof = 6*N
    self.tdof = N
    self.n    = self.pdof + self.tdof

    super(ccStationarity_dt,self).__init__(**kwargs)
    Pl=self.Pl = getPLeg(-1.0,6)
    Pr=self.Pr = getPLeg(1.0,6)

    D =self.D  = getDmatLeg(6)
    D2= self.D2 = matrix_power(D,2) 
    D3= self.D3 = matrix_power(D,3) 
    D4= self.D4 = matrix_power(D,4) 
    D5= self.D5 = matrix_power(D,5) 

    Rl=self.Rl = np.tensordot(Pl,Pl,axes=0)
    Rr=self.Rr = np.tensordot(Pr,Pr,axes=0)

    self.Fl = -D3.dot(Rl).dot(D3.transpose())\
              +D4.dot(Rl).dot(D2.transpose())\
              -D5.dot(Rl).dot(D.transpose())

    self.Fr = -D3.dot(Rr).dot(D3.transpose())\
              +D4.dot(Rr).dot(D2.transpose())\
              -D5.dot(Rr).dot(D.transpose())

    self.Gl = self.Fl.transpose() + self.Fl
    self.Gr = self.Fr.transpose() + self.Fr


  def value(self,x,res):
    for i in range(0,self.N-1):
      ipl=6*i
      ipc=6*(i+1)
      ipr=6*(i+2)
      xl =x[ipl:ipc]
      xr =x[ipc:ipr]
      tl = x[self.pdof+i]
      tr = x[self.pdof+i+1]
      res[i] = xl.transpose().dot(self.Fr).dot(xl)*pow(tr,6)\
               -xr.transpose().dot(self.Fl).dot(xr)*pow(tl,6)
               
  def diff_1(self,x,res):
    for i in range(0,self.N-1):
      ipl=6*i
      ipc=6*(i+1)
      ipr=6*(i+2)
      xl =x[ipl:ipc]
      xr =x[ipc:ipr]
      itl=  self.pdof+i
      itr=  self.pdof+i+1
      tl = x[itl]
      tr = x[itr]
      xrFlxr = xr.dot(self.Fl).dot(xr)
      Glxr = self.Gl.dot(xr) 
      Grxl = self.Gr.dot(xl) 
      xlFrxl = xl.dot(self.Fr).dot(xl)

      res[i,ipl:ipc]    =  Grxl*pow(tr,6)
      res[i,ipc:ipr]    = -Glxr*pow(tl,6)
      res[i,itl]        = -6.0*xrFlxr*pow(tl,5)
      res[i,itr]        =  6.0*xlFrxl*pow(tr,5)


  def diff_2(self,idx,x,res):
    pass
  def diff_2_contraction(self,x,lam,res):
    #self.res.fill(0.0)
    pass

class cConstraintContinuity(cBilateralConstraint):
  """
      Class representing the condition of continuity of a piece-wise polynomial
      of degree 6 divided in N intervals.  Such a constraint states that the
      given derivates must be equal at each side of the bounadary between
      intervals.  For a scallar function q:R->R we have

  """
  def __init__(self,**kwargs):
    """
      Paramenters
      -----------
        N: unsigned int
          Number of intervals
        ders: array of unsigned ints
          Array containing the derivaties that we ask to be continuous.  0
          means C^0
    """
    N = kwargs['N']
    self.N=N
    self.pdof = 6*N
    self.n = 7*N
    self.tdof = N
    self.ders = kwargs['ders']

    kwargs['len_']=(N-1)*len(self.ders)

    super(cConstraintContinuity,self).__init__(**kwargs)
    Pl=getPLeg(-1.0,6)
    Pr=getPLeg( 1.0,6)

    D1 =getDmatLeg(6)
    self.PrD= [matrix_power(D1,d).dot(Pr) for d in self.ders]
    self.PlD= [matrix_power(D1,d).dot(Pl) for d in self.ders]


  def value(self,x,res):
    i0 = 0
    for i in range(0,self.N-1):
      for di,d in enumerate(self.ders):
        xl =x[6*i:6*(i+1)]
        xr =x[6*(i+1):6*(i+2)]
        tl = x[self.pdof+i]
        tr = x[self.pdof+i+1]
        res[i0] = xl.transpose().dot(self.PrD[di])*pow(tr,d)\
                -xr.transpose().dot(self.PlD[di])*pow(tl,d)
        i0+=1
               
  def diff_1(self,x,res):
    i0=0
    for j in range(0,self.N-1):#loop on the intervals
      jl=6*j    # Left
      jc=6*(j+1)# Center
      jr=6*(j+2)# Right
      jtl = self.pdof + j
      jtr = self.pdof + j+1
      tl = x[jtl]   # Left  interval
      tr = x[jtr] # Right interval
      xl =x[jl:jc]
      xr =x[jc:jr]
      for id_,d in enumerate(self.ders):
        res[i0,jl:jc] =  self.PrD[id_]*pow(tr,d)
        res[i0,jc:jr] = -self.PlD[id_]*pow(tl,d)
        if d>0:
          res[i0,jtr] =  d*self.PrD[id_].dot(xl)*pow(tr,d-1)
          res[i0,jtl] = -d*self.PlD[id_].dot(xr)*pow(tl,d-1)
        i0+=1

  def diff_2(self,idx,x,res):
    pass
  def diff_2_contraction(self,x,lam,res):
    #self.res.fill(0.0)
    pass

class cConstraintBoundaryInternal(cAffineBilateralConstraint):
  """
      Class representing the condition of fixed values at the
      boundaries of position, 1,2 and 3th derivatives (which are
      zero), and fixed values of position at the boundaries
      between intervals of a piece-wise polynomial of degree 6
      divided in N intervals.
  """
  def __init__(self,**kwargs):
    """
      Paramenters
      -----------
        p: array of double
            array containing the points to attain
    """
    self.points_ = kwargs['p']
    N= self.N = len(self.points_)-1   # Number of intervals
    pdof=self.pdof = 6*N # Polynomial dof
    tdof=self.tdof = N   # Time dof
    n = self.n = pdof+ tdof  # Total dof

    Pl=getPLeg(-1.0,6)  
    Pr=getPLeg( 1.0,6)

    D1 =getDmatLeg(6)
    DPl = D1.dot(Pl)
    DPr = D1.dot(Pr)
    D2Pl = matrix_power(D1,2).dot(Pl)
    D2Pr = matrix_power(D1,2).dot(Pr)

    lm = 2*(N-1)+6
    A0=np.zeros((lm,n))
    b0=np.zeros((lm,))

    self.tdof = N
    
    A0[0,:6] = Pl
    A0[1,:6] = DPl
    A0[2,:6] = D2Pl

    b0[0]    = self.points_[0]
    b0[1]    = 0.0
    b0[2]    = 0.0

    A0[-3 ,pdof-6:pdof] = Pr
    A0[-2 ,pdof-6:pdof] = DPr
    A0[-1 ,pdof-6:pdof] = D2Pr

    b0[-3]    = self.points_[-1]
    b0[-2]    = 0.0
    b0[-1]    = 0.0

    i=3
    for ip,p in enumerate(self.points_[1:-1]):
      A0[i,6*ip:6*(ip+1)]   = Pr
      ip+=1
      A0[i+1,6*ip:6*(ip+1)] = Pl
      b0[i]  =p
      b0[i+1]=p
      i+=2

#    print('------------   b0 ----------------------')
#    print(b0)
#    print('------------   A0 ----------------------')
#    print(A0)
#    print('------------------')
    b0 = -b0
    super(cConstraintBoundaryInternal,self).__init__(A=A0,b=b0)



if __name__ == '__main__':
  np.set_printoptions(threshold=np.nan,linewidth=500,precision=3,suppress=True)
  np.seterr(all='raise')
  N=2
  c = cConstraintContinuity(N=N,ders=[0,1,2,3])
  jac = np.zeros((len(c),7*N))
  val = np.zeros((len(c),))
  x = np.ones((7*N,))

  c.diff_1(x,jac)
  c.value(x,val)

  print(jac)
  print(val)
