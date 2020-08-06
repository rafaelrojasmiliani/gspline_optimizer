import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfromroots
from . import cBilateralConstraint

from polabsmax.polabsmax import cPolAbsMax


def timeMap(tau, a, b):
    return 0.5 * (1.0 - tau) * a + 0.5 * (1.0 + tau) * b


class cMaxVelocityConstraint(cBilateralConstraint):
    """
      Represets the constraints in the max of the absolute value of the
      derivative of a 5-th order polynomial. This is implemented with a
      function which gives the maximum of a 4th polynomial analysing the roots
      its 3th order polynomial derivative.
    """

    def __init__(self, _max, _N):
        kwargs = dict()
        kwargs['len_'] = 1
        super(self.__class__, self).__init__(len_=1, tol=_max)

        self.pmrc = cPolAbsMax(deg=4, interval=[-1.0, 1.0])
        #print("cMaxVelocityConstraint kwargs = ",kwargs)
        self.max_ = _max
        self.N_ = _N
        self.pdof_ = 6 * _N  # Polynomial dof
        self.tdof_ = _N  # Time dof

        self.dPidTj = np.matrix(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
             [-1.0 / 2.0, 0.0, 3.0 / 2.0, 0.0, 0.0,
              0.0], [0.0, -3.0 / 2.0, 0.0, 5.0 / 2.0, 0.0,
                     0.0], [3.0 / 8.0, 0.0, -30.0 / 8.0, 0.0, 35.0 / 8.0, 0.0],
             [0.0, 15.0 / 8.0, 0.0, -70.0 / 8.0, 0.0, 63.0 / 8.0]])
        self.dTidPj = np.linalg.inv(self.dPidTj)
        self.y = np.zeros((6 * self.N_, ))

    def value(self, x, res):
        """
          Returns the value of the maximum absolute value
          attained by the piece wise 5th order polynomial
          presented by x
        """
        max_, _, _, _, _ = self.getMax(x)
        res[:] = max_

    def getMax(self, x):
        """
          Get the position and the signed value of the max-abs of the
          piece-wise 5th order polynomial described by x.

          Paramenters:
          ----------
            x: np.array of floats
              Coefficients of the polynomial in Taylor basis

          Returns:
          -------
            max_: float
              siged abs-max
            idxI_: uint
             interval where the abs-max is attained.
            r_: float
             value the place of the interval where the maximum is attained.
             this is paramentrized to the interval  [-1,1]
            idxI_: uint
               a number indicating where the maximum is attained.
              -2 is left boundary, -1 is right boundary, and a value
              grater than 0 is a root of its derivative.
        """
        #    print('-..----------------------------++++++')
        #    print('in getMax jacobian shape of x ',x.shape)
        #    print('-..----------------------------++++++')
        # First we get the derivative of x w.r.t. the Taylor basis.
        # so y is the polynomial representing the derivative of x
        self.y = self.changeBaseLeg2TayAndDiff(x)
        # now we get the maximum derivative at the first interval
        max_, r_, idxR_ = self.pmrc.getAbsMaxWithSign(
            self.y[0:5], _root=True, _rootidx=True)
        rp_ = r_
        a = 0.0
        b = x[self.pdof_]
        r_ = timeMap(r_, a, b)
        idxI_ = 0
        max_ *= 2.0 / x[self.pdof_]
        # print(self.y[0:6])
        #print('val= %f, r = %f, idxR = %f'%(max_,r_,idxR_))
        # We look of the derivative at all intervals
        for j in range(1, self.N_):
            jl = 6 * j  # Left
            jr = 6 * (j + 1)  # Right
            val, r, idxR = self.pmrc.getAbsMaxWithSign(
                self.y[jl:jr - 1], _root=True, _rootidx=True)
            rp = r
            a = b
            b += x[self.pdof_ + j]
            r = timeMap(r, a, b)
            val *= 2.0 / x[self.pdof_ + j]
            # print(self.y[jl:jr])
            #print('val= %f, r = %f, idxR = %f'%(val,r,idxR))
            if abs(val) > abs(max_):
                max_ = val
                idxI_ = j
                r_ = r
                rp_ = rp
                idxR_ = idxR


#    print('----------------------------------')
#    print('maximum velocity at interval %d'%idxI_)
#    print('----------------------------------')
        return (max_, idxI_, r_, idxR_, rp_)

    def changeBaseLeg2TayAndDiff(self, x):
        """
          This procedure returns the coeffients of the first derivative of
          polynomial represented by x in Taylor basis. x must be given in Legendre
          basis.
        """
        #    print('-..----------------------------++++++')
        #    print('jacobian shape of x ',x.shape)
        #    print('-..----------------------------++++++')
        assert x.shape[0] == 7 * self.N_
        for i in range(0, self.N_):
            i0 = 6 * i
            i1 = 6 * (i + 1)
            #      print('============================')
            #      print('i',i)
            #      print('i0',i0)
            #      print('i1',i1)
            #      print('dPidTj',self.dPidTj.shape)
            #      print('x[i0:i1].shape',x[i0:i1].shape)
            #      print('x.shape',x.shape)
            #      print('self.N_',self.N_)
            self.y[i0:i1] = self.dPidTj.transpose().dot(x[i0:i1])
            self.y[i0:i1] = \
                np.array([(j+1.0)*y
                          for j, y in enumerate(self.y[i0+1:i1])]+[0.0])
        return self.y

    def diff_1(self, x, res):
        """
          Returns the derivative of the velocity of the
          pw 5-th order polynomial represented by x in the
          Legendre basis.
        """
        # First we get the interval where the maximum is attained and the index
        # and value of the root or boundary where this is attained.
        max_, idxI, tvmax, idxR, r = self.getMax(x)
        # ATENTION: NOW self.y containts the derivative
        # of x
        self.y = self.changeBaseLeg2TayAndDiff(x)

        res = res[0, :]
        tau = x[self.pdof_ + idxI]
        res[self.pdof_ + idxI] = -(max_) / tau

        res = res[6 * idxI:6 * (idxI + 1)]

        self.pmrc.diff_1(self.y[6 * idxI:6 * (idxI + 1) - 1], res[1:])
        #    print('----------------------------------')
        #    print('----------------------------------')
        #    print(res)
        res[0] = 0.0
        res[1] = res[1]
        res[2] = 2.0 * res[2]
        res[3] = 3.0 * res[3]
        res[4] = 4.0 * res[4]
        res[5] = 5.0 * res[5]
        #    print('-----first--------------')
        #    print(self.dPidTj.shape)
        #    print(res)
        #    print(res[:,].shape)
        #    print('--------------------')
        res[:] = self.dPidTj.dot(res)
        #    print('-------befor-------------------------')
        #    print(res)
        #    print('----------------------------------')
        #    print('----------------------------------')
        res[:] *= 2.0 / x[self.pdof_ + idxI]


class cMaxAccelerationConstraint(cBilateralConstraint):
    """
      Represets the constraints in the max of the absolute value of
      the derivative of a 5-th order polynomial. This
      is implemented with a function which gives the
      maximum of a 4th polynomial analysing the roots
      its 3th order polynomial derivative.
    """

    def __init__(self, _max, _N):
        kwargs = dict()
        kwargs['len_'] = 1
        super(self.__class__, self).__init__(len_=1, tol=_max)

        self.pmrc = cPolAbsMax(deg=3, interval=[-1.0, 1.0])
        #print("cMaxVelocityConstraint kwargs = ",kwargs)
        self.max_ = _max
        self.N_ = _N
        self.pdof_ = 6 * _N  # Polynomial dof
        self.tdof_ = _N  # Time dof

        self.dPidTj = np.matrix(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
             [-1.0 / 2.0, 0.0, 3.0 / 2.0, 0.0, 0.0,
              0.0], [0.0, -3.0 / 2.0, 0.0, 5.0 / 2.0, 0.0,
                     0.0], [3.0 / 8.0, 0.0, -30.0 / 8.0, 0.0, 35.0 / 8.0, 0.0],
             [0.0, 15.0 / 8.0, 0.0, -70.0 / 8.0, 0.0, 63.0 / 8.0]])
        self.dTidPj = np.linalg.inv(self.dPidTj)
        self.y = np.zeros((6 * self.N_, ))

    def value(self, x, res):
        """
          Returns the value of the maximum absolute value
          attained by the piece wise 5th order polynomial
          presented by x
        """
        max_, _, _, _ = self.getMax(x)
        res[:] = max_

    def getMax(self, x):
        """
          Get the position and the signed value of the max-abs of the piece-wise
          5th order polynomial described by x.

          Paramenters:
          ----------
            x: np.array of floats
              Coefficients of the polynomial in Taylor basis

          Returns:
          -------
            max_: float
              siged abs-max
            idxI_: uint
             interval where the abs-max is attained.
            r_: float
             value the place of the interval where the maximum is attained. this
             is paramentrized to the interval  [-1,1]
            idxI_: uint
               a number indicating where the maximum is attained.
              -2 is left boundary, -1 is right boundary, and a value
              grater than 0 is a root of its derivative.
        """
        #    print('-..----------------------------++++++')
        #    print('in getMax jacobian shape of x ',x.shape)
        #    print('-..----------------------------++++++')
        # First we get the derivative of x w.r.t. the Taylor basis.
        # so y is the polynomial representing the derivative of x
        self.y = self.changeBaseLeg2TayAndDiff(x)
        # now we get the maximum derivative at the first interval
        max_, r_, idxR_ = self.pmrc.getAbsMaxWithSign(
            self.y[0:4], _root=True, _rootidx=True)
        a = 0.0
        b = x[self.pdof_]
        r_ = timeMap(r_, a, b)
        idxI_ = 0
        max_ *= (2.0 / x[self.pdof_])**2
        # print(self.y[0:6])
        #print('val= %f, r = %f, idxR = %f'%(max_,r_,idxR_))
        # We look of the derivative at all intervals
        for j in range(1, self.N_):
            jl = 6 * j  # Left
            jr = 6 * (j + 1)  # Right
            val, r, idxR = self.pmrc.getAbsMaxWithSign(
                self.y[jl:jr - 2], _root=True, _rootidx=True)
            a = b
            b += x[self.pdof_ + j]
            r = timeMap(r, a, b)
            val *= (2.0 / x[self.pdof_ + j])**2
            # print(self.y[jl:jr])
            #print('val= %f, r = %f, idxR = %f'%(val,r,idxR))
            if abs(val) > abs(max_):
                max_ = val
                idxI_ = j
                r_ = r
                idxR_ = idxR
        max_ = max_

        return (max_, idxI_, r_, idxR_)

    def changeBaseLeg2TayAndDiff(self, x):
        """
          This procedure returns the coeffients of the first derivative of
          polynomial represented by x in Taylor basis. x must be given in Legendre
          basis.
        """
        #    print('-..----------------------------++++++')
        #    print('jacobian shape of x ',x.shape)
        #    print('-..----------------------------++++++')
        assert x.shape[0] == 7 * self.N_
        for i in range(0, self.N_):
            i0 = 6 * i
            i1 = 6 * (i + 1)
            #      print('============================')
            #      print('i',i)
            #      print('i0',i0)
            #      print('i1',i1)
            #      print('dPidTj',self.dPidTj.shape)
            #      print('x[i0:i1].shape',x[i0:i1].shape)
            #      print('x.shape',x.shape)
            #      print('self.N_',self.N_)
            self.y[i0:i1] = self.dPidTj.transpose().dot(x[i0:i1])
            self.y[i0:i1] = \
                np.array([(j+1.0)*y
                          for j, y in enumerate(self.y[i0+1:i1])]+[0.0])
            self.y[i0:i1] = \
                np.array([(j+1.0)*y
                          for j, y in enumerate(self.y[i0+1:i1])]+[0.0])
        return self.y

    def diff_1(self, x, res):
        """
          Returns the derivative of the velocity of the
          pw 5-th order polynomial represented by x.
        """
        # First we get the interval where the maximum is attained and the index
        # and value of the root or boundary where this is attained.
        max_, idxI, tamax, idxR, r = self.getMax(x)
        # ATENTION: NOW self.y containts the derivative
        # of x
        self.y = self.changeBaseLeg2TayAndDiff(x)

        res = res[0, :]

        tau = x[self.pdof_ + idxI]
        xdd = self.y[6 * idxI + 6 * (idxI + 1)]
        res[self.pdof_ + idxI] = -(max_) / tau

        res = res[6 * idxI:6 * (idxI + 1)]

        self.pmrc.diff_1(xdd[:-2], res[0, 2:])
        res[0] = 0.0
        res[1] = 0.0
        res[2] = 2.0 * res[2]
        res[3] = 6.0 * res[3]
        res[4] = 12.0 * res[4]
        res[5] = 20.0 * res[5]
        #    print('--------------------')
        #    print(self.dPidTj.shape)
        #    print(res.shape)
        #    print(res[:,].shape)
        #    print('--------------------')
        res[:] = self.dPidTj.dot(res[0, :])


if __name__ == '__main__':
    np.seterr(invalid='raise')
    deg = 5
    maxdx = np.zeros((deg + 1, ))
    interval = [-1.2, 1.3]
    rootc = cPolAbsMax(deg=deg, interval=interval)

    np.set_printoptions(precision=2)

    pr = polyfromroots([-1, -0.2, 0, 0.8, 1.2])

    def pol(lam_):
        return [pr[0], pr[1], pr[2], lam_, pr[4], pr[5]]

    fig, ax = plt.subplots()
    time = np.arange(interval[0], interval[1], 0.001)
    for lam_ in np.arange(pr[3] - 1.0, pr[3] + 1.0, 0.075):
        x = np.array(pol(lam_))
        y = polyval(time, x)
        ax.plot(time, y)
        sup, r = rootc.getAbsMaxWithSign(x, _root=True)
        rootc.diff_1(x, maxdx)
        ax.plot(r, sup, 'o')
        print(sup, r, maxdx)

    ax.grid()
    plt.show()
