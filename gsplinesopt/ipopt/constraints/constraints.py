import numpy as np


class cConstraint(object):
    """
        Base class for constraints. Bassicaly adds the
        __len__  function.
    """

    # def __init__(self,_dof=None,_len=None,_pdeg=None,_isvet=None,**kwargs):

    def __init__(self, **kwargs):
        """
          Paramenters
          -----------
          len_ : unsigned int
                 Number of constraints contained in an instance.
        """
        self.len_ = kwargs['len_']
        super(cConstraint, self).__init__()

    def __len__(self):
        return self.len_


class cBilateralConstraint(cConstraint):
    """
        Base class for Bilateral constraints.  It adds the member functions for
        -tol/2 <= f_i(x)  <= tol/2
    """

    def __init__(self, **kwargs):
        """
          Paramenters
          -----------
          len_ : unsigned int (for parend cConstraint)
                 Number of constraints contained in an instance.
          tol  : unsigned int
                 Tollerance for the constraint
        """
        super(cBilateralConstraint, self).__init__(**kwargs)
        if 'tol' not in kwargs:
            self.tol = 0.0
        else:
            self.tol = kwargs['tol']
        pass

    def upperBounds(self):
        return np.ones((len(self), )) * self.tol * 0.5

    def lowerBounds(self):
        return -np.ones((len(self), )) * self.tol * 0.5


class cUnilateralUpperConstraint(cConstraint):
    """
        Base class for Unilateral upper constraints.  It adds the member
        functions for -Inf <= f_i(x)  <= 0
    """

    def __init__(self, **kwargs):
        super(cUnilateralUpperConstraint, self).__init__(**kwargs)
        pass

    def upperBounds(self):
        return np.zeros((len(self), ))

    def lowerBounds(self):
        return -np.ones((len(self), )) * (1.0e20)


class cUnilateralLowerConstraint(cConstraint):
    """
        Base class for Unilateral lower constraints.
        It adds the member functions for
         0 <= f_i(x)  <= inf
    """

    def __init__(self, **kwargs):
        super(cUnilateralLowerConstraint, self).__init__(**kwargs)
        pass

    def upperBounds(self):
        return np.ones((len(self), )) * (1.0e20)

    def lowerBounds(self):
        return np.zeros((len(self), ))


class cPresendenceConstraint(cUnilateralUpperConstraint):
    """
        Class for a presedence constraint. (on the state)
            x[i1] <= x[i2] <= x[i3] <= ...
    """

    def __init__(self, **kwargs):
        """
          Paramenters
          -----------
          arrayIdx:   array of unsigned int
                      array with a sequence of variables
                      with a order constraint
        """
        kwargs['len_'] = len(kwargs['arrayIdx']) - 1
        super(cPresendenceConstraint, self).__init__(**kwargs)
        self.arrayIdx = kwargs['arrayIdx']

    def value(self, x, res):
        for i in range(len(self.arrayIdx) - 1):
            res[i] = x[self.arrayIdx[i + 1]] - x[self.arrayIdx[i]]

    def diff_1(self, x, res):
        for i in range(len(self.arrayIdx) - 1):
            res[i, self.arrayIdx[i + 1]] = 1.0
            res[i, self.arrayIdx[i]] = -1.0

    def diff_2(self, idx, x, res):
        pass


class cAffineBilateralConstraint(cBilateralConstraint):
    """
        Class for a linear bilateral constraint
        as
            Ax +b = 0
    """

    def __init__(self, **kwargs):
        """
          Paramenters
          -----------
          A: numpy.array nxn
            Matrix of the defintion of the linear constrait
          b: numpy.array n
            vector of the definition
        """
        self.A = kwargs['A']
        self.b = kwargs['b']
        assert (self.A.shape[0] == self.b.shape[0])
        kwargs['len_'] = self.A.shape[0]
        super(cAffineBilateralConstraint, self).__init__(**kwargs)

    def value(self, x, res):
        res[:] = self.A.dot(x) + self.b

    def diff_1(self, x, res):
        res[:, :] = self.A

    def diff_2(self, idx, x, res):
        pass

    def diff_2_contraction(self, x, lam, res):
        # self.res.fill(0.0)
        pass


class cAffineUnilateralLowerConstraint(cUnilateralLowerConstraint):
    """
        Class for a linear bilateral constraint
        as
            Ax +b = 0
    """

    def __init__(self, **kwargs):
        """
          Paramenters
          -----------
          A: numpy.array nxn
            Matrix of the defintion of the linear constrait
          b: numpy.array n
            vector of the definition
        """
        self.A = kwargs['A']
        self.b = kwargs['b']
        assert (self.A.shape[0] == self.b.shape[0])
        kwargs['len_'] = self.A.shape[0]
        super(cAffineUnilateralLowerConstraint, self).__init__(**kwargs)

    def value(self, x, res):
        res[:] = self.A.dot(x) + self.b

    def diff_1(self, x, res):
        res[:, :] = self.A

    def diff_2(self, idx, x, res):
        pass

    def diff_2_contraction(self, x, lam, res):
        # self.res.fill(0.0)
        pass


class cQuadraticScalarBilateralConstraint(cBilateralConstraint):
    """
        Class for a linear bilateral constraint
        as
            x^T P x +b^T x + c = 0
    """

    def __init__(self, **kwargs):
        """
          Paramenters
          -----------
          _P: numpy.array nxn
            Matrix of the defintion of the linear constrait
          _b: numpy.array n
            vector of the definition
          _c: float
            see definition
        """
        self.P_ = kwargs['_P'].copy()
        self.b_ = kwargs['_b'].copy()
        self.c_ = kwargs['_c'].copy()
        assert (self.P_.shape[0] == self.P_.shape[1])
        assert (self.P_.shape[0] == self.b_.shape[0])
        kwargs['len_'] = 1
        super(cQuadraticScalarBilateralConstraint, self).__init__(**kwargs)

    def value(self, _x, _res):
        _res[:] = self.P_.dot(_x).dot(_x) + self.b_.dot(_x) + self.c_

    def diff_1(self, _x, _res):
        _res[:, :] = (self.P_.T + self.P_).dot(_x) + self.b_

    def diff_2(self, _idx, _x, _res):
        _res[:, :] = (self.P_.T + self.P_)

    def diff_2_contraction(self, x, lam, res):
        # self.res.fill(0.0)
        pass
