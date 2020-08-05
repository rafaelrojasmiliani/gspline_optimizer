from time import process_time
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, dok_matrix  # ,block_diag
from scipy.sparse import bsr_matrix, lil_matrix
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
from gsplines.basis0010 import cBasis0010
from gsplines.gspline import cSplineCalc


class cCost0010(object):
    """
      This class implements the calculus of the L2 norm of the jerk for a
      piece wise 5th order polynomial  curve in R^n which pass by a set of
      via-points.  Each polynomial is represented in the Legendre base.  Such
      curve q: R -> R^n is class C^4(R,R^n), i.e. it is continuous up to its
      4th derivative.  Such a curve is completely defined by a vector tau \in
      R^N, where N is the number of intervals that define the curve. So in
      the context of this class we will reefer as x to such a vector
      containing the length of each time interval where each 5th order
      polynomial is defined.
    """

    def __init__(self, _wp, _T):
        """
          Initialize an instance of this class given a set of via points.
          Such set must contain at least 3 points in R^n. The dimension of
          the space where the curve q(t) lies as well as the number of
          intervals is
          computed using the input.

          Parameters:
          ----------
            _q: 2D float np.array
              Array which contains the via points to get the total L2 norm of
              the Jerk. The dimension of the space where the curve lies is
              computed as _q.shape[1] and the number of intervals is
              _q.shape[0]-1.
        """

        self.T_ = _T
        self.dim_ = _wp.shape[1]
        self.N_ = _wp.shape[0] - 1
        self.wp_ = _wp.copy()
        self.basis_ = cBasis0010()
        self.splcalc_ = cSplineCalc(self.dim_, self.N_, self.basis_)

        self.initPerfornaceIndicators()

        self.P_ = np.eye(self.N_) - (1.0 / self.N_) * \
            np.ones((self.N_, self.N_))

        self.b_ = self.splcalc_.eval_b(self.wp_)

        self.A_ = None


    def initPerfornaceIndicators(self):
        self.wpfCalls = 0  # waypoint functions
        self.gradCalls = 0
        self.evalCalls = 0
        self.twpf = 0
        self.tcall = 0
        self.tgrad = 0
        self.tspsinv_ = 0
        self.tspscov_ = 0

    def printPerformanceIndicators(self):
        print('Number of cost function evaluations = {:d}'.format(
            self.evalCalls))
        print('Number of gradient evaluations = {:d}'.format(self.gradCalls))

        print('Number of waypoint constraints evaluations = {:d}'.format(
            self.wpfCalls))

        print('Mean time in cost function evalution = {:.4f}'.format(
            self.tcall / self.evalCalls))
        print('Mean time in gradient evalution = {:.4f}'.format(
            self.tgrad / self.gradCalls))
        print('Mean time in waypointConstraints evalution = {:.4f}'.format(
            self.twpf / self.wpfCalls))

        print('Total time in cost function evalution = {:.4f}'.format(
            self.tcall))
        print('Total time in gradient evalution = {:.4f}'.format(self.tgrad))
        print('Total time in waypoint constraint evalution= {:.4f}'.format(
            self.twpf))
#        print('Total time in sparse matrix evaluation = {:.4f}'.format(
#            self.tspscal_))

    def getFirstGuess(self):

        wp = self.wp_
#        ds = np.linalg.norm(wp[:-1, :] - wp[1:, :], axis=1)
        ds = np.ones((self.N_, ))
        ds = ds/np.sum(ds)*self.T_

        return ds

    def waypointConstraints(self, _tauv):
        '''Solves the waypoint constraints.  Computes the coefficient of the
            basis functions given the interval lenghts.'''
        t = process_time()
        self.A_ = self.splcalc_.eval_A(_tauv)
        y = spsolve(self.A_, self.b_)
        #        print(y)
        #        print(self.y0_)
        #        print('------------------------------------')
        #        print(y)
        self.wpfCalls += 1
        self.twpf += process_time() - t

        return y

    def eval_yTdQdiz(self, tau, idx, y, z):
        """
          Evaluate the quadratic form y^T dQdti z, where dQdti is the
          derivative of the Q matrix of the cost function with respect to t_i

          Parameters:
          ----------
            x: np.array float
              vector containing the time intervals.
          Returns:
          -------
            scalar.
        """
        res = 0.0
        omegai6 = -5.0*0.5 * np.power(2.0 / tau[idx], 6)
        i0 = idx * 6 * self.dim_
        for idim in range(0, self.dim_):
            j0 = i0 + idim * 6
            res += (
                450.0 * y[j0 + 3] * z[j0 + 3] + 3150.0 * y[j0 + 3] * z[j0 + 5]
                + 7350.0 * y[j0 + 4] * z[j0 + 4] + 3150.0 * y[j0 + 5] *
                z[j0 + 3] + 61740.0 * y[j0 + 5] * z[j0 + 5]) * omegai6
        return res

    def eval_yTQz(self, tau, y, z):
        """
          Evaluate the quadratic form y^T Q z,  where the Q matrix
          is defined in the cost function.

          Parameters:
          ----------
            x: np.array float
              vector containing the time intervals.
          Returns:
          -------
            scalar.
        """
        res = 0.0
        for iinter in range(0, self.N_):
            omegai5 = np.power(2.0 / tau[iinter], 5.0)
            i0 = iinter * 6 * self.dim_
            for idim in range(0, self.dim_):
                j0 = i0 + idim * 6
                res += (450.0 * y[j0 + 3] * z[j0 + 3] + 3150.0 * y[j0 + 3] *
                        z[j0 + 5] + 7350.0 * y[j0 + 4] * z[j0 + 4] +
                        3150.0 * y[j0 + 5] * z[j0 + 3] +
                        61740.0 * y[j0 + 5] * z[j0 + 5]) * omegai5
        return res

    def __call__(self, _tauv):
        """
          Evaluate the total jerk for a set of intervals _tauv.

          Parameters:
          ----------
            _tauv: np.array float
              vector containing the time intervals.
          Returns:
          -------
            scalar:
              L2 norm of the Jerk
        """
        t = process_time()
        y = self.waypointConstraints(_tauv)
        res = self.eval_yTQz(_tauv, y, y)

        self.evalCalls += 1
        self.tcall += process_time() - t

        return res

    def gradient(self, _tauv, res):
        t = process_time()
        y = self.waypointConstraints(_tauv)
        A = self.A_

        for i in range(0, self.N_):
            z = self.splcalc_.eval_dAdtiy(_tauv, i, y)
            dydt = spsolve(A, z)
            res[i] = -2.0*self.eval_yTQz(_tauv, y, dydt) + \
                self.eval_yTdQdiz(_tauv, i, y, y)

        self.gradCalls += 1
        self.tgrad += process_time() - t
        return self.P_.dot(res)
