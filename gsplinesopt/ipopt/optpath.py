from .ipoptinterface import cIpoptInterface
from gsplines.basis import cBasis0010
from gsplines.functionals import cJerkL2Norm
from gsplines.functionals import cConvexCombinationL2Norm
from gsplines.interpolator import interpolate


def minimum_jerk_path(_wp):
    basis = cBasis0010()
    cost = cJerkL2Norm(_wp, basis)
    execution_time = _wp.shape[0] - 1
    ipopt = cIpoptInterface(cost, execution_time)
    tauv, info = ipopt.solve()

    result = interpolate(tauv, _wp, basis)

    del ipopt

    return result


def minimum_weighed_speed_jerk_path(_wp, _k):
    alpha = _k
    N = _wp.shape[0] - 1
    execution_time = 4.0*float(N)/np.sqrt(2.0)
    k4 = np.power(_k, 4)
    alpha = k4 / (1.0 + k4)
    basis = cBasis1010(alpha)
    cost = cConvexCombinationL2Norm(_wp, basis, 1, 3, alpha)
    ipopt = cIpoptInterface(cost, execution_time)
    tauv, info = ipopt.solve()

    result = interpolate(tauv, _wp, basis)

    del ipopt

    return result
