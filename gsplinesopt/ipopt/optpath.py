from .ipoptinterface import cIpoptInterface
from gsplines.basis import cBasis0010
from gsplines.functionals import cJerkL2Norm
from gsplines.interpolator import interpolate


def minimumjerkpath(_wp):
    basis = cBasis0010()
    cost = cJerkL2Norm(_wp, basis)
    ipopt = cIpoptInterface(cost)
    tauv, info = ipopt.solve()

    result = interpolate(tauv, _wp, basis)

    del ipopt

    return result
