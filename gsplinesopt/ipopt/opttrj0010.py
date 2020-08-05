from .cost0010canonic import cCost0010Canonic
from .cost0010 import cCost0010
from .ipoptinterface import cIpoptInterface
from gsplines.piecewisefunction import cPiecewiseFunction

def opttrj0010(_wp, _T, _printPerformace=False):
    cost = cCost0010(_wp, _T)
    ipopt = cIpoptInterface(cost)
    tauv, info = ipopt.solve()
    q = cost.splcalc_.getSpline(tauv, _wp)
    if _printPerformace:
        cost.printPerformanceIndicators()
    return q


def opttrj0010canonic(_wp, _T, _printPerformace=False):
    cost = cCost0010Canonic(_wp, _T)
    ipopt = cIpoptInterface(cost)
    tauv, info = ipopt.solve()
    q = cost.splcalc_.getSpline(tauv, _wp)
    if _printPerformace:
        cost.printPerformanceIndicators()
    return q
