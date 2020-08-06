
try:
    from modules import gsplines
except ImportError:
    pass
import numpy as np
import unittest
import matplotlib.pyplot as plt
import time
from threading import Thread

from gsplinesopt.ipopt.optpath import minimumjerkpath


class cMyTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(cMyTest, self).__init__(*args, **kwargs)
        import sys
        np.set_printoptions(
            linewidth=5000000,
            formatter={'float': '{:+10.3e}'.format},
            threshold=sys.maxsize)

    def test(self):
        N = 4
        dim = 6
        wp = np.random.rand(N+1, dim)

        q = minimumjerkpath(wp)


def main():
    unittest.main()


if __name__ == '__main__':
    main()


