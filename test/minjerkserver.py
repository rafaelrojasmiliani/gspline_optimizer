import xmlrpc.client
try:
    from modules import gsplines
except ImportError:
    pass
from gsplinesopt.services.xmlrpc import cGplinesOptXMLRPCServer
import numpy as np
import json
import unittest
import matplotlib.pyplot as plt
import time
from threading import Thread



class cMyTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(cMyTest, self).__init__(*args, **kwargs)
        import sys
        np.set_printoptions(
            linewidth=5000000,
            formatter={'float': '{:+10.3e}'.format},
            threshold=sys.maxsize)

        self.path_ =  '/mjt'
        self.port_ = 8001
        self.server_ = None
        self.thread_ = None
        self.url_ = 'http://127.0.0.1:{}{}'.format(self.port_, self.path_)

    def setUp(self):
        server = cGplinesOptXMLRPCServer(self.path_, self.port_)
        self.server_ = server

        self.thread_ = Thread(target=server.serve_forever)
        self.thread_.start()

    def tearDown(self):
        self.server_.shutdown()


    def test_container(self):
        pass


def main():
    unittest.main()


if __name__ == '__main__':
    main()

