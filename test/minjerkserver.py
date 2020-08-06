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
from gsplines.services.gsplinesjson import piecewise2json, json2piecewise
from gsplines.plottool import show_piecewisefunction



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


    def test(self):
        server = xmlrpc.client.ServerProxy(self.url_)

        dim = 6  # np.random.randint(2, 6)
        N = 5  # np.random.randint(3, 120)

        wp = (np.random.rand(N + 1, dim) - 0.5) * 2.0 * np.pi
        args = {'unique_id': 0,
                'maximum_speed': 10,
                'maximum_acceleration': 100,
                'sampling_time': 0,
                'operator_vector': 0,  # ?
                'execution_time': 25,  # total
                'regularization_factor': 0,  #
                'basis_type': 0,  # a keyword
                'waypoints': wp.tolist()}


        json_args = json.dumps(args)
        qjson = server.gsplines_minimum_jerk(json_args)
        
        q = json2piecewise(qjson)

        show_piecewisefunction(q)


def main():
    unittest.main()


if __name__ == '__main__':
    main()

