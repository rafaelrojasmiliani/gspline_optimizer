#!/usr/bin/env python
from __future__ import print_function
from functools import wraps

try:  # Python 2.X
    from SimpleXMLRPCServer import SimpleXMLRPCServer
    from SimpleXMLRPCServer import SimpleXMLRPCRequestHandler
    from SocketServer import ThreadingMixIn
    import ConfigParser
    import StringIO
except ImportError:  # Python 3.X
    from xmlrpc.server import SimpleXMLRPCServer
    from xmlrpc.server import SimpleXMLRPCRequestHandler
    from socketserver import ThreadingMixIn

import json
import numpy as np
import math
import os
import sys
import time
import traceback
import time
import gsplines
import abc


def debug_response(function):
    @wraps(function)
    def decorator(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            tb_list = traceback.extract_tb(exc_tb)
            for i, tb in enumerate(tb_list):
                if tb[0].find(__file__) < 0:
                    raise RuntimeError(' "{}" @ {}:{}'.format(
                        str(e), tb_list[i - 1][0], tb_list[i - 1][1]))
            raise RuntimeError(' "{}" @ {}:{}'.format(
                str(e), tb_list[-1][0], tb_list[-1][1]))

    return decorator


class cGplinesOptXMLRPCServer(object):
    ''' XMLRPC service which stores gsplines and
        retrieve their values on request.
    '''

    def __init__(self, _service_path, _port):
        class RequestHandler(SimpleXMLRPCRequestHandler):
            rpc_paths = (_service_path, )

        server = SimpleXMLRPCServer(('0.0.0.0', _port),
            requestHandler = RequestHandler,
            logRequests = False,
            allow_none = True)
        server.register_instance(self)

        self.server_ = server
        self.trajectories={}
        self.trajectories_deriv_={}
        self.follower=None
        self.kp_=0.001
        self.kacc_=10000


    def serve_forever(self):
        self.server_.serve_forever()

    def shutdown(self):
        self.server_.shutdown()
        self.server_.socket.close()

    def gsplines_optimal_trajectory(self, _jsonreq):
        ''' Generates desired trajectory'''
        json_dict = json.loads(_jsonreq)
        wp = np.array(json_dict['waypoints'])
        N = wp.shape[0] - 1
        dim = wp.shape[1]
        
        qd_max = json_dict['maximum_speed']
        qdd_max = json_dict['maximum_acceleration']
        min_T = json_dict['execution_time']

