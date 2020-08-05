#!/usr/bin/env python
from __future__ import print_function
from functools import wraps
from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server  import SimpleXMLRPCRequestHandler
from socketserver import ThreadingMixIn
import json
import numpy as np
import math
import os
import sys
import time
import traceback
import pathlib
CWD = pathlib.Path(__file__).parent.absolute()
modpath = pathlib.Path(CWD, 'gspline')
sys.path.append(str(modpath))
modpath = pathlib.Path(CWD, 'vstoolsur')
sys.path.append(str(modpath))
modpath = pathlib.Path(CWD, 'vstoolsur', 'vsdk')
sys.path.append(str(modpath))
from vsurt.urdk.urdk import cUrdk
from opttrj.opttrj0010 import opttrj0010
from gsplines import piecewise2json
from gsplines import cSplineCalc
from gsplines import cBasis1010
from gsplines import cBasis0010

def what(function):
    @wraps(function)
    def decorator(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            tb_list = traceback.extract_tb(exc_tb)
            for i, tb in enumerate(tb_list):
                if tb[0].find(__file__) < 0:
                    raise RuntimeError(' "{}" @ {}:{}'.format(str(e), tb_list[i-1][0], tb_list[i-1][1]))
            raise RuntimeError(' "{}" @ {}:{}'.format(str(e), tb_list[-1][0], tb_list[-1][1]))
    return decorator

class cMjtServer(object):

    # (internal use) custom initialization of the XMLRPC server

    def __init__(self):

        urmodel = cUrdk(_ip='10.10.238.32')

        self.urmodel_ = urmodel

    @what
    def get_server(self):
        class RequestHandler(SimpleXMLRPCRequestHandler):
            rpc_paths = ('/mjt',)

        #https://gist.github.com/mcchae/280afebf7e8e4f491a66
        class SimpleThreadXMLRPCServer(ThreadingMixIn, SimpleXMLRPCServer):
            pass

        hostname = '0.0.0.0'
        port = int(5000)
        server = SimpleThreadXMLRPCServer((hostname, port), requestHandler=RequestHandler, logRequests=False, allow_none=True)
        server.register_instance(self)
        return server

    @what
    def trajectory_generate(self, _jsonreq):
        ''' Generates desired trajectory'''
        print('planning specification:\n{}'.format(json.dumps(json.loads(_jsonreq), indent=4)))
        urmodel = self.urmodel_

        json_dict = json.loads(_jsonreq)
        wp = np.array(json_dict['waypoints'])
        N = wp.shape[0] - 1
        dim = wp.shape[1]
        
        qd_max = json_dict['maximum_speed']
        qdd_max = json_dict['maximum_acceleration']
        min_T = json_dict['execution_time']
        
        TN = N
        trajectory = opttrj0010(wp, TN)
        time_array = np.arange(0.0, TN, TN/10000.0)
        qd_time = trajectory.deriv()(time_array)
        qdd_time = trajectory.deriv(2)(time_array)
        pd_max = np.max(qd_time)
        pdd_max = np.max(qdd_time)
        v_max = np.max(np.array([urmodel.vel(q, qd)[:3] for q, qd in zip(qd_time, qdd_time)]))

        Tv = pd_max / qd_max * TN
        Ta = np.sqrt(pdd_max / qdd_max) * TN
        Tsafe = v_max / 0.50 * TN
        Topt = min([max([Tv, Ta, Tsafe]), min_T])
        print([Tv, Ta, Tsafe])
        print(Topt)
        tauv = trajectory.tau_ / TN * Topt
        splcalc = cSplineCalc(dim, N, cBasis0010())
        trajectory = splcalc.getSpline(tauv, wp)
        
        result = piecewise2json(trajectory)
        del splcalc
        # print('trajectory generate: {}'.format(trajectory.unique_id))
        return result

if __name__ == '__main__':
    mjt = cMjtServer()
    server = mjt.get_server()
    server.register_introspection_functions()
    server.serve_forever()
