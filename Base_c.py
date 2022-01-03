import pandas as pd
import numpy as np
import datetime
from dateutil import parser
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../IB/ib_stocks_app')

import cfg
import proj_utils
import proj_ibtws_c
import threading
import proj_sec_scraping_utils
import proj_snapshotXmlReader
import proj_finStatementsXmlReader
import proj_sics_handler_c
from proj_ticker_data_c_v2 import Ticker_data_c
from cfg import logging

class Base_c:
    def __init__(self):
        self._main_df = None

        #==================
        # field: self.ibapi
        # description: handle of IB API APP
        self.ibapi    = None

        #==================
        # field: self.ibapi
        # description: handle of IB API RUN thread
        self.ibapi_run_thread = None

        #==================
        # Field: host_addr
        # Description: local computer ip addr
        self.host_addr = "192.168.0.152"

        #==================
        # Field: finStatementXmlReader
        # handle to XML reader to decode IB fund data
        self.finStatementXmlReader = proj_finStatementsXmlReader.finStatementsXmlReader_c()

    ##==============================
    ## Calculations
    ##==============================

    ##==============================
    ## Utilities
    ##==============================
    def norm_ser(self,
                 s : pd.DataFrame):
        def get_norm_val(curr_val,
                         _max,
                         _min):
            if ((type(curr_val) is float) or (type(curr_val) is int)):
                return (curr_val - _min) / (_max - _min)
            return pd.np.nan

        _s = s.astype(float)
        _MAX = _s.max()
        _MIN = _s.min()

        for idx, i in enumerate(s):
            _s[idx] = get_norm_val(float(i), _MAX, _MIN)

        return _s

    def norm_dates(self,
                       main_df : pd.DataFrame,
                       break_to_seconds = False):
            """
            Takes a main_df and returns a 'standart' dates vector according to
            whet 'patt' var defines
            """

            patt = "%Y/%m/%d"
            date_col = main_df["Date"]
            new_date_col = date_col

            if break_to_seconds:
                patt = "%Y/%m/%d %H:%M:%S"

            for idx, date in enumerate(date_col):
                d = parser.parse(date)
                new_date_col[idx] = d.strftime(patt)
            return new_date_col

    def clean_prices(self,
                         price_ser : pd.Series):
        """
        Takes a general price series and returns only digits and . for floats
        (removes $ and other signs)
        """
        price_ser = price_ser.astype(str)
        extr = price_ser.str.extract(r'(\d.+)', expand=False)
        extr = extr.astype(float)
        return extr

    def start_ibapi_app(self,
                        host = None,
                        port = 7496,
                        clientId = 10):
        assert host != None, "Trying to start ibapi app - host is None"
        self.ibapi = proj_ibtws_c.IbTws(host = host,
                                   port = port,
                                   clientId = clientId)

        self.ibapi_run_thread = threading.Thread(target = self.ibapi.run)
        self.ibapi_run_thread.start()
        while (not self.ibapi.isConnected()):
            cfg.time.sleep(1)
            print("Sleeping while ibapi is connectin")
        return

    def close_ibapi_app(self):
        self.ibapi_run_thread.join()
        self.ibapi.Close()
        self.ibapi = None
        return

    def call_ibapi_function(self, function_name, **kwargs):
        if (self.ibapi is None):
            self.start_ibapi_app(host = self.host_addr)
        print(f'IBAPI calling function {function_name} with kwargs:\n{kwargs}')
        self.ibapi.call_function(function_name, **kwargs)

    ##==============================
    ## Gets
    ##==============================
    def get_main_df(self):
        return self._main_df

    def get_fund_vector(self):

        return
    ##==============================
    ## Sets
    ##==============================
    def set_main_df(self, main_df : pd.DataFrame):
        self._main_df = main_df
