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
import proj_local_data_mngr_c
import proj_BarData_c
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

        #==================
        # Field: local_data_mngr
        # Description:
        self.local_data_mngr = proj_local_data_mngr_c.localDataMngr_c()

    ##==============================
    ## Calculations
    ##==============================

    ##==============================
    ## Utilities
    ##==============================
    def match_2_ser_objects(self,
                            s1 : pd.Series,
                            s2 : pd.Series):
        """
        Extending s1 or s2 to be the same length as the longest one -
        the extenstion is based on dates.
        """
        # if (len(s1) > len(s2)):
            # s_extend = s2
            # s_template = s1
        # else:
            # s_extend = s1
            # s_template = s2

        # # first assert that all dates in s_extend are in s_template
        # date_template = s_template['date'].values
        # for temp_date in s_extend['date']:
            # if (temp_date not in date_template):
                # print(f'{__name__}: {temp_date} not in {s_template.name}')
                # return

        return
    def extend_series_to_match_series(self,
                                      s_to_extend : pd.DataFrame,
                                      s_to_match  : pd.DataFrame):
        extended_ser = pd.Series()
        vals = []
        res = {}
        # assert size differnce
        assert len(s_to_extend) < len(s_to_match)

        # assert order of vectors dates
        to_extend_dates = s_to_extend['date']
        to_match_dates  = s_to_match['date']
        assert to_extend_dates[-1] > to_extend_dates[0]
        assert to_match_dates[-1]  > to_match_dates[0]

        # first make sure that the fundumental data "to_match_date" has the furthest data point
        new_dates = to_match_dates
        for date in new_dates:
            for i in range(len(to_extend_dates)-1):
                val      =
                low_date = to_extend_dates[i]
                high_date = to_extend_dates[i+1]

                if date > low_date and date < high_date:
                    res.update({date : s_to_extend.loc[]})


        return

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

        while (not self.ibapi.isConnected()):
            cfg.time.sleep(1)
            print("Sleeping while ibapi is connectin")
        self.ibapi_run_thread = threading.Thread(target = self.ibapi.run)
        self.ibapi_run_thread.start()
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
        # TODO
        return

    def get_fund_data(self, ticker : str):
        """
        returns a data object from ticker object
        """
        ticker = ticker.upper()

        # first look localy for raw xml data
        data = self.finStatementXmlReader.try_get_processed_fund_data(ticker)

        # if data does not exists localy - get from IB
        if data is None:
            print(f"{__name__} : did not find fundumental local copy for {ticker} - getting from IB")
            self.call_ibapi_function(cfg.GET_FUNDUMENTALS,
                                     ticker = ticker)
            self.finStatementXmlReader.set_ticker(ticker)
            data = self.finStatementXmlReader.get_fundamentals_obj()

        # otherwise if data exists check for date validity
        else:
            if self.local_data_mngr.is_data_out_of_date(ticker, "IB"):

                # get old data first
                self.finStatementXmlReader.set_ticker(ticker)
                old_data = self.finStatementXmlReader.get_fundamentals_obj()

                # download new data
                self.call_ibapi_function(cfg.GET_FUNDUMENTALS, ticker = ticker)
                _Q_data, _K_data = self.finStatementXmlReader.parse_comp_data()
                new_data = {cfg.Q_data : _Q_data, cfg.K_data : _K_data}
                data = self.local_data_mngr.add_raw_data_to_existing_processed_raw_data(new_data, old_data)
                self.finStatementXmlReader.save_fund_obj(ticker, merged_data)
        # ticker_obj    = Ticker_data_c()
        # ticker_obj.set_ticker(ticker)
        # ticker_obj.set_raw_data(data, "IB")
        # ticker_obj.set_raw_statements()
        # return ticker_obj.get_raw_statements()
        return data
    ##==============================
    ## Sets
    ##==============================
    def set_main_df(self, main_df : pd.DataFrame):
        self._main_df = main_df
