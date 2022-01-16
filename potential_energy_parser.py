from Base_c import *
import numpy as np
from scipy.optimize import basinhopping

class Potential_energy_parser(Base_c):
    """
    sources of potential energy:
    1) regression to the mean
    2) news
    3) current fundumental ratios
    4) more???
    """
    BUY = 1
    SELL = -1
    NO_BUY_NO_SELL = 0

    def __init__(self):
        Base_c.__init__(self)

        #===============
        # Field: _U
        # Desciption: vector of potential energies post calculation
        self._U       = None

        #===============
        # Field: _news_vector
        # Desciption: vector of normelised news potentials
        self._news_vector = None

        #===============
        # Field: _ratios_vector
        # Desciption: multi-column vector of different ratios
        self._ratios_vector = None

        #===============
        # Field: _fund_data_path
        # Desciption: holds the path for the fundumental ticker data
        self._fund_data_path = None




    ##==============================
    ## Calculations
    ##==============================
    def calc_regression_to_mean(self, price_vector : pd.Series, price_in_question : float):
        xs = np.array([i for i in reversed(range(len(price_vector)))])
        ys = self.clean_prices(price_vector).to_numpy()
        R2, func, med = self._find_best_fitting_mean(xs, ys)
        distance_from_mean = price_in_question/func[0]
        if (price_in_question < func[0] - med):
            buy_sell = 1
        elif (price_in_question > func[0] + med):
            buy_sell = -1
        else:
            buy_sell = 0
        return buy_sell, distance_from_mean

    def _calc_least_median(self, xs, ys, guess_theta1):
        def least_median_abs_1d(x: np.ndarray):
            # get an array X and sort it
            X = np.sort(x)

            # find the midpoint
            h = len(X)//2

            # diff is the ?
            diffs = X[h:] - X[:h]

            # Returns the indices of the minimum values along an axis.
            # in this case - find the point along time where the 2nd half is the biggsest over the
            # firt half
            min_i = np.argmin(diffs)

            # return the median
            return diffs[min_i]/2 + X[min_i]


        def best_median(theta1):
            # rs = ys - theta1*x2
            # meaning - reasults functinon is the actual price vector (y)
            # minus some constant theta1*the xs vector (why)
            rs = ys - theta1*xs

            # then find the actual median of the function above
            theta0 = least_median_abs_1d(rs)
            return np.median(np.abs(rs - theta0))

        # find the minimum of a function with an initial guess
        res = basinhopping(best_median, guess_theta1)
        theta1 = res.x[0]
        theta0 = least_median_abs_1d(ys - theta1*xs)
        return np.array([theta0, theta1]), res.fun

    def _calc_R2(self,
                y_vec : np.ndarray,
                f     : np.ndarray):

        mean = y_vec.mean()
        ss_tot = np.sum((y_vec - mean)**2)
        ss_res = np.sum((y_vec - f)**2)
        return 1 - (ss_res/ss_tot)

    def _find_best_fitting_mean(self,
                               xs : np.ndarray,
                               ys : np.ndarray):
        min_window = 5
        best_R2 = -2.0
        best_func = None
        best_med = None

        for ws in range(min_window,len(ys)//2):
            ws = ws*2
            _xs = xs[:ws]
            _ys = ys[:ws]
            guess0 = _ys.mean()
            theta, med = self._calc_least_median(_xs, _ys, guess0)
            temp_func  = theta[1]*_xs + theta[0]
            R2 = self._calc_R2(_ys, temp_func)
            if R2 > best_R2:
                best_R2 = R2
                best_func = temp_func
                best_med  = med

        return (R2, best_func, best_med)


    def calc_ratios_vectors(self,
                           list_of_vectors : list,
                           price_vector    : pd.Series
                           ):
        """
        """
        ratios_df = pd.DataFrame()
        RATIOS_TAGS = [
                       "p/e", # price to earnings
                       "p/pm", # price to profit margin
                       "p/gm"  # price to gross margin
                      ]

        for v in list_of_vectors:
            assert type(v) is pd.Series
            res_vec   = price_vector/v
            res_vec.name = f'{v.name}/price'
            ratios_df = pd.concat([ratios_df, res_vec])
        return ratios_df

    ##==============================
    ## Utilities
    ##==============================



    ##==============================
    ## Gets
    ##==============================
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


    def _get_fundamentals_vectors(self, ticker):
        raw_proc_data = self.get_fund_data(ticker)
        k_df = pd.DataFrame.from_dict(raw_proc_data[K_data])
        q_df = pd.DataFrame.from_dict(raw_proc_data[Q_data])
        return (q_df, k_df)

    def get_news_vector(self):
        return self._news_vector
    def get_ratios_vector(self):
        return self._ratios_vector
    def get_U(self):
        return self._U


    ##==============================
    ## Sets
    ##==============================
    def set_news_vector(self, news_vector:pd.DataFrame):
        self._news_vector = news_vector

    def set_ratios_vector(self, ratios_vector:pd.DataFrame):
        self._ratios_vector = ratios_vector

    def set_U(self, U : pd.DataFrame):
        self._U = U
