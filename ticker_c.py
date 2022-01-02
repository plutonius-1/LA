import pandas as pd
import datetime
from dateutil import parser


class Ticker_c:
    def __init__(self):
        self._main_df = None

    ########################################
    # Calculations Functions
    ########################################

    def calc_velocity_in_window(self,
                                t : str,
                                clean_norm_price_ser : pd.Series,
                                dates_ser : pd.Series,
                                window : int):
        """
        calc_velocity_in_window checks the diffrence in price devided by len of window
        window is always int to be in the same units of time as the price vector is
        """
        assert window > 0, "Window must be > 0"
        t_idx = dates_ser.loc[dates_ser == t].index.values[0]
        t_val = clean_norm_price_ser[t_idx]
        t0_val = clean_norm_price_ser[window]
        return (t_val - t0_val) / window


    def calc_mass(self,
                  t : str,
                  clean_norm_volum_ser : pd.Series,
                  dates_ser : pd.Series,
                  window : int):
        mass = 0.0
        t_idx = dates_ser.loc[dates_ser == t].index.values[0]
        for i in range(window):
            try:
                t_val = clean_norm_volum_ser[t_idx + i]
                mass += t_val
            except:
                pass

        return mass / window


    def parse_kinetic_energy(self,
                             main_df : pd.DataFrame,
                             window_size : int,
                             price_type = "High"):


        # Field: kinetic_energy
        # Description: list of results of calculations of 0.5*m*v**2
        kinetic_energy       = {}

        # Field: TODO - add vim mapping for field/Desctiption
        dates                = main_df["Date"]

        #
        clean_norm_prices    = self.norm_ser(self.get_clean_prices(main_df[price_type]))

        #
        clean_norm_volumes   = self.norm_ser(self.get_clean_prices(main_df["Volume"]))


        for date in dates:
            v = self.calc_velocity_in_window(t = str(date),
                                             clean_norm_price_ser = clean_norm_prices,
                                             dates_ser = dates,
                                             window = window_size)
            m = self.calc_mass(t = str(date),
                          clean_norm_volum_ser = clean_norm_volumes,
                          dates_ser = dates,
                          window = window_size)

            kinetic_energy.update({date : 0.5*m*(v**2)})


        return pd.DataFrame.from_dict(data = kinetic_energy,
                                      orient = "index")



    def parse_potential_energy(self):
        """
        sources of potential energy:
        1) regression to the mean
        2) news
        3) current fundumental ratios
        4) more???
        """
        return
    ########################################
    # Utility Functions
    ########################################

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


    def get_norm_dates(self,
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


    def get_clean_prices(self,
                         price_ser : pd.Series):
        """
        Takes a general price series and returns only digits and . for floats
        (removes $ and other signs)
        """
        price_ser = price_ser.astype(str)
        extr = price_ser.str.extract(r'(\d.+)', expand=False)
        extr = extr.astype(float)
        return extr

    ########################################
    # GETS
    ########################################
    def get_main_df(self):
        return self._main_df


    ########################################
    # SETS
    ########################################

    def set_main_df(self, main_df : pd.DataFrame):
        """
        Sets the main dataframe for the ticker object
        """
        self._main_df = main_df


