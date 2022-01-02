import pandas as pd
import datetime
from dateutil import parser

main_df = pd.read_csv("HistoricalData.csv")


"""
we analyze accordingn to Langrangian:
L = K - U
where
    K = Kinetic Energy
    U = Potential Energy

---- Kinetic Energy ----
In regular mechinal cases - Kinetical energy is easy to define:
  K = 0.5*m*v^2
where:
    m = mass
    v = velocity
The question becomes - what are mass and velocity in out case:
we can define:

    m = Trade Volume (Normilzed for period?)

    v = d/dy(price):
        v = (price(t) - price(t0)) / abs(t - t0)

---- Potential Energy ----
This is a little trickier:
things to include:

    * regresion/Distance from mean
    * news
    * current ratios(p/e etc...)

"""
def norm_ser(s : pd.DataFrame):
    # TODO - deal with dirty series
    def get_norm_val(curr_val,
                     _max,
                     _min):

        return (curr_val - _min) / (_max - _min)
    _s = s.astype(float)
    _MAX = _s.max()
    _MIN = _s.min()

    for idx, i in enumerate(s):
        _s[idx] = get_norm_val(float(i), _MAX, _MIN)

    return _s

def get_norm_dates(main_data : pd.DataFrame,
                   break_to_seconds = False):

        patt = "%Y/%m/%d"
        date_col = main_data["Date"]
        new_date_col = date_col

        if break_to_seconds:
            patt = "%Y/%m/%d %H:%M:%S"

        for idx, date in enumerate(date_col):
            d = parser.parse(date)
            new_date_col[idx] = d.strftime(patt)
        return new_date_col

def get_clean_price(price_ser : pd.DataFrame):
    extr = price_ser.str.extract(r'(\d.+)', expand=False)
    extr = extr.astype(float)
    return extr

def calc_velocity(t : str,
                  t0 : str,
                  main_df : pd.DataFrame,
                  price_ser : pd.Series,
                  window = None):
    """
    t - str: in date format
    t0 - str: in date format
    """
    t_idx = main_df.loc[main_df["Date"] == t].index.values[0]
    t0_idx = main_df.loc[main_df["Date"] == t0].index.values[0]
    price_ser = norm_ser(price_ser)

    t_val  = price_ser[t_idx]
    if (window == None):
        t0_val = price_ser[t0_idx]
        dist   = abs(t0_idx - t_idx)
    else:
        t0_val = price_ser[int(window)]
        dist = int(window)

    val_delta = t_val - t0_val
    return val_delta / dist

def calc_mass(t : str,
              t0 : str,
              volume_ser : pd.Series):

    t_idx = main_df.loc[main_df["Date"] == t].index.values[0]
    norm_volume_ser = norm_ser(volume_ser)

    # t0_idx = main_df.loc[main_df["Date"] == t0].index.values[0]

    # norm_volume_ser = norm_ser(volume_ser)
    # t_val = norm_volume_ser[t_idx]

    # if (window == None):
        # t0_val = norm_volume_ser[t0_idx]
        # dist = abs(t0_idx - t_idx)
    # else:
        # t0_val = norm_volume_ser[int(window)]
        # dist = int(window)

    # return 1
    return norm_volume_ser[t_idx]



def clean_main_df(main_df : pd.DataFrame):
    def clean_dollar(x):
        try:
            if "$" in x:
                return x.split("$")[1]
            else:
                return x
        except:
            return x
    return main_df.applymap(clean_dollar)
