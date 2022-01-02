import pandas as pd
import numpy as np
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt

df = pd.read_csv('HistoricalData.csv')

def least_median(xs, ys, guess_theta1):
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

def calc_R2(y_vec : np.ndarray,
            f     : np.ndarray):

    mean = y_vec.mean()
    ss_tot = np.sum((y_vec - mean)**2)
    ss_res = np.sum((y_vec - f)**2)
    return 1 - (ss_res/ss_tot)

def clean_prices(price_ser : pd.Series):
    """
    Takes a general price series and returns only digits and . for floats
    (removes $ and other signs)
    """
    price_ser = price_ser.astype(str)
    extr = price_ser.str.extract(r'(\d.+)', expand=False)
    extr = extr.astype(float)
    return extr

def find_best_fitting_mean(xs,
                           ys):
    min_window = 1
    window = 2
    best_R2 = -2.0
    best_func = None
    best_med = None

    for ws in range(20,len(ys)//2):
        ws = ws*2
        print(f"Trying window size: {ws}")
        _xs = xs[:ws]
        _ys = ys[:ws]
        guess0 = _ys.mean()
        theta, med = least_median(_xs, _ys, guess0)
        temp_func  = theta[1]*_xs + theta[0]
        R2 = calc_R2(_ys, temp_func)
        if R2 > best_R2:
            best_R2 = R2
            best_func = temp_func
            best_med  = med

    return (R2, best_func, best_med)

xs = df["Date"].to_numpy()
xs = np.array([i for i in reversed(range(len(df["Date"])))])
ys = clean_prices(df["Open"]).to_numpy()

R2, func, med= find_best_fitting_mean(xs, ys)
# theta, med = least_median(xs, ys, 150.0)
# active = ((ys < theta[1]*xs + theta[0] + med) & (ys > theta[1]*xs + theta[0] - med))
# not_active = np.logical_not(active)
# print(calc_R2(ys, theta[1]*xs + theta[0]))
# plt.plot(xs[not_active], ys[not_active], 'g.')
# plt.plot(xs[active], ys[active], 'r.')
# plt.plot(xs, theta[1]*xs + theta[0], 'b')
# plt.plot(xs, theta[1]*xs + theta[0] + med, 'b--')
# plt.plot(xs, theta[1]*xs + theta[0] - med, 'b--')
plt.plot(xs[:len(func)], ys[:len(func)])
plt.plot(xs[:len(func)], func)
plt.plot(xs[:len(func)], func + med, "b--")
plt.plot(xs[:len(func)], func - med, "b--")
plt.show()
