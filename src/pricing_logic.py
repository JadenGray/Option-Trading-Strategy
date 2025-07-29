import numpy as np
from scipy.stats import norm

# black scholes option pricing - taken from pricing app and adapted to allow for np vectors
def black_scholes_vector(S, K, T, r, sigma, otype):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    # put and call formulae
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put  = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return np.where(otype == 'call', call, put)

# work out the difference between market prices and our theoretical prices
def signed_edge(row):
    if row['bs_price'] >= row['ask']:
        return row['bs_price'] - row['ask']   # buy signal (at offer)
    elif row['bs_price'] <= row['bid']:
        return row['bs_price'] - row['bid']   # sell signal (at bid)
    else:
        return 0.0                            

# compute the magnitude of trade (used in opening and add trades)
def get_trade_magnitude(edge, available, entry_thr, mults=(1.5, 2.0), sizes=(2, 3)):
    if abs(edge) >= entry_thr:
        if edge >= (mults[0] * entry_thr):
            if abs(edge) >= (mults[1] * entry_thr):
                return min(sizes[1], available) # buy 3 (big edge)
            else:
                return min(sizes[0], available) # buy 2 (big-ish edge)
        else:
            return min(1, available)  # buy 1 (medium edge)
    return 0

# compute size of trade (positive for buy, negative for sell)
def size_from_edge(edge, available, entry_thr, mults=(1.5, 2.0), sizes=(2, 3)):
    # use magnitude funciton
    mag = get_trade_magnitude(edge, available, entry_thr, mults=(1.5, 2.0), sizes=(2, 3))
    if edge >= 0:
        return mag
    else:
        return -mag 

# decide whether to exit a long position based on the edge
def get_exit_signal(edge, exit_thr):
        if abs(edge) < exit_thr:
            return 2    # exit both
        elif edge <= -exit_thr:
            return 1    # exit longs only
        elif edge >=  exit_thr:
            return -1   # exit shorts only
        else:
            return 0    # no exit