import pandas as pd
import numpy as np

# synthetically fill the option price data
# done because a lot of the data is unavailable leading to hours of periods with no pricing available, breaking the bot
def fill_option_bars_with_mids(df, time_thr=5, vol_col='sigma'):
    filled = []
    df['mid'] = (df['open'] + df['close']) / 2
    for i in range(len(df) - 1):
        curr = df.iloc[i]
        nxt  = df.iloc[i+1]
        # compute best guess of available price
        mid_curr = 0.5 * (curr.open + curr.close)
        filled.append(
            pd.Series({
                'datetime':      curr.datetime,
                'mid':          mid_curr,
                'expiry_date':   curr.expiry_date,
                'option_symbol': curr.option_symbol,
                vol_col:         curr[vol_col],
                'bs_price':      curr.bs_price
            })
        )
        # fill gaps with synthetic bars
        delta_min = int((nxt.datetime - curr.datetime).total_seconds() // 60)
        if delta_min > 1:
            for m in range(1, delta_min):
                ts = curr.datetime + pd.Timedelta(minutes=m)
                minutes_to_next = delta_min - m
                # choose target for the expected price
                if minutes_to_next <= time_thr:
                    target = nxt.mid # move towards next available price
                    frac   = 0.5
                else:
                    target = curr.bs_price # move towards theoretical price (feels like cheating seems like a reasonable way to fill data)
                    frac   = 0.25 # don't move as far towards
                centre = mid_curr + frac * (target - mid_curr) # compute expectation of next price
                # simulate some noise
                var    = curr[vol_col] ** 2
                mid_s  = np.round(np.random.normal(centre, np.sqrt(var)), 2)
                filled.append(
                    pd.Series({
                        'datetime':      ts,
                        'mid':          mid_s,
                        'expiry_date':   curr.expiry_date,
                        'option_symbol': curr.option_symbol,
                        vol_col:         curr[vol_col],
                        'bs_price':      curr.bs_price
                    })
                )
    # last bar of real prices
    last = df.iloc[-1]
    mid_last = 0.5 * (last.open + last.close)
    filled.append(
        pd.Series({
            'datetime':      last.datetime,
            'mid':          mid_last,
            'volume':        last.volume,
            'expiry_date':   last.expiry_date,
            'option_symbol': last.option_symbol,
            vol_col:         last[vol_col],
            'bs_price':      last.bs_price
        })
    )
    # return the filled df
    return pd.DataFrame(filled).reset_index(drop=True)

# fill the option bars for each option and return full df 
def fill_option_bars(bars_df, time_thr=5, vol_col='sigma'):
    filled_list = []
    # loop through each option
    for opt_sym, grp in bars_df.groupby('option_symbol'):
        grp_sorted = grp.sort_values('datetime')
        filled = fill_option_bars_with_mids(grp_sorted, time_thr=time_thr, vol_col=vol_col)
        filled_list.append(filled)
    # return full df
    return pd.concat(filled_list, ignore_index=True)