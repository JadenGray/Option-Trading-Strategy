import numpy as np
import pandas as pd
from .filling import fill_option_bars
from .pricing_logic import black_scholes_vector, signed_edge

# volatility estimates: https://corporatefinanceinstitute.com/resources/career-map/sell-side/capital-markets/exponentially-weighted-moving-average-ewma/
def prepare_option_data(bars_df, u_df, strike, option_type, r=0.055, vol_half_life_min=8):
    # merge asof underlying + option bars
    u_b = u_df.sort_values('datetime').set_index('datetime')
    b_b = bars_df.sort_values('datetime').set_index('datetime')
    bb = pd.merge_asof(
        u_b, b_b,
        left_index=True, right_index=True,
        direction='nearest', tolerance=pd.Timedelta('1min')
    ).dropna(subset=['option_symbol']).reset_index()
    bb = bb.rename(columns={'price':'underlying_price'})
    # time to expiry
    bb['T'] = (bb['expiry_date'] - bb['datetime']).dt.total_seconds() / (365*24*3600)
    # realized, historical volatility using EWMA
    ret = np.log(bb['underlying_price']).diff()
    alpha = 1 - np.exp(np.log(0.5) / vol_half_life_min) # use the decided weight
    bb['sigma'] = np.sqrt(ret.ewm(alpha=alpha).var() * 252 * 6.5 * 60)
    
    # theoretical price
    bb['bs_price'] = black_scholes_vector(
        bb['underlying_price'], strike, bb['T'], r, bb['sigma'], option_type
    )

    # prepare for filling
    bars_prepped = bb[['datetime','open','high','low','close','volume','option_symbol','expiry_date','sigma','bs_price']]
    # fill data
    opt_df = fill_option_bars(bars_prepped)

    # Merge again to align timestamps
    u_df = u_df.sort_values('datetime').set_index('datetime')
    opt_df = opt_df.sort_values('datetime').set_index('datetime')
    df = pd.merge_asof(
        u_df, opt_df,
        left_index=True, right_index=True,
        direction='nearest', tolerance=pd.Timedelta('1min')
    ).dropna(subset=['option_symbol']).reset_index()
    df = df.rename(columns={'price':'underlying_price'})

    # recompute final metrics on filled bars
    df['T'] = (df['expiry_date'] - df['datetime']).dt.total_seconds() / (365*24*3600)
    ret = np.log(df['underlying_price']).diff()
    df['sigma'] = np.sqrt(ret.ewm(alpha=alpha).var() * 252 * 6.5 * 60)
    df['bs_price'] = black_scholes_vector(
        df['underlying_price'], strike, df['T'], r, df['sigma'], option_type
    )

    # use a constant 5 cent spread as there aren't any real market spreads easily accessible
    df['bid'] = np.round(df['mid'] - 0.025, 3)
    df['ask'] = np.round(df['mid'] + 0.025, 3)
    df['edge'] = df.apply(signed_edge, axis=1) 
    return df
