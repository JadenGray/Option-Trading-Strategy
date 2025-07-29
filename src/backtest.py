# import required packages
import pandas as pd
import time
from tqdm import tqdm
from datetime import timedelta
# load functions from other python scripts
from .data_fetching import fetch_underlying, fetch_option_bars, adjust_to_business_day, build_option_ticker
from .data_prep import prepare_option_data
from .logging import generate_trade_log, compute_daily_pnl

# backtesting engine function
def run_backtest(
    symbol: str = "SPY",
    strike: float = 440.0,
    option_type: str = "call",
    window_start: str = "2023-07-01",
    window_end: str = "2023-10-30",
    expiry_delta_days: int = 14,
    r: float = 0.0525, # adjusted to 2023
    edge_entry: float = 1.0, 
    edge_exit: float = 0.2,
    max_pos: int = 10,
    vol_half_life_min: float = 8,
    fetch_data: bool = False
):
    if fetch_data:
        # get the underlying asset price data from polygon
        u_df = fetch_underlying(symbol, window_start, window_end)
        # save the underlying asset data
        u_df.to_csv('data/underlying.csv', index=False)
        
        days = pd.to_datetime(u_df['datetime']).dt.date.unique()
        opt_frames = []
        
        # for each day in the underlying df get the option bars from polygon
        # progress bar as takes a long time to run (limited to 5 calls/min, 1 for each day)
        for d in tqdm(sorted(days), desc='Fetching options'):
            # ensure we have a valid contract (expires on a trading day)
            exp = adjust_to_business_day(d + timedelta(days=expiry_delta_days))
            # get the option symbol that we are trading on that day
            ticker = build_option_ticker(symbol, exp, strike, option_type)
            # pull the bars
            bars = fetch_option_bars(ticker, d.strftime('%Y-%m-%d'))
            if not bars.empty: # check for empty day (debug)
                bars['expiry_date'] = pd.to_datetime(exp)
                opt_frames.append(bars)
            # 5 calls per minute
            time.sleep(12)
        # merge the data and save
        bars_df = pd.concat(opt_frames, ignore_index=True)
        bars_df.to_csv('data/option_bars.csv', index=False)
    else:
        # load the data from a run before
        u_df   = pd.read_csv('data/underlying.csv', parse_dates=['datetime'])
        bars_df = pd.read_csv('data/option_bars.csv', parse_dates=['datetime','expiry_date'])

    # run the pipeline, make the trades and compute the pnl
    df = prepare_option_data(bars_df, u_df, strike, option_type, r=r, vol_half_life_min=vol_half_life_min)
    trades_df = generate_trade_log(df, edge_entry, edge_exit, max_pos=max_pos)
    pnl_daily = compute_daily_pnl(trades_df)

    # return the dfs
    return df, trades_df, pnl_daily