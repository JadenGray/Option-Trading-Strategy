import os, time 
import pandas as pd
import numpy as np
import requests
from datetime import datetime, date, timedelta
from scipy.stats import norm
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay

np.random.seed(42)

API_KEY = os.getenv('POLYGON_API_KEY', 'vbGmuUKSvRTv651AKaU6wweNTMAn2caB')
BASE_URL = 'https://api.polygon.io'

# -- Helper functions --
def polygon_get(path, params=None):
    p = params.copy() if params else {}
    p['apiKey'] = API_KEY
    r = requests.get(BASE_URL + path, params=p)
    r.raise_for_status()
    return r.json()


def fetch_underlying(symbol, start, end):
    out = polygon_get(
        f"/v2/aggs/ticker/{symbol}/range/1/minute/{start}/{end}",
        {'adjusted':'true','sort':'asc','limit':50000}
    )
    df = pd.DataFrame(out.get('results', []))
    df['datetime'] = pd.to_datetime(df['t'], unit='ms')
    return df[['datetime', 'c']].rename(columns={'c': 'price'})


def fetch_option_bars(opt_ticker, date_str):
    out = polygon_get(
        f"/v2/aggs/ticker/{opt_ticker}/range/1/minute/{date_str}/{date_str}",
        {'adjusted':'true','sort':'asc','limit':50000}
    )
    df = pd.DataFrame(out.get('results', []))
    if df.empty:
        return df
    df['datetime'] = pd.to_datetime(df['t'], unit='ms')
    df['option_symbol'] = opt_ticker
    return df[['datetime','o','h','l','c','v','option_symbol']].rename(
        columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'}
    )


def adjust_to_business_day(d: date) -> date:
    return (pd.Timestamp(d) + BDay(0)).date()


def build_option_ticker(symbol, expiry_date, strike, option_type):
    yy, mm, dd = expiry_date.year % 100, expiry_date.month, expiry_date.day
    letter = 'C' if option_type.lower().startswith('c') else 'P'
    strike_int = int(round(strike*1000))
    return f"O:{symbol}{yy:02d}{mm:02d}{dd:02d}{letter}{strike_int:08d}"


def black_scholes_vectorized(S, K, T, r, sigma, otype):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put  = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return np.where(otype == 'call', call, put)


def size_from_edge(edge, available, entry_thr, exit_thr, mids=(1.5, 2.0), sizes=(2, 3)):
    if edge < exit_thr:
        return 0
    if edge >= mids[1]:
        return min(sizes[1], available)
    if edge >= mids[0]:
        return min(sizes[0], available)
    if edge >= entry_thr:
        return min(1, available)
    return 0

def get_exit_signal(edge, exit_thr):
        if abs(edge) < exit_thr:
            return 2    # exit both
        elif edge < -exit_thr:
            return 1    # exit longs only
        elif edge >  exit_thr:
            return -1   # exit shorts only
        else:
            return 0    # no exit

def run_backtest(*,
                 symbol: str = 'SPY',
                 strike: float = 480.0,
                 option_type: str = 'call',
                 window_start: str = '2024-01-01',
                 window_end:   str = '2024-03-31',
                 expiry_delta_days: int = 14,
                 r: float = 0.01,
                 edge_entry: float = 0.5,
                 edge_exit:  float = 0.125,
                 max_pos: int = 3,
                 vol_half_life_min: float = 11,
                 warmup_minutes: int = 30,
                 fetch_data: bool = True):
    # --- (1) Fetch & preprocess as before ---
    if fetch_data:
        u_df = fetch_underlying(symbol, window_start, window_end)
        days = pd.to_datetime(u_df['datetime']).dt.date.unique()
        opt_frames = []
        for d in tqdm(sorted(days), desc='Fetching options'):
            exp = adjust_to_business_day(d + timedelta(days=expiry_delta_days))
            ticker = build_option_ticker(symbol, exp, strike, option_type)
            bars = fetch_option_bars(ticker, d.strftime('%Y-%m-%d'))
            if not bars.empty:
                bars['expiry_date'] = pd.to_datetime(exp)
                opt_frames.append(bars)
            time.sleep(0.2)
        opt_df = pd.concat(opt_frames, ignore_index=True)
    else:
        u_df   = pd.read_csv('underlying.csv', parse_dates=['datetime'])
        opt_df = pd.read_csv('option_bars.csv', parse_dates=['datetime','expiry_date'])

    u_df = u_df.sort_values('datetime').set_index('datetime')
    opt_df = opt_df.sort_values('datetime').set_index('datetime')
    df = pd.merge_asof(
        u_df, opt_df,
        left_index=True, right_index=True,
        direction='nearest', tolerance=pd.Timedelta('1min')
    ).dropna(subset=['option_symbol']).reset_index()
    df = df.rename(columns={'price':'underlying_price'})

    mid = 0.005 * np.round((df['open'] + df['close']) / 2 / 0.005)

    # 1b) enforce a fixed $0.01 spread
    df['bid'] = np.round(mid - 0.005,3)
    df['ask'] = np.round(mid + 0.005,3)
    df['T'] = (df['expiry_date'] - df['datetime']).dt.total_seconds() / (365*24*3600)
    ret = np.log(df['underlying_price']).diff()
    alpha = 1 - np.exp(np.log(0.5) / vol_half_life_min)
    df['sigma'] = np.sqrt(ret.ewm(alpha=alpha).var() * 252 * 6.5 * 60)
    df = df.iloc[warmup_minutes:].copy()

    df['bs_price'] = black_scholes_vectorized(
        df['underlying_price'], strike, df['T'], r, df['sigma'], option_type
    )
    df['edge'] = np.maximum(df['bs_price'] - df['ask'], df['bid'] - df['bs_price'])

    # --- (2) Initialize trade logging, position/book, and ID counter ---
    df['available'] = np.random.randint(1, 6, size=len(df))
    trade_logs  = []
    strike_key  = f"{symbol}_{strike:.3f}_{option_type}"
    positions   = {strike_key: 0}
    book        = {strike_key: []}
    next_id     = 1
    
    

     # --- GROUP BY EXPIRY / OPTION SYMBOL ---
    for opt_sym, subdf in tqdm(df.groupby('option_symbol'), desc='Expiry Groups'):
        subdf = subdf.sort_index()
        subdf = subdf[subdf['T'] > 0]

        # Initialize per-expiry state
        positions = {opt_sym: 0}
        book      = {opt_sym: []}

        # Iterate minute-by-minute for this contract
        for ts, row in subdf.iterrows():
            edge       = row['edge']
            bs_ask     = row['bs_price'] - row['ask']
            bs_bid     = row['bid']      - row['bs_price']
            curr_qty   = positions[opt_sym]
            total_avail= int(row['available'])

            if ts in [subdf.index[-2], subdf.index[-1]]:
                exit_signal = 2
            else:
                exit_signal = get_exit_signal(edge, edge_exit)

            # PHASE A: CLOSE?
            if ((curr_qty > 0 and exit_signal in (1,2)) or
                (curr_qty < 0 and exit_signal in (-1,2))) and curr_qty != 0 and total_avail > 0:

                to_close = abs(curr_qty)
                closed   = 0
                pnl      = 0.0
                close_ids= []
                price    = row['bid'] if curr_qty>0 else row['ask']

                # FIFO‐close with sign‑aware P&L
                while closed < to_close and book[opt_sym] and closed < total_avail:
                    lot         = book[opt_sym].pop(0)
                    lot_qty     = lot['qty']
                    entry_price = lot['price']

                    if lot_qty > 0:
                        pnl += lot_qty * (price - entry_price) * 100
                    else:
                        pnl += (-lot_qty) * (entry_price - price) * 100

                    close_ids.append(lot['id'])
                    closed += abs(lot_qty)

                pnl = round(pnl, 1)

                # Log the aggregate close
                this_id = next_id; next_id += 1
                trade_logs.append({
                    'id':               this_id,
                    'timestamp':        row['datetime'],
                    'symbol':           opt_sym,
                    'action':           'close',
                    'side':             'sell' if curr_qty>0 else 'buy',
                    'price':            price,
                    'quantity':         closed,
                    'position_after':   curr_qty - np.sign(curr_qty)*closed,
                    'corresponding_ids':close_ids,
                    'pnl':              pnl
                })

                positions[opt_sym] = curr_qty - np.sign(curr_qty)*closed
                total_avail      -= closed
            
            is_final_two = ts in [subdf.index[-2], subdf.index[-1]]
            # PHASE B: OPEN from flat
            if not is_final_two and curr_qty == 0 and total_avail > 0:
                # determine size as before
                size = (
                    3 if abs(edge) >= 2*edge_entry else
                    2 if abs(edge) >= 1.5*edge_entry else
                    1 if abs(edge) >= edge_entry else
                    0
                )
                size = min(size, total_avail, max_pos)
                if size > 0:
                    open_qty = np.sign(edge) * size
                    price    = row['ask'] if open_qty>0 else row['bid']
                    side     = 'buy' if open_qty>0 else 'sell'
                    this_id  = next_id; next_id += 1

                    trade_logs.append({
                        'id':               this_id,
                        'timestamp':        row['datetime'],
                        'symbol':           opt_sym,
                        'action':           'open',
                        'side':             side,
                        'price':            price,
                        'quantity':         abs(open_qty),
                        'position_after':   open_qty,
                        'corresponding_ids':[],
                        'pnl':              0.0
                    })
                    book[opt_sym].append({'id': this_id, 'qty': open_qty, 'price': price})
                    curr_qty = open_qty
                    positions[opt_sym] = curr_qty
                    total_avail -= abs(open_qty)

            # PHASE C: ADD to existing (only if we didn't just open)
            if not is_final_two and curr_qty != 0 and total_avail > 0:
                # only add if edge still favors same side
                if (curr_qty > 0 and bs_ask  > edge_entry) or (curr_qty < 0 and bs_bid  > edge_entry):
                    # compute tiered desired position
                    tier    = 3 if abs(edge) >= 2*edge_entry else \
                            2 if abs(edge) >= 1.5*edge_entry else \
                            1
                    desired = np.sign(curr_qty) * min(tier, max_pos)
                    delta   = desired - curr_qty
                    # **only proceed if delta would increase magnitude** (same sign)
                    if delta * curr_qty > 0:
                        add_amt = int(np.sign(delta) * min(abs(delta), total_avail))
                        if add_amt != 0:
                            price   = row['ask'] if add_amt > 0 else row['bid']
                            side    = 'buy'   if add_amt > 0 else 'sell'
                            this_id = next_id; next_id += 1

                            trade_logs.append({
                                'id':               this_id,
                                'timestamp':        row['datetime'],
                                'symbol':           opt_sym,
                                'action':           'add',
                                'side':             side,
                                'price':            price,
                                'quantity':         abs(add_amt),
                                'position_after':   curr_qty + add_amt,
                                'corresponding_ids':[],
                                'pnl':              0.0
                            })
                            book[opt_sym].append({'id': this_id, 'qty': add_amt, 'price': price})
                            curr_qty           += add_amt
                            positions[opt_sym] = curr_qty
                            total_avail        -= abs(add_amt)

    # --- Post‑day forced close for leftovers ---
        curr_qty = positions[opt_sym]
        if curr_qty != 0 and book[opt_sym]:
            last   = subdf.iloc[-1]
            price  = np.round((last.bid + last.ask) / 2,3)
            to_close = abs(curr_qty)
            closed   = 0
            pnl      = 0.0
            close_ids= []

            while closed < to_close and book[opt_sym]:
                lot         = book[opt_sym].pop(0)
                lot_qty     = lot['qty']
                entry_price = lot['price']

                if lot_qty > 0:
                    pnl += lot_qty * (price - entry_price) * 100
                else:
                    pnl += (-lot_qty) * (entry_price - price) * 100

                close_ids.append(lot['id'])
                closed += abs(lot_qty)

            this_id = next_id; next_id += 1
            trade_logs.append({
                'id':               this_id,
                'timestamp':        last['datetime'],
                'symbol':           opt_sym,
                'action':           'close',
                'side':             'sell' if curr_qty>0 else 'buy',
                'price':            price,
                'quantity':         closed,
                'position_after':   0,
                'corresponding_ids':close_ids,
                'pnl':              round(pnl,1)
            })
            positions[opt_sym] = 0

    # --- (4) Build trades_df & back‑fill opens’ corresponding_ids ---
    trades_df = pd.DataFrame(trade_logs)

    # Rebuild close_map globally
    close_map = {}
    for _, r in trades_df[trades_df.action=='close'].iterrows():
        for oid in r['corresponding_ids']:
            close_map.setdefault(oid, []).append(r['id'])

    # Assign clean mappings to opens/adds
    mask = trades_df.action.isin(['open', 'add'])
    trades_df.loc[mask, 'corresponding_ids'] = (
        trades_df.loc[mask, 'id'].map(lambda oid: close_map.get(oid, []))
    )

    trades_df.to_csv('trades.csv', index=False)

    # --- (5) Daily PnL & plot (unchanged) ---
    if not trades_df.empty:
        closes   = trades_df[trades_df.action=='close']
        daily    = closes.copy()
        daily['date'] = daily['timestamp'].dt.date
        pnl_daily = daily.groupby('date')['pnl'].sum().reset_index()
        pnl_daily.columns = ['datetime','daily_pnl']
        pnl_daily['cumulative_pnl'] = pnl_daily['daily_pnl'].cumsum()
        pnl_daily.to_csv('daily_pnl.csv', index=False)
    else:
        pnl_daily = pd.DataFrame(columns=['datetime','daily_pnl','cumulative_pnl'])
    plt.figure(figsize=(10,4))
    plt.plot(pnl_daily['datetime'], pnl_daily['cumulative_pnl'], label='Cumulative PnL')
    plt.title(f"{symbol} {option_type.title()} {strike} backtest")
    plt.xlabel('Date'); plt.ylabel('PnL'); plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    return df, trades_df, pnl_daily




if __name__=='__main__':
    a, b, c = run_backtest(fetch_data=False)
    a.to_csv('bid_asks.csv', index=False)
