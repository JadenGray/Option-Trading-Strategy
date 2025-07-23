import os, time
import pandas as pd
import numpy as np
import requests
from datetime import datetime, date, timedelta
from scipy.stats import norm
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay

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

    df['bid'], df['ask'] = df['open'], df['close']
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
    trade_logs  = []
    strike_key  = f"{symbol}_{strike:.3f}_{option_type}"
    positions   = {strike_key: 0}
    book        = {strike_key: []}
    next_id     = 1

    # --- (3) The minute‐by‐minute loop ---
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Trading'):
        curr_qty = positions[strike_key]
        total_avail = int(row['volume']) if not np.isnan(row['volume']) else 0
        edge       = row['edge']
        bs_ask     = row['bs_price'] - row['ask']
        bs_bid     = row['bid']      - row['bs_price']

        exit_signal = get_exit_signal(edge, edge_exit)

        # PHASE A: CLOSE?
        want_close = (
            (curr_qty > 0 and exit_signal in (1,2)) or
            (curr_qty < 0 and exit_signal in (-1,2))
        )

        if want_close and curr_qty != 0 and total_avail > 0:
            to_close = abs(curr_qty)
            closed = 0
            pnl = 0.0
            close_ids = []
            price = row['bid'] if curr_qty>0 else row['ask']
            # FIFO through the book
            while closed < to_close and book[strike_key] and closed < total_avail:
                lot = book[strike_key].pop(0)
                closed += abs(lot['qty'])
                pnl += lot['qty'] * (price - lot['price']) * 100
                close_ids.append(lot['id'])
            pnl = round(pnl)

            # log one aggregate close
            this_id = next_id; next_id += 1
            trade_logs.append({
                'id':               this_id,
                'timestamp':        row['datetime'],
                'symbol':           strike_key,
                'action':           'close',
                'side':             'sell' if curr_qty>0 else 'buy',
                'price':            price,
                'quantity':         closed,
                'position_after':   curr_qty - np.sign(curr_qty)*closed,
                'corresponding_ids':close_ids,
                'pnl':              pnl
            })

            # update position & remaining volume
            curr_qty -= np.sign(curr_qty) * closed
            positions[strike_key] = curr_qty
            total_avail -= closed
            # fall through to Phase B if flat

        # PHASE B: OPEN from flat
        opened_this_bar = False
        if curr_qty == 0 and total_avail > 0:
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
                    'symbol':           strike_key,
                    'action':           'open',
                    'side':             side,
                    'price':            price,
                    'quantity':         abs(open_qty),
                    'position_after':   open_qty,
                    'corresponding_ids':[],
                    'pnl':              0.0
                })
                book[strike_key].append({'id': this_id, 'qty': open_qty, 'price': price})
                curr_qty = open_qty
                positions[strike_key] = curr_qty
                total_avail -= abs(open_qty)
                opened_this_bar = True

        # PHASE C: ADD to existing (only if we didn't just open)
        if curr_qty != 0 and total_avail > 0:
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
                            'symbol':           strike_key,
                            'action':           'add',
                            'side':             side,
                            'price':            price,
                            'quantity':         abs(add_amt),
                            'position_after':   curr_qty + add_amt,
                            'corresponding_ids':[],
                            'pnl':              0.0
                        })
                        book[strike_key].append({'id': this_id, 'qty': add_amt, 'price': price})
                        curr_qty           += add_amt
                        positions[strike_key] = curr_qty
                        total_avail        -= abs(add_amt)

    # --- (4) Build trades_df & back‑fill opens’ corresponding_ids ---
    trades_df = pd.DataFrame(trade_logs)
    # map open_id → [close_ids…]
    close_map = {}
    for _, r in trades_df[trades_df.action=='close'].iterrows():
        for oid in r.corresponding_ids:
            close_map.setdefault(oid, []).append(r.id)
    # assign to opens
    is_open_or_add = trades_df.action.isin(['open', 'add'])
    trades_df.loc[is_open_or_add, 'corresponding_ids'] = trades_df.id.map(lambda oid: close_map.get(oid, []))

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
    run_backtest(fetch_data=False)
