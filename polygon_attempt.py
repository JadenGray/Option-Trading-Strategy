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


def build_option_ticker(symbol: str, expiry_date: date, strike: float, option_type: str) -> str:
    yy, mm, dd = expiry_date.year % 100, expiry_date.month, expiry_date.day
    letter = 'C' if option_type.lower().startswith('c') else 'P'
    strike_int = int(round(strike * 1000))
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
                 vol_half_life_min: float = 11,
                 warmup_minutes: int = 30,
                 max_pos: int = 10,
                 fetch_data: bool = True):
    print(f"Running backtest with expiry in {expiry_delta_days}d and vol HL={vol_half_life_min}m")

     # -- Fetch/load data --
    if fetch_data:
        u_df = fetch_underlying(symbol, window_start, window_end)
        trading_days = pd.to_datetime(u_df['datetime']).dt.date.unique()
        opt_frames = []
        for d in tqdm(sorted(trading_days), desc="Fetching option bars"):
            exp = adjust_to_business_day(d + timedelta(days=expiry_delta_days))
            ticker = build_option_ticker(symbol, exp, strike, option_type)
            bars = fetch_option_bars(ticker, d.strftime('%Y-%m-%d'))
            if not bars.empty:
                bars['expiry_date'] = pd.to_datetime(exp)
                opt_frames.append(bars)
            time.sleep(0.2)
        opt_df = pd.concat(opt_frames, ignore_index=True)
    else:
        u_df = pd.read_csv('underlying.csv', parse_dates=['datetime'])
        opt_df = pd.read_csv('option_bars.csv', parse_dates=['datetime','expiry_date'])

    # -- Align timestamps; drop rows without options --
    u_df = u_df.sort_values('datetime').set_index('datetime')
    opt_df = opt_df.sort_values('datetime').set_index('datetime')
    df = pd.merge_asof(
        u_df, opt_df,
        left_index=True, right_index=True,
        direction='nearest', tolerance=pd.Timedelta('1min')
    ).dropna(subset=['option_symbol']).rename(columns={'price':'underlying_price'})
    df = df.reset_index()

    # Compute time-to-expiry T
    df['bid'], df['ask'] = df['open'], df['close']
    df['T'] = (df['expiry_date'] - df['datetime']).dt.total_seconds() / (365*24*3600)

    # Volatility EWMA; drop warmup
    ret = np.log(df['underlying_price']).diff()
    alpha = 1 - np.exp(np.log(0.5) / vol_half_life_min)
    df['sigma'] = np.sqrt(ret.ewm(alpha=alpha).var() * 252 * 6.5 * 60)
    df = df.iloc[warmup_minutes:].copy()

    # Pricing & edge
    df['bs_price'] = black_scholes_vectorized(
        df['underlying_price'], strike, df['T'], r, df['sigma'], option_type
    )
    df['edge'] = np.maximum(df['bs_price'] - df['ask'], df['bid'] - df['bs_price'])

    # -- Trading logic with position tracking --
    trade_logs = []  # execution logs
    positions = {}   # sym -> current signed qty

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Trading"):
        sym = row['option_symbol']
        avail = int(row['volume']) if not np.isnan(row['volume']) else 0
        edge = row['edge']
        # compute signal and target
        sig = 1 if (row['bs_price'] - row['ask'] > edge_entry) else (-1 if (row['bid'] - row['bs_price'] > edge_entry) else 0)
        # force-close condition
        if sig == 0 or edge < edge_exit:
            target_qty = 0
        else:
            base_size = size_from_edge(edge, avail, edge_entry, edge_exit)
            target_qty = sig * min(base_size, max_pos)
        curr_qty = positions.get(sym, 0)
        delta = target_qty - curr_qty
        if delta != 0 and avail > 0:
            exec_qty = int(np.sign(delta) * min(abs(delta), avail))
            side = 'buy' if exec_qty > 0 else 'sell'
            price = row['ask'] if exec_qty > 0 else row['bid']
            # determine action
            if curr_qty == 0:
                action = 'open'
            elif np.sign(delta) == np.sign(curr_qty):
                action = 'add'
            else:
                action = 'close'
            new_qty = curr_qty + exec_qty
            positions[sym] = new_qty
            trade_logs.append({
                'timestamp': row['datetime'],
                'symbol': sym,
                'action': action,
                'side': side,
                'price': price,
                'quantity': abs(exec_qty),
                'position_after': new_qty
            })

from collections import deque

# Running trade record and position tracker
trades = []
positions = {}  # key: option_symbol, value: deque of (price, size)

for index, row in df.iterrows():
    ts = row['timestamp']
    symbol = row['option_symbol']
    price = row['mid_price']
    volume = row['volume']
    signal = row['signal']
    edge = row['edge']
    
    # Get current pos
    pos = cum_positions.get(symbol, 0)
    
    if signal == 0 or abs(edge) < edge_exit:
        # Close all
        trade_size = abs(pos)
        if trade_size == 0:
            continue  # nothing to close
        
        action = 'sell' if pos > 0 else 'buy'
        close_qty = min(trade_size, volume)
        
        # FIFO PnL calc
        pnl = 0
        if symbol not in positions:
            positions[symbol] = deque()
        
        position_book = positions[symbol]
        if pos > 0:
            # Selling, so match with earlier buys
            remaining = close_qty
            while remaining > 0 and position_book:
                open_price, size = position_book[0]
                matched = min(remaining, size)
                pnl += matched * (price - open_price)
                if matched == size:
                    position_book.popleft()
                else:
                    position_book[0] = (open_price, size - matched)
                remaining -= matched
        else:
            # Buying, so match with earlier sells
            remaining = close_qty
            while remaining > 0 and position_book:
                open_price, size = position_book[0]
                matched = min(remaining, size)
                pnl += matched * (open_price - price)
                if matched == size:
                    position_book.popleft()
                else:
                    position_book[0] = (open_price, size - matched)
                remaining -= matched
        
        # Update cum pos
        cum_positions[symbol] = pos - close_qty if pos > 0 else pos + close_qty
        
        trades.append({
            'timestamp': ts,
            'option_symbol': symbol,
            'action': action,
            'price': price,
            'volume': close_qty,
            'position_after': cum_positions[symbol],
            'pnl': pnl
        })
    
    elif signal != 0 and abs(pos) < max_pos:
        # Add position
        delta = np.sign(signal) * min(3, max_pos - abs(pos), volume)
        if delta == 0:
            continue
        
        action = 'buy' if delta > 0 else 'sell'
        if symbol not in positions:
            positions[symbol] = deque()
        
        # Record opening leg for PnL
        positions[symbol].append((price, abs(delta)))
        cum_positions[symbol] = pos + delta
        
        trades.append({
            'timestamp': ts,
            'option_symbol': symbol,
            'action': action,
            'price': price,
            'volume': abs(delta),
            'position_after': cum_positions[symbol],
            'pnl': 0.0
        })

# Build DataFrame
trades_df = pd.DataFrame(trades)

     # -- Daily PnL --
    if not trades_df.empty:
        closes = trades_df[trades_df['action']=='close']
        daily = closes.copy()
        daily['date'] = daily['timestamp'].dt.date
        pnl_daily = daily.groupby('date')['pnl'].sum().reset_index()
        pnl_daily.columns = ['datetime','daily_pnl']
        pnl_daily['cumulative_pnl'] = pnl_daily['daily_pnl'].cumsum()
        pnl_daily.to_csv('daily_pnl.csv', index=False)
    else:
        pnl_daily = pd.DataFrame(columns=['datetime','daily_pnl','cumulative_pnl'])

    # -- Plot --
    plt.figure(figsize=(10,4))
    plt.plot(pnl_daily['datetime'], pnl_daily['cumulative_pnl'], label='Cumulative PnL')
    plt.title(f"{symbol} {option_type.title()} {strike} backtest")
    plt.xlabel('Date'); plt.ylabel('PnL'); plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    return df, trades_df, pnl_daily

if __name__ == '__main__':
    run_backtest(fetch_data=False)