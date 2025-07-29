import pandas as pd
import numpy as np
from tqdm import tqdm
from .run_phases import process_expiry_group
from .pricing_logic import get_exit_signal, size_from_edge

def remap_corresponding_ids(trades_df: pd.DataFrame) -> pd.DataFrame:
    # ensure IDs are ints (debug)
    trades_df['id'] = trades_df['id'].astype(int)

    # build open to closes map
    open_to_closes: dict[int, list[int]] = {}
    for _, close_row in trades_df[trades_df.action == 'close'].iterrows():
        # get the close id
        close_id = int(close_row['id'])
        # add it to the corresponding_ids
        for oid in close_row['corresponding_ids']:
            oid = int(oid)
            open_to_closes.setdefault(oid, []).append(close_id)

    # apply mapping per row
    def _map_row(row):
        row_id = int(row['id'])
        if row['action'] in ('open', 'add'):
            return [open_to_closes.get(row_id, [])[0]]
        else:
            return row['corresponding_ids']
    # update the trades df
    trades_df['corresponding_ids'] = trades_df.apply(_map_row, axis=1)
    
    return trades_df


# make the trade logs
def generate_trade_log(df, edge_entry, edge_exit, max_pos):
    # randomly generate market availability - designed to simulate some market restrictions
    df['available'] = np.random.randint(1, 6, size=len(df))
    trade_logs = []
    # loop through each option
    for opt_sym, subdf in tqdm(df.groupby('option_symbol'), desc='Expiry Groups'):
        # make the trades
        logs = process_expiry_group(
            subdf,
            edge_entry=edge_entry,
            edge_exit=edge_exit,
            size_from_edge=size_from_edge,
            get_exit_signal=get_exit_signal,
            max_pos=max_pos
        )
        # put the trades in the log
        trade_logs.extend(logs)
    # update the corresponding ids
    trades_df = pd.DataFrame(trade_logs)
    trades_df = remap_corresponding_ids(trades_df)
      
    return trades_df

# function to sum over the trades and compute the final daily pnl
def compute_daily_pnl(trades_df):
    # empty trades day (debug)
    if trades_df.empty:
        return pd.DataFrame(columns=['datetime','daily_pnl','cumulative_pnl'])
    
    # only record pnl on close trades
    closes = trades_df[trades_df.action == 'close'].copy()
    closes['date'] = closes['timestamp'].dt.date
    
    # add up the close pnls
    pnl_daily = closes.groupby('date')['pnl'].sum().reset_index()
    # make the df recording day, pnl and cumulative pnl from start of backtest
    pnl_daily.columns = ['datetime', 'daily_pnl']
    pnl_daily['cumulative_pnl'] = pnl_daily['daily_pnl'].cumsum()
    
    return pnl_daily