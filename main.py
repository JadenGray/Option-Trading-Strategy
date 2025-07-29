from src import run_backtest
from src.plotting import plot_pnl
from dotenv import load_dotenv
load_dotenv()
import random
import numpy as np
SEED = 100
random.seed(SEED)
np.random.seed(SEED)


# example usage

# run backtest with default parameters (fetch data set to false) 
df, trades, pnl = run_backtest()

# save outputs
df.to_csv("data/bid_asks.csv", index=False)
trades.to_csv("data/trades.csv", index=False)
pnl.to_csv("data/daily_pnl.csv", index=False)

# generate and save PnL plot
fig = plot_pnl(pnl, df['symbol'].iloc[0] if 'symbol' in df.columns else 'SPY', df['option_type'].iloc[0] if 'option_type' in df.columns else 'call', pnl.columns[0])
fig.savefig("plots/pnl_plot.png")

print("Backtest complete, Saved PnL plots and csv files")