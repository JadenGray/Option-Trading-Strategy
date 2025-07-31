from src import run_backtest
from src.plotting import plot_pnl
from dotenv import load_dotenv
load_dotenv()
import random
import numpy as np
import re
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
# get the strike price for the title
option_symbol = df.loc[0, 'option_symbol'][2:]
match = re.search(r'[A-Z]+(\d{6})([CP])(\d{8})$', option_symbol)
strike_price =int(int(match.group(3)) / 1000)
fig = plot_pnl(pnl, df['symbol'].iloc[0] if 'symbol' in df.columns else 'SPY', df['option_type'].iloc[0] if 'option_type' in df.columns else 'call', strike_price)
fig.savefig("plots/pnl_plot.png")

print("Backtest complete, Saved PnL plots and csv files")