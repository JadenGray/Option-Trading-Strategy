
import matplotlib.pyplot as plt

# make the pnl plot
def plot_pnl(pnl_daily, symbol, option_type, strike):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(pnl_daily['datetime'], pnl_daily['cumulative_pnl'], label='Cumulative PnL')
    ax.set_title(f"{symbol} {option_type.title()} {strike} Backtest")
    ax.set_xlabel('Date')
    ax.set_ylabel('PnL')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig