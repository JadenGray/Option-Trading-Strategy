# import packages
import os
import pandas as pd
import requests
from datetime import date
from pandas.tseries.offsets import BDay

# polygon api key
API_KEY = os.getenv("POLYGON_API_KEY")
BASE_URL = "https://api.polygon.io"

# polygon api url call funciton
def polygon_get(path, params=None):
    p = params.copy() if params else {}
    p['apiKey'] = API_KEY # api key
    r = requests.get(BASE_URL + path, params=p) # use the url to get the data
    r.raise_for_status()
    return r.json() # return the data in json format

# underlying asset data 
def fetch_underlying(symbol, start, end):
    out = polygon_get(
        f"/v2/aggs/ticker/{symbol}/range/1/minute/{start}/{end}",
        {'adjusted':'true','sort':'asc','limit':50000}
    )
    df = pd.DataFrame(out.get('results', [])) # convert to pd df
    df['datetime'] = pd.to_datetime(df['t'], unit='ms')
    return df[['datetime', 'c']].rename(columns={'c': 'price'})

# get the option price data from polygon using api key
def fetch_option_bars(opt_ticker, date_str):
    # option bar data
    out = polygon_get(
        f"/v2/aggs/ticker/{opt_ticker}/range/1/minute/{date_str}/{date_str}",
        {'adjusted':'true','sort':'asc','limit':50000}
    )
    df = pd.DataFrame(out.get('results', [])) # pd df
    if df.empty:
        return df # debug
    df['datetime'] = pd.to_datetime(df['t'], unit='ms')
    df['option_symbol'] = opt_ticker
    # rename the columns for their meaning
    return df[['datetime','o','h','l','c','v','option_symbol']].rename(
        columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'}
    )

# use pandas BDay function to make sure the day is a trading day
def adjust_to_business_day(d: date) -> date:
    return (pd.Timestamp(d) + BDay(0)).date()

# option symbol
def build_option_ticker(symbol, expiry_date, strike, option_type):
    yy, mm, dd = expiry_date.year % 100, expiry_date.month, expiry_date.day # expiry date
    letter = 'C' if option_type.lower().startswith('c') else 'P' # allow for calls and puts
    strike_int = int(round(strike*1000)) # strike price
    return f"O:{symbol}{yy:02d}{mm:02d}{dd:02d}{letter}{strike_int:08d}" # return the symbol in the correct format