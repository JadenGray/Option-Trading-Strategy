import numpy as np
import pandas as pd

# first, close any open positions if we get the signal
def close_positions(ts, row, positions, book, trade_logs, next_id, exit_signal):
    curr_qty   = positions # current position
    total_avail= int(row['available']) # market availablity
    if ((curr_qty > 0 and exit_signal in (1,2)) or
        (curr_qty < 0 and exit_signal in (-1,2))) and total_avail > 0: # signals to close

        # initialise variables to allow for closing
        to_close  = abs(curr_qty)
        closed    = 0
        pnl       = 0.0
        close_ids = []
        price     = row['bid'] if curr_qty > 0 else row['ask'] # price we will close at (reverse trade)

        while closed < to_close and book and closed < total_avail:
            # take the first lot in the book and close it
            lot       = book.pop(0)
            lot_qty   = lot['qty']
            entry_pr  = lot['price']

            # compute pnl
            pnl += (lot_qty * (price - entry_pr) if lot_qty>0
                    else -lot_qty * (entry_pr - price))
            # note the id of the trade that we closed
            close_ids.append(lot['id'])
            closed += abs(lot_qty)

        pnl     = round(pnl, 3) # small float debug
        this_id = next_id; next_id += 1 # move on to the next trade

        # log the close trade
        trade_logs.append({
            'id':               this_id,
            'timestamp':        row['datetime'],
            'symbol':           row['option_symbol'],
            'action':           'close',
            'side':             'sell' if curr_qty>0 else 'buy',
            'price':            price,
            'quantity':         closed,
            'position_after':   curr_qty - np.sign(curr_qty)*closed,
            'corresponding_ids':close_ids,
            'pnl':              pnl
        })
        # update the position and return the updated state
        positions -= np.sign(curr_qty)*closed
    return positions, book, trade_logs, next_id

# if no close then look to open a position
def open_positions(row, positions, book, trade_logs, next_id,
                   size_from_edge, edge_entry, edge_exit):
    
    curr_qty   = positions
    total_avail= int(row['available'])
    # get the size of the trade
    size       = size_from_edge(row['edge'], total_avail, edge_entry, edge_exit)
    # if conditions allow and we get the signal
    if curr_qty == 0 and size != 0 and total_avail > 0:
        # make the trade and update the position
        open_qty = size
        price    = row['ask'] if open_qty>0 else row['bid']
        side     = 'buy' if open_qty>0 else 'sell'

        this_id = next_id; next_id += 1
        trade_logs.append({
            'id':               this_id,
            'timestamp':        row['datetime'],
            'symbol':           row['option_symbol'],
            'action':           'open',
            'side':             side,
            'price':            price,
            'quantity':         abs(open_qty),
            'position_after':   open_qty,
            'corresponding_ids':[],
            'pnl':              0.0
        })
        book.append({'id': this_id, 'qty': open_qty, 'price': price})
        positions = open_qty
    return positions, book, trade_logs, next_id

def add_positions(row, positions, book, trade_logs, next_id,
                  size_from_edge, edge_entry, edge_exit, max_pos):
    # same as opening, can adjust to include maximum position at some point
    curr_qty    = positions
    total_avail = int(row['available'])
    size        = size_from_edge(row['edge'], total_avail, edge_entry, edge_exit)
    desired     = curr_qty+size
    if abs(desired) >= max_pos:
        desired = max_pos * np.sign(desired)
    delta       = desired - curr_qty

    if curr_qty != 0 and delta * curr_qty > 0 and total_avail > 0:
        add_amt = int(delta)
        price   = row['ask'] if add_amt>0 else row['bid']
        side    = 'buy'  if add_amt>0 else 'sell'

        this_id = next_id; next_id += 1
        trade_logs.append({
            'id':               this_id,
            'timestamp':        row['datetime'],
            'symbol':           row['option_symbol'],
            'action':           'add',
            'side':             side,
            'price':            price,
            'quantity':         abs(add_amt),
            'position_after':   curr_qty + add_amt,
            'corresponding_ids':[],
            'pnl':              0.0
        })
        book.append({'id': this_id, 'qty': add_amt, 'price': price})
        positions += add_amt
    return positions, book, trade_logs, next_id

# at the end of the day we might have to force close a position
def forced_close(subdf, positions, book, trade_logs, next_id):
    # attempt to close regardless of signal at the end
    curr_qty = positions
    if curr_qty != 0 and book:
        last = subdf.iloc[-1]
        price = round((last.bid + last.ask)/2, 3)
        to_close = abs(curr_qty)
        closed   = 0
        pnl      = 0.0
        close_ids= []
        while closed < to_close and book:
            lot        = book.pop(0)
            lot_qty    = lot['qty']
            entry_price= lot['price']

            pnl += (lot_qty * (price - entry_price) if lot_qty>0
                    else -lot_qty * (entry_price - price))
            close_ids.append(lot['id'])
            closed += abs(lot_qty)

        this_id = next_id; next_id += 1
        trade_logs.append({
            'id':               this_id,
            'timestamp':        last['datetime'],
            'symbol':           last['option_symbol'],
            'action':           'close',
            'side':             'sell' if curr_qty>0 else 'buy',
            'price':            price,
            'quantity':         closed,
            'position_after':   0,
            'corresponding_ids':close_ids,
            'pnl':              round(pnl,3)
        })
        positions = 0
    return positions, book, trade_logs, next_id

# run full trading logic
def process_expiry_group(subdf, edge_entry, edge_exit, size_from_edge,
                         get_exit_signal, max_pos):
    trade_logs = []
    next_id    = 1
    positions  = 0
    book       = []

    # filter out expired timestamps (debug)
    subdf = subdf.sort_index()
    subdf = subdf[subdf['T'] > 0]

    for ts, row in subdf.iterrows():
        # is this one of the last two bars?
        is_final_two = ts in subdf.index[-2:]
        exit_signal  = 2 if is_final_two else get_exit_signal(row['edge'], edge_exit)

        # closing first
        positions, book, trade_logs, next_id = close_positions(
            ts, row, positions, book, trade_logs, next_id, exit_signal
        )
        # if not, open
        positions, book, trade_logs, next_id = open_positions(
            row, positions, book, trade_logs, next_id,
            size_from_edge, edge_entry, edge_exit
        )
        # if not, add
        positions, book, trade_logs, next_id = add_positions(
            row, positions, book, trade_logs, next_id,
            size_from_edge, edge_entry, edge_exit, max_pos
        )

    # finally, force‐close leftovers at the end of the day
    positions, book, trade_logs, next_id = forced_close(subdf, positions, book, trade_logs, next_id)
    return trade_logs