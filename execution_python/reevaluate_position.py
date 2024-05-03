import os
import numpy as np
import datetime
import csv
import pandas as pd
import json
import math
import requests

base_path = r'C:\Users\georg\OneDrive\Documents\st0nks\forex'
units_per_e = 500
lstm_length = 50
bet_size = 5
margin_max_valid_point = 0.0004

def get_path(path_obj):
    if 'item' not in path_obj.keys():
        raise Exception('get_path requires item field.')
    if path_obj['item'] == 'account_metadata':
        return os.path.join(base_path, 'state', 'account_metadata.json')
    if path_obj['item'] == 'account_metadata_stream':
        return os.path.join(base_path, 'streaming', 'CLIENTACCOUNTMARGIN_CLIENTACCOUNTMARGIN.txt')
    if path_obj['item'] == 'currency_stream':
        if 'pair' in path_obj.keys():
            return os.path.join(base_path, 'streaming', f'PRICES_PRICE.{env["markets"][path_obj["pair"]]}.txt')
    if path_obj['item'] == 'env':
        return os.path.join(base_path, 'files', 'env.json')
    if path_obj['item'] == 'orders':
        return os.path.join(base_path, 'files', 'orders.json')
    if path_obj['item'] == 'tmp':
        return os.path.join(base_path, 'files', 'tmp.json')
    if path_obj['item'] == 'value_history':
        if 'pair' in path_obj.keys():
            return os.path.join(base_path, 'value_history', f'{path_obj["pair"]}.csv')
    raise Exception('Path not found.')

def read_file(path_obj, default='throw_exception'):
    path = get_path(path_obj)
    if not os.path.exists(path):
        if default == 'throw_exception':
            raise Exception('Did not find file.')
        return default
    if path.endswith('.json'):
        with open(path, 'r') as obj:
            return json.load(obj)
    if path.endswith('.txt'):
        with open(path, 'r') as obj:
            return obj.read()
        
def write_file(path_obj, content, method='w'):
    # Validate file type
    path = get_path(path_obj)
    if len(list(filter(lambda x: path.endswith(x), ['.json', '.txt', '.csv']))) == 0:
        raise Exception('Invalid file type.')
    # Write content using text writer
    if path.endswith('.json'):
        content = json.dumps(content, indent=2)
    if path.endswith('.json') or path.endswith('.txt'):
        with open(path, method) as obj:
            obj.write(content)
    # Write content to CSV
    if path.endswith('.csv'):
        with open(path, method, newline='') as obj:
            csv_writer = csv.writer(obj)
            if isinstance(content[0], list):
                csv_writer.writerows(content)
            else:
                csv_writer.writerow(content)

env = read_file({'item': 'env'})
tmp = read_file({'item': 'tmp'})

# Get most recent account metrics
def get_account_metadata():
    def find_latest_valid_line(metadata_lines):
        for ml in list(reversed(metadata_lines)):
            if len(ml) == 0:
                continue
            ml = json.loads(ml)
            ml = {k: float(ml[k]) for k in ml.keys()}
            return ml
        return None
    
    metadata_stream = read_file({'item': 'account_metadata_stream'}, default=None)
    if not metadata_stream:
        return None
    
    metadata_lines = metadata_stream.split('\n')
    new_metadata_line = find_latest_valid_line(metadata_lines)
    if not new_metadata_line:
        return None
    
    # Purely for viewing pleasure
    write_file({'item': 'account_metadata'}, new_metadata_line)
    return new_metadata_line
    

# Streaming text --> new line to update value history
def ingest_text(pair):
    def find_latest_valid_line(pair_lines):
        for pl in list(reversed(pair_lines)):
            if len(pl) == 0:
                continue
            pl = json.loads(pl)
            pl['bid'] = float(pl['Bid'])
            pl['ask'] = float(pl['Offer'])
            if pl['ask'] < (np.exp(margin_max_valid_point) * pl['bid']):
                return pl
        return None

    pair_stream = read_file({'item': 'currency_stream', 'pair': pair}, default=None)
    pair_file_path = get_path({'item': 'currency_stream', 'pair': pair})
    if os.path.exists(pair_file_path):
        os.remove(pair_file_path)
    if not pair_stream:
        return None
    
    pair_lines = pair_stream.split('\n')
    new_pair_line = find_latest_valid_line(pair_lines)
    if not new_pair_line:
        return None
    
    bid = units_per_e * np.log(new_pair_line['bid'])
    ask = units_per_e * np.log(new_pair_line['ask'])
    new_pair_line = {
        'bid': bid,
        'ask': ask,
        'value': np.mean([bid, ask]),
        'margin': ask - bid,
        'timestamp': new_pair_line['timestamp']
    }
    return new_pair_line

# Pair history --> LSTM input
def get_lstm_input(pair, pair_history):
    ticker_num = list(env['markets'].keys()).index(pair)
    value_delta = np.diff(pair_history.value)
    business_day_timestamps = list(map(lambda ts: ts - (86400 * 2) * math.floor((ts - datetime.datetime.strptime('2014-04-19 00:00', '%Y-%m-%d %H:%M').timestamp()) / (86400 * 7)), pair_history.timestamp))
    time_delta_log = np.log(list(map(lambda x: min(max(x, 1), np.exp(16)), np.diff(np.array(business_day_timestamps)))))
    weekdays = list(map(lambda ts: min(int(datetime.datetime.fromtimestamp(ts).strftime('%w')), 5), pair_history.timestamp))
    hour_chunks = list(map(lambda ts: math.floor(int(datetime.datetime.fromtimestamp(ts).strftime('%H')) / 4), pair_history.timestamp))
    pair_data = pair_history.iloc[1:, ].assign(
        value_delta = value_delta, 
        time_delta_log = time_delta_log,
        weekdays = weekdays[1:],
        hour_chunks = hour_chunks[1:],
        ticker_num = ticker_num
    )
    # Returns 3D (1 x lstm length x input dim) numpy array. 
    # Model predictions on this give you a single value.
    return np.concatenate(
        (
            np.array(pair_data.value_delta)[:, np.newaxis],
            np.array((pair_data.time_delta_log - 8) / 8)[:, np.newaxis],
            np.eye(6)[pair_data.weekdays.astype('int')],
            np.eye(6)[pair_data.hour_chunks.astype('int')],
            np.eye(len(env['markets'].keys()))[pair_data.ticker_num.astype('int')],
        ), axis=1)[np.newaxis, :, :]

# LSTM input --> binary prediction (-1 vs. 1)
def predict_bet(lstm_input):
    from keras.models import load_model
    model_list = [load_model(os.path.join(base_path, 'models', file)) for file in os.listdir(os.path.join(base_path, 'models'))]
    model_predictions = np.array([model.predict(lstm_input)[:, 0] for model in model_list])
    return 2 * (np.median((model_predictions), axis=0) > 0.5) - 1

# Make order with desired quantity
def make_order(pair, desired_quantity, new_pair_line, orders):

    def convert_to_real_price(log_price):
        return round(math.exp(log_price / units_per_e), 6)
    
    payload = {
        "MarketId": env['markets'][pair],
        "Direction": "Buy" if desired_quantity > 0 else "Sell",
        "Quantity": abs(desired_quantity),
        "BidPrice": convert_to_real_price(new_pair_line['bid']),
        "OfferPrice": convert_to_real_price(new_pair_line['ask']),
        "Reference": "GCAPI",
        "PriceTolerance": 1,
        "TradingAccountId": env['auth']['trading_account_id']
    }

    # Send request
    response = requests.post(
        f"{env['auth']['base_url']}/order/newtradeorder",
        headers={'Content-Type': 'application/json'},
        params = {
            'UserName': env['auth']['username'],
            'Session': tmp['session_id']
        },
        json = payload)
    # Buttress payload with additional context
    payload.update({
        "timestamp": datetime.datetime.now().timestamp(),
        "pair": pair,
        "margin": new_pair_line['margin'],
        "strike_price": new_pair_line['value']
    })
    # Add success/error information
    if (response.status_code == 200):
        order_response = json.loads(response.content.decode("utf-8"))
        print('response')
        print(order_response)
        payload.update({
            "Status": order_response['Status'],
            "OrderId": order_response['OrderId']
        })
    else:
        err = f"Observed status code {response.status_code}"
        print(err)
        payload.update({
            'error': err,
            'Status': None
        })
    orders = orders + [payload]
    write_file({'item': 'orders'}, orders)
    return orders


def main():
    orders = read_file({'item': 'orders'}, default=[])
    orders = list(filter(lambda o: 
        not ((o['Status'] == 2) and ((datetime.datetime.now().timestamp() - o['timestamp']) > 900))
        , orders))

    account_metadata = get_account_metadata()
    if not account_metadata:
        print('Unable to retrieve account metadata.')
        return None
    
    if account_metadata['NetEquity'] < 3000:
        print('Balance is too low to submit orders.')
        return None

    for pair in env['markets'].keys():
        # Processes currency streams, updates value history.
        new_pair_line = ingest_text(pair)
        if not new_pair_line:
            continue
        
        # Stop processing for this currency if there's an error and/or an active bet that still needs to bake.
        matching_orders = list(filter(lambda o: o['pair'] == pair, orders))
        unexpired_orders = list(filter(lambda o: ('error' in o.keys()) or
                (abs(new_pair_line['value'] - o['strike_price']) < bet_size)
                , matching_orders))
        if len(unexpired_orders) > 0:
            continue

        # Make a binary prediction for this pair based on feeding LSTM-friendly data through existing models.
        pair_history = pd.read_csv(os.path.join(base_path, 'value_history', f'{pair}.csv'))
        if pair_history.shape[0] < (lstm_length + 1):
            continue
        lstm_input = get_lstm_input(pair, pair_history.tail(lstm_length + 1))
        bet_prediction = predict_bet(lstm_input)
        # Bet = 
        # direction (1 vs -1) depending on model
        # * (80% of balance / number of pairs)
        # * Leverage for pair (restricted to 20x)
        # / value of base currency in USD, converting the desired amount in USD --> that currency
        if (not pair.startswith('USD')) and (not pair.endswith('USD')):
            raise Exception('Base currency calculation assumes USD is part of pair.')
        base_currency_value = 1 if pair.endswith('USD') else np.exp(new_pair_line['value'] / units_per_e)
        new_position = bet_prediction * account_metadata['NetEquity'] * (0.8 / len(env['markets'].keys())) * env['leverage'][pair] / base_currency_value
        new_position = int(np.sign(new_position) * (1000 * math.floor(abs(new_position) / 1000)))
        existing_position = sum(list(map(lambda b: b['Quantity'], matching_orders)))
        orders = make_order(pair, new_position - existing_position, new_pair_line, orders)

    write_file({'item': 'orders'}, orders)

        
main()
