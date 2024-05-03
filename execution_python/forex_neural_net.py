import pandas as pd
import os
import numpy as np
import keras
from keras import layers
import datetime
import math
from keras.models import load_model
import json

# Idea: every time you make a call/put, it either:
# Succeeds (bet is liquidated after pair moves 1% towards desired direction)
# Fails (bet is liquidated after pair moves 1% against desired direction)
# This betting strategy can be managed with a binary classification algorithm.

base_path = r'C:\Users\georg\OneDrive\Documents\st0nks\forex'

units_per_e = 500 # every 500 units = currency value multiplies by e; 1 unit = an 0.2% increase

# Trains multiple LSTM models
def train_models(train_in, train_out, test_in, test_out, epochs_list, copies):
    def create_model():
        # Keep model as small as possible to prevent overfitting
        model = keras.Sequential()
        model.add(layers.LSTM(32, activation='tanh'))
        model.add(layers.Dense(32, activation='tanh'))
        model.add(layers.Dense(1, activation='sigmoid')) # Sigmoid returns 0-1 value for binary classification
        return model
    
    # Low batch size & low learning rate enhances model stability while taking slightly longer to execute
    vb = 2
    batch_size = 8
    model_list = []

    for copy in range(copies):
        model_path = os.path.join(base_path, 'models', f'model_{copy}.h5')
        for iteration in range(len(epochs_list)):
            model = create_model() if (iteration == 0) else load_model(model_path)
            model.compile(
                loss = 'binary_crossentropy',
                optimizer = keras.optimizers.Adam(learning_rate=0.0004),
                metrics=['binary_accuracy']
            ) 
            model.fit(train_in, train_out, 
                batch_size = batch_size, 
                epochs = (epochs_list[iteration] - epochs_list[iteration - 1]) if (iteration > 0) else epochs_list[iteration],
                verbose=vb,
                validation_data=(test_in, test_out)
            )
            model.save(model_path)
            model_list.append(model)
    return model_list

def model_predict(model_list, test_in):
    model_predictions = np.array(list(map(lambda model: model.predict(test_in)[:, 0], model_list)))
    return 1 * (np.median((model_predictions), axis=0) > 0.5)


ticker_list = ['AUDUSD', 'USDCAD', 'USDCHF', 'EURUSD', 'USDJPY', 'GBPUSD']
floor_timestamp = datetime.datetime.strptime('2014-04-19 00:00', '%Y-%m-%d %H:%M').timestamp()
bet_size = 5
lstm_length = 50
for ticker_num, ticker in enumerate(ticker_list):
    # Load data
    pair_data = pd.read_csv(os.path.join(r'C:\Users\georg\OneDrive\Documents\st0nks\forex\value_history', f'{ticker}.csv'))
    pair_data = pair_data.assign(ticker_num = ticker_num)
    # Get gain/loss and time out of it
    value_delta = np.diff(pair_data.value)
    pair_timestamps = list(map(lambda ts: ts - (86400 * 2) * math.floor((ts - floor_timestamp) / (86400 * 7)), pair_data.timestamp))
    # Time delta, accounting for the existence of weekends where markets are closed.
    time_delta_log = np.log(list(map(lambda x: min(max(x, 1), np.exp(16)), np.diff(np.array(pair_timestamps)))))
    # 0-5 (5 = saturday & sunday)
    weekdays = list(map(lambda ts: min(int(datetime.datetime.fromtimestamp(ts).strftime('%w')), 5), pair_data.timestamp))
    # 0-5 = 4-hour chunks of the day. Dataset is in UTC time
    hour_chunks = list(map(lambda ts: math.floor(int(datetime.datetime.fromtimestamp(ts).strftime('%H')) / 4), pair_data.timestamp))

    pair_data = pair_data.iloc[1:, ].assign(
        value_delta = value_delta, 
        time_delta_log = time_delta_log,
        weekdays = weekdays[1:],
        hour_chunks = hour_chunks[1:]
    )
    pair_data.index = range(0, pair_data.shape[0])
    # Get indexes for 0.5% chances in forex pair
    pair_data = pair_data.assign(overall_reward = 0)
    # Indices to bet on
    with open(os.path.join(base_path, 'act_indices.json'), 'r') as obj:
        act_indices = json.load(obj)
    overall_reward = np.diff(pair_data.loc[act_indices, 'value'])
    pair_data.loc[act_indices[1:], 'overall_reward'] = overall_reward

    # Set LSTM I/O based on pair data
    act_indices = list(filter(lambda x: x >= lstm_length, act_indices[1:]))
    pair_data = pair_data.assign(gain = False, weight = 1)
    pair_data.loc[act_indices, 'gain'] = pair_data.loc[act_indices, 'overall_reward'] > 0
    pair_data.loc[act_indices, 'weight'] = np.abs(pair_data.loc[act_indices, 'overall_reward']) / bet_size

    def get_io(pair_data, act_indices):
        lstm_inputs_raw = np.array([np.array(pair_data.loc[(act_indices[act_index_num - 1]-lstm_length+1):(act_indices[act_index_num - 1]), ].loc[:, ['value_delta', 'time_delta_log', 'weekdays', 'hour_chunks', 'ticker_num']]) for act_index_num in range(1, len(act_indices))])
        lstm_input = np.concatenate(
            (
                lstm_inputs_raw[:, :, 0][:, :, np.newaxis], # Value delta
                ((lstm_inputs_raw[:, :, 1] - 8) / 8)[:, :, np.newaxis], # Time delta
                np.eye(6)[lstm_inputs_raw[:, :, 2].astype('int')], # Categorical, weekday
                np.eye(6)[lstm_inputs_raw[:, :, 3].astype('int')], # Categorical, time of day
                np.eye(len(ticker_list))[lstm_inputs_raw[:, :, 4].astype('int')] # Categorical, currency
            ), axis=2)
        lstm_output = 1 * pair_data.loc[act_indices[1:], ].gain
        weights = pair_data.loc[act_indices[1:], ].weight
        margins = pair_data.loc[act_indices[1:], ].margin
        return lstm_input, lstm_output, weights, margins

    train_cutoff = 0.7 * pair_data.shape[0]
    train_in, train_out, train_weights, train_margins = get_io(pair_data, list(filter(lambda x: x < train_cutoff, act_indices)))
    test_in, test_out, test_weights, test_margins = get_io(pair_data, list(filter(lambda x: x >= train_cutoff, act_indices)))

    if ticker_num == 0:
        train_in_all, train_out_all, train_weights_all, train_margins_all = train_in, train_out, train_weights, train_margins
        test_in_all, test_out_all, test_weights_all, test_margins_all =  test_in, test_out, test_weights, test_margins
    else:
        train_in_all = np.concatenate((train_in_all, train_in), axis=0)
        train_out_all = np.concatenate((train_out_all, train_out), axis=0)
        train_weights_all = np.concatenate((train_weights_all, train_weights), axis=0)
        train_margins_all = np.concatenate((train_margins_all, train_margins), axis=0)

        test_in_all = np.concatenate((test_in_all, test_in), axis=0)
        test_out_all = np.concatenate((test_out_all, test_out), axis=0)
        test_weights_all = np.concatenate((test_weights_all, test_weights), axis=0)
        test_margins_all = np.concatenate((test_margins_all, test_margins), axis=0)

# Using multiple models is more stable than using a single model for use cases with low signal-to-noise ratios
model_list = train_models(train_in_all, train_out_all, test_in_all, test_out_all, [10], 15)

# Benchmarks
train_prediction = model_predict(model_list, train_in_all)
train_prediction_accuracy = np.mean(train_out_all == train_prediction)
prediction = model_predict(model_list, test_in_all)
prediction_accuracy = np.mean(test_out_all == prediction)
