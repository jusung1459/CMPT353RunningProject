import os
import pathlib
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timezone
import matplotlib.pyplot as plt
import warnings
from scipy import signal

## arg data: pass in the returned cleaned data from combine
def custom_filter(data):
    for i in range(0,len(data)):
        b, a = signal.butter(3, 0.35, btype='lowpass', analog=False)
        low_passed_butter = signal.filtfilt(b, a, data[i]['gFx'])
        data[i]['filtered_gFx'] = low_passed_butter

        b, a = signal.butter(3, 0.35, btype='lowpass', analog=False)
        low_passed_butter = signal.filtfilt(b, a, data[i]['gFy'])
        data[i]['filtered_gFy'] = low_passed_butter

        b, a = signal.butter(3, 0.35, btype='lowpass', analog=False)
        low_passed_butter = signal.filtfilt(b, a, data[i]['gFz'])
        data[i]['filtered_gFz'] = low_passed_butter

    return data

def parseAccl(data, splits):
    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    data['time'] = data['time'] - data['time'][0]
    # data['time'] = pd.to_datetime(data['time'],format= '%H:%M:%S' ).dt.time
    # data['time'] = data['time'].values.astype(np.int64) // 10 ** 6

    data['timestamp'] = data['time'].round('50ms')
    data = data.groupby(['timestamp'], as_index=False).mean()
    # print(data)
    splitted_data = []
    for split in splits:
        # print(split)
        index = data.loc[data['timestamp'] == split].index[0]
        splitted_data.append(data.iloc[index:index+2400,]) ## 120s * 10
    return splitted_data

def parseSplits(data):
    data['start_time'] = data['start_time'] - data['start_time'][0]
    data['start_time'] = data['start_time']*1000
    data['start_time'] = pd.to_datetime(data['start_time'], unit='ms')
    data['start_time'] = data['start_time'] - data['start_time'][0]
    data = data.iloc[1: , :]
    return data['start_time'].iloc[::2,]

def parseActivity(data, splits):
    data['timestamp'] = data['timestamp'] - data['timestamp'][0]
    data['timestamp'] = data['timestamp']*1000
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data['timestamp'] = data['timestamp'] - data['timestamp'][0]
    splitted_data = []
    for split in splits:
        # print(split)
        index = data.loc[data['timestamp'] == split].index[0]
        splitted_data.append(data.iloc[index:index+120,])
    return splitted_data

def combine(activity_data, accl_data):
    data = []
    for i in range(len(activity_data)):
        acc = accl_data[i]
        act = activity_data[i]
        acc['timestamp2'] = acc['timestamp'].dt.floor('s')
        acc = pd.merge(acc, act, left_on='timestamp2', right_on='timestamp', how='left')
        acc = acc.fillna(method='ffill')
        data.append(acc)
    return data

def main():
    warnings.filterwarnings("ignore")
    input_directory = pathlib.Path(sys.argv[1])
    
    accl = pd.read_csv(input_directory / 'accl.csv', index_col=False)
    splits = pd.read_csv(input_directory / 'splits.csv')
    activity = pd.read_csv(input_directory / 'activity.csv')

    splits = parseSplits(splits)
    accl = parseAccl(accl, splits)
    activity = parseActivity(activity, splits)
    combined = combine(activity, accl)
    filtered_combined = custom_filter(combined)
    print(filtered_combined)

    ## plot
    plt.plot(filtered_combined[3]['timestamp_x'],filtered_combined[3]['filtered_gFx'], label='butter filter')
    plt.plot(filtered_combined[3]['timestamp_x'],filtered_combined[3]['gFx'], label='x')
    # plt.plot(combined[3]['timestamp_x'],accl[3]['gFy'], label='y')
    # plt.plot(combined[3]['timestamp_x'],accl[3]['gFz'], label='z')
    # plt.plot(activity[3]['timestamp'],activity[3]['heart_rate'], label='heartrate')
    
    plt.title("Heart Rate versus Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Heart Rate")
    plt.legend()

    plt.show()

    # i = 0
    # for subdata in combined:
    #     name = 'out' + str(i) + '.csv'
    #     i += 1
    #     subdata.to_csv(name, index=False)

    
if __name__ == "__main__":
    main()
