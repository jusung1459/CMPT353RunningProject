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
from scipy.integrate import simpson

## arg data: pass in the returned cleaned data from combine
def custom_filter(data):
    for i in range(0,len(data)):
        b, a = signal.butter(3, 0.35, btype='lowpass', analog=False)
        low_passed_butter = signal.filtfilt(b, a, data[i]['gFx'])
        data[i]['filtered_gFx'] = low_passed_butter
        data[i]['filtered_gFx'] = data[i]['filtered_gFx'].abs()

        b, a = signal.butter(3, 0.25, btype='lowpass', analog=False)
        low_passed_butter = signal.filtfilt(b, a, data[i]['gFy'])
        data[i]['filtered_gFy'] = low_passed_butter
        data[i]['filtered_gFy'] = data[i]['filtered_gFy'].abs()

        b, a = signal.butter(3, 0.35, btype='lowpass', analog=False)
        low_passed_butter = signal.filtfilt(b, a, data[i]['gFz'])
        data[i]['filtered_gFz'] = low_passed_butter
        data[i]['filtered_gFz'] = data[i]['filtered_gFz'].abs()

        data[i] = data[i].drop(['timestamp2', 'timestamp_y', 'cadence', 'fractional_cadence'],axis=1)

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

def split_waves(data):
    wave = []
    first_dip = 0
    dip_flag = False
    # print(data)
    for i in range(1, len(data)-1):
        if ((data['filtered_gFy'][i-1] > data['filtered_gFy'][i]) and (data['filtered_gFy'][i] < data['filtered_gFy'][i+1])):
            first_dip = i
            break
    # first_dip = 1280
    for i in range(first_dip+1, len(data)-1):
    # for i in range(1280, 1380):
        if ((data['filtered_gFy'][i-1] > data['filtered_gFy'][i]) and (data['filtered_gFy'][i] < data['filtered_gFy'][i+1])):
            if (not dip_flag):
                dip_flag = True
            else:
                if ((i-first_dip+1) > 6):
                    wave.append(data.loc[first_dip:i])
                    # plt.plot(range(i-first_dip+1),data.loc[first_dip:i]['filtered_gFy'])
                first_dip = i
                dip_flag = False
    # data.loc[1350:1400]['filtered_gFx'].to_csv("x_subsetData", index=False)
    # plt.show()
    # distance = avg velocity * time
    # avg velocity = area under curve of acceleration
    distance = []
    for i in range(0, len(wave)):
        area = simpson(wave[i]['filtered_gFy'], dx=5)
        cur_distance = round(area * len(wave[i]))
        distance.append(cur_distance)
    # print(distance)
    # plt.hist(distance, bins=25)
    # plt.show()
    return distance
            

def main():
    warnings.filterwarnings("ignore")
    input_directory_tread = pathlib.Path(sys.argv[1])
    input_directory_out = pathlib.Path(sys.argv[1])
    
    ## treadmill data
    accl = pd.read_csv(input_directory_tread / 'accl.csv', index_col=False)
    splits = pd.read_csv(input_directory_tread / 'splits.csv')
    activity = pd.read_csv(input_directory_tread / 'activity.csv')

    splits = parseSplits(splits)
    accl = parseAccl(accl, splits)
    activity = parseActivity(activity, splits)
    combined = combine(activity, accl)
    filtered_combined = custom_filter(combined)
    # print(filtered_combined)
    distance = []
    for sub_data in filtered_combined:
        distance += split_waves(sub_data)

    plt.hist(distance, bins=50)
    # plt.show()

    # save file
    pd.DataFrame({"distance":distance}).to_csv("treadmill_distance.csv", index=False)

    ## outside data
    accl = pd.read_csv(input_directory_out / 'accl.csv', index_col=False)
    splits = pd.read_csv(input_directory_out / 'splits.csv')
    activity = pd.read_csv(input_directory_out / 'activity.csv')

    splits = parseSplits(splits)
    accl = parseAccl(accl, splits)
    activity = parseActivity(activity, splits)
    combined = combine(activity, accl)
    filtered_combined = custom_filter(combined)
    # print(filtered_combined)
    distance = []
    for sub_data in filtered_combined:
        distance += split_waves(sub_data)

    plt.hist(distance, bins=50)
    # plt.show()

    # save file
    pd.DataFrame({"distance":distance}).to_csv("outside_distance.csv", index=False)

    
if __name__ == "__main__":
    main()
