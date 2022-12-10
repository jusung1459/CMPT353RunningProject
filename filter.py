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
from parse_run import parseAccl, parseSplits, parseActivity, combine

## arg data: pass in the returned cleaned data from combine
def custom_filter(data):
    for i in range(0,len(data)):
        b, a = signal.butter(3, 0.35, btype='lowpass', analog=False)
        low_passed_butter = signal.filtfilt(b, a, data[i]['gFx'])
        data[i]['filtered_gFx'] = low_passed_butter

        b, a = signal.butter(3, 0.25, btype='lowpass', analog=False)
        low_passed_butter = signal.filtfilt(b, a, data[i]['gFy'])
        data[i]['filtered_gFy'] = low_passed_butter

        b, a = signal.butter(3, 0.35, btype='lowpass', analog=False)
        low_passed_butter = signal.filtfilt(b, a, data[i]['gFz'])
        data[i]['filtered_gFz'] = low_passed_butter

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
    plt.plot(filtered_combined[3]['timestamp_x'],filtered_combined[3]['filtered_gFy'], label='butter filter')
    # plt.plot(combined[3]['timestamp_x'],high_butter, label='butter high')
    plt.plot(filtered_combined[3]['timestamp_x'],filtered_combined[3]['gFy'], label='y')
    # plt.plot(combined[3]['timestamp_x'],low_passed_cheb)
    # plt.plot(combined[3]['timestamp_x'],accl[3]['gFy'], label='y')
    # plt.plot(combined[3]['timestamp_x'],accl[3]['gFz'], label='z')
    # plt.plot(activity[3]['timestamp'],activity[3]['heart_rate'], label='heartrate')
    
    plt.title("Filtered Acceleration data versus Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Acceleration")
    plt.legend()

    plt.show()

    
if __name__ == "__main__":
    main()
