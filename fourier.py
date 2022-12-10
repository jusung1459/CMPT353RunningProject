import os
import pathlib
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timezone
import matplotlib.pyplot as plt
import warnings
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from parse_run import parseAccl, parseSplits, parseActivity, combine, custom_filter

def main():
    warnings.filterwarnings("ignore")
    input_directory_outside = pathlib.Path(sys.argv[1])
    input_directory_treadmill = pathlib.Path(sys.argv[2])
    
    accl_outside = pd.read_csv(input_directory_outside / 'accl.csv', index_col=False)
    splits_outside = pd.read_csv(input_directory_outside / 'splits.csv')
    activity_outside = pd.read_csv(input_directory_outside / 'activity.csv')

    splits_outside = parseSplits(splits_outside)
    accl_outside = parseAccl(accl_outside, splits_outside)
    activity_outside = parseActivity(activity_outside, splits_outside)
    combined_outside = combine(activity_outside, accl_outside)
    filtered_combined_outside = custom_filter(combined_outside)

    accl_treadmill = pd.read_csv(input_directory_treadmill / 'accl.csv', index_col=False)
    splits_treadmill = pd.read_csv(input_directory_treadmill / 'splits.csv')
    activity_treadmill = pd.read_csv(input_directory_treadmill / 'activity.csv')

    splits_treadmill = parseSplits(splits_treadmill)
    accl_treadmill = parseAccl(accl_treadmill, splits_treadmill)
    activity_treadmill = parseActivity(activity_treadmill, splits_treadmill)
    combined_treadmill = combine(activity_treadmill, accl_treadmill)
    filtered_combined_treadmill = custom_filter(combined_treadmill)

    ## plot
    filtered_values_gFx_outside = filtered_combined_outside[3]['filtered_gFx'].values
    filtered_values_gFy_outside = filtered_combined_outside[3]['filtered_gFy'].values
    filtered_values_gFz_outside = filtered_combined_outside[3]['filtered_gFz'].values

    filtered_values_gFx_treadmill = filtered_combined_treadmill[3]['filtered_gFx'].values
    filtered_values_gFy_treadmill = filtered_combined_treadmill[3]['filtered_gFy'].values
    filtered_values_gFz_treadmill = filtered_combined_treadmill[3]['filtered_gFz'].values

    N = filtered_values_gFx_outside.size
    yf_x_outside = fft(signal.detrend(filtered_values_gFx_outside))
    yf_y_outside = fft(signal.detrend(filtered_values_gFy_outside))
    yf_z_outside = fft(signal.detrend(filtered_values_gFz_outside))
    xf_outside = fftfreq(N, d=1/N)[:N//2]

    yf_x_treadmill = fft(signal.detrend(filtered_values_gFx_treadmill))
    yf_y_treadmill = fft(signal.detrend(filtered_values_gFy_treadmill))
    yf_z_treadmill = fft(signal.detrend(filtered_values_gFz_treadmill))
    xf_treadmill = fftfreq(N, d=1/N)[:N//2]

    print(stats.mannwhitneyu(yf_x_outside, yf_x_treadmill).pvalue)
    print(stats.mannwhitneyu(yf_y_outside, yf_y_treadmill).pvalue)
    print(stats.mannwhitneyu(yf_z_outside, yf_z_treadmill).pvalue)

    plt.title("Fourier Transform of Butter Filtered Results (Outside)")
    plt.plot(xf_outside, np.abs(yf_x_outside[:N//2]), label='Filtered gFx')
    plt.plot(xf_outside, np.abs(yf_y_outside[:N//2]), label='Filtered gFy')
    plt.plot(xf_outside, np.abs(yf_z_outside[:N//2]), label='Filtered gFz')
    # plt.title("Fourier Transform of Butter Filtered Results (Treadmill)")
    # plt.plot(xf_treadmill, np.abs(yf_x_treadmill[:N//2]), label='Filtered gFx')
    # plt.plot(xf_treadmill, np.abs(yf_y_treadmill[:N//2]), label='Filtered gFy')
    # plt.plot(xf_treadmill, np.abs(yf_z_treadmill[:N//2]), label='Filtered gFz')
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

    
if __name__ == "__main__":
    main()
