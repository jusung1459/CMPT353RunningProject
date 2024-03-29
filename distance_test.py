import os
import pathlib
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import math
import matplotlib.pyplot as plt
from scipy import stats

def main():
    outside = pd.read_csv('outside_distance.csv')
    treadmill = pd.read_csv('treadmill_distance.csv')
    # print(outside)
    # print(treadmill)

    outside['distance'] = (outside['distance']/100).round()
    outside = outside.loc[outside['distance'] >= 40]
    treadmill['distance'] = (treadmill['distance']/100).round()
    outside['distance'] = outside['distance'] ** 2
    treadmill['distance'] = treadmill['distance'] ** 2

    print("treadmill data points: " + str(len(treadmill['distance'])))
    print("treadmill data points: " + str(len(outside['distance'])))

    p_value = stats.ttest_ind(outside['distance'], treadmill['distance'])
    print(p_value)
    print(stats.levene(treadmill['distance'], outside['distance']))
    
    plt.hist(treadmill['distance'], bins=100, label='Treadmill')
    plt.hist(outside['distance'], bins=100, label='Outside')
    plt.title("Histogram of Jump Height")
    plt.xlabel("Jump height (mm*g)")
    plt.ylabel("Occurences")
    plt.legend()
    # plt.show()

    print(outside['distance'].mean())
    print(treadmill['distance'].mean())

main()
