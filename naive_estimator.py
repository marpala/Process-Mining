import numpy as np
import pandas as pd
from constants import *
from helper_functions import ProgressPrinter

# Average estimator based on the algorithm from the paper
# "Cycle Time Prediction: When Will This Case Finally Be Finished?"
# by B.F. van Dongen, R.A. Crooy, W.M.P. van der Aalst


# Writes estimation column in place in the df instance of anlz_test
def write_estimation(anlz_train, anlz_test, verbose=False):

    df_train = anlz_train.df
    df_train = df_train[df_train['eventID'].isin(anlz_train.final_events)]
    df_test = anlz_test.df
    total_times = anlz_train.case_times
    running_avg = {'avg': 0, 'len': 0}

    def update_avg(case):
        time = total_times[case]
        running_avg['avg'] = (
            running_avg['avg'] * running_avg['len'] + time) / (running_avg['len'] + 1)
        running_avg['len'] = running_avg['len'] + 1

    j = 0
    l = len(df_train)
    progress = ProgressPrinter(len(df_test), "Computing naive estimation. This may take a while...")
    for i in df_test.index:
        while j != l and df_test.loc[i][EVENT_TIME] >= df_train.iloc[j][EVENT_TIME]:
            update_avg(df_train.iloc[j][CASE_NAME])
            j += 1
        df_test.at[i, AVG_PRED] = max(
            0, running_avg['avg'] - df_test.loc[i][TIME_SINCE_START])
        if verbose:
            print('Running average for eventID {} (index {}): {}'.format(
                df_test.loc[i]['eventID'], i, running_avg['avg']))
        progress.updateProgress()
