import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from helper_functions import ProgressPrinter
from constants import *


# fit on columns we want in format['column_name1','column_name2','column_name3']
# can be from constants
# There is a way with importance, maybe for next time
def random_forest_regression(anlz_train, anlz_test, columns, verbose=False):
    # preprocess data
    print('Making a random forest regression model from the training data...', end='\r')
    df_train = anlz_train.df
    df_test = anlz_test.df
    time_ids = anlz_train.final_events
    time_ids.sort()

    #Devide the Test set into 10 parts:
    #df_test should be chronological
    df_list = np.array_split(df_test, 10)
    predictionlist = []
    for i in df_list:
        date = i[EVENT_TIME].iloc[0]
        df_train_past = df_train[df_train[EVENT_TIME] < date]
        for j in range(len(time_ids)):
            if time_ids[j] not in df_train_past[EVENT_ID].values:
                if j == 0:
                    df_train_past = np.array([])
                    break
                final = time_ids[j-1]
                indx = pd.Index(df_train_past[EVENT_ID]).get_loc(final)
                df_train_past = df_train_past.loc[0:indx, :]
                break

        if len(df_train_past) == 0:
            predictionlist.extend([np.NAN]*len(i))
            continue
        df_train_time = df_train_past[TIME_UNTIL_FINISHED]
        df_train_past = df_train_past[columns]
        df_test_full = i[columns]

        df_test_full = pd.get_dummies(df_test_full)
        df_train_past = pd.get_dummies(df_train_past)

        missing_cols = set(df_train_past.columns) - set(df_test_full.columns)
        # Add a missing column in test set with default value equal to 0
        for c in missing_cols:
            df_test_full[c] = 0
        # same for train dataset
        missing_cols = set(df_test_full.columns) - set(df_train_past.columns)
        for c in missing_cols:
            df_train_past[c] = 0

        rfr = RandomForestRegressor(n_estimators=100, max_depth=5)
        rfr.fit(df_train_past, df_train_time)

        predictions = rfr.predict(df_test_full)
        predictionlist.extend(predictions)

    for i in range(len(predictionlist)):
        if predictionlist[i] == np.NAN:
            predictionlist[i] = df_test[AVG_PRED].loc[i]

    # add prediction to df_test
    df_test[FOREST_PRED] = predictionlist


