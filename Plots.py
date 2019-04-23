import pandas as pd
import matplotlib.pyplot as plt
import sys
from constants import *

def Mseplotting(out_df, predictors):
    # out_df is output csv, predictors are our predictors we want to plot
    #plots mse of a starting day in days^2

    a_df = out_df.copy()
    a_df.dropna(inplace = True)
    a_df.reset_index(drop = True, inplace = True)
    a_df.sort_values(TIME_SINCE_START, inplace = True)
    a_df[DAYS_SINCE_START] = a_df[TIME_SINCE_START]//86400
    a_df[TIME_UNTIL_FINISHED] = a_df[TIME_UNTIL_FINISHED] // 86400
    list_series = []
    for j in predictors:
        a_df[j] = a_df[j] // 86400

        df_s = a_df[[TIME_UNTIL_FINISHED, DAYS_SINCE_START, j]]
        thesum = {}
        for i in df_s.itertuples():
            actual = getattr(i, '_1')
            days = int(getattr(i, '_2'))
            predicted = getattr(i, '_3')
            if days in thesum:
                thesum[days][0] += (actual - predicted) ** 2
                thesum[days][1] += 1
            else:
                thesum[days] = [(actual - predicted) ** 2, 1]
        mse_dict = {}
        for key in thesum:
            mse_dict[key] = thesum[key][0] / thesum[key][1]
        dfmse = pd.Series(mse_dict)
        list_series.append(dfmse)

    fig, ax = plt.subplots(figsize=(15, 10))
    for i in list_series:
        i.plot(axes=ax)
    ax.set_xlabel('Days already past since start')
    ax.set_ylabel('MSE')
    ax.legend(predictors, title='legend')
    plt.show()


#a_df is an output csv, predictor is the predictor we want to plot (only one as it does not make sens to put multiple predictors in one dotplot)
#call this funciton multiple times for more predictors
def DotPlot(out_df, predictor):
    # xPast time passed util event fires
    # yRemain time remaing until case terminates
    # yEst time remaining unitl estimated case end

    a_df = out_df.copy()
    #stolen from roel
    a_df.dropna(inplace = True)
    a_df.reset_index(drop = True, inplace = True)
    a_df.sort_values(TIME_SINCE_START, inplace = True)
    a_df[DAYS_SINCE_START] = a_df[TIME_SINCE_START]//86400
    a_df[TIME_UNTIL_FINISHED] = a_df[TIME_UNTIL_FINISHED] // 86400

    # Create data
    a_df[predictor] = a_df[predictor] / 86400
    df_s = a_df[[TIME_UNTIL_FINISHED, DAYS_SINCE_START, predictor]]

    xPast = []
    yRemain = []
    yEst = []

    for i in df_s.itertuples():
        actual = getattr(i, '_1')
        days = int(getattr(i, '_2'))
        predicted = getattr(i, '_3')

        xPast.append(days)
        yRemain.append(actual)
        yEst.append(predicted)

    gRemain = (xPast, yRemain)
    gEst = (xPast, yEst)

    data = (gRemain, gEst)
    colors = ("black", "red")

    #Create plot
    fig, ax = plt.subplots(figsize=(15, 10))

    for data, color in zip(data, colors):
        x, y = data
        ax.scatter(x, y, c=color, edgecolors='none')

    plt.show()

