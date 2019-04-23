import numpy as np
import pandas as pd
from constants import *
from helper_functions import ProgressPrinter

class HistoryTrainer:

    def __init__(self, trainingDF, progress, final_ids):
        self.df = trainingDF[[EVENT_ID, CASE_NAME, EVENT_NAME, EVENT_TIME, TIME_UNTIL_FINISHED]]
        self.index = 0
        self.progress = progress
        self.historyAvg = {}
        self.historyCount = {}
        self.histories = {}
        self.final_ids = self.df[self.df[EVENT_ID].isin(final_ids)]
        self.final_ids_list = final_ids
        self.maxIndex = len(self.final_ids)

    def updateDictionaries(self, hist, rt):
        if hist in self.historyCount:
            self.historyAvg[hist] = self.historyAvg[hist] * self.historyCount[hist] + rt
            self.historyCount[hist] += 1
            self.historyAvg[hist] = self.historyAvg[hist] / self.historyCount[hist]
        else:
            self.historyAvg[hist] = rt
            self.historyCount[hist] = 1

    def trainUntilTime(self, timestamp):
        #Immediate check to make sure we're not out of bounds.
        if self.index >= self.maxIndex:
            return 

        #There are still cases we haven't trained on.
        allowedCases = []
        while self.final_ids.iloc[self.index][EVENT_TIME] <= timestamp:
            allowedCases.append(self.final_ids.iloc[self.index][CASE_NAME])
            self.index += 1
            self.progress.updateProgress()
            if self.index >= self.maxIndex:
                break

        #Create a new dataframe consisting only of "legal" events 
        trainDF = self.df[self.df[CASE_NAME].isin(allowedCases)]

        #Go through each legal event and train.
        for i in trainDF.index :
            rt = trainDF.loc[i][TIME_UNTIL_FINISHED]
            case = trainDF.loc[i][CASE_NAME]
            e_id = trainDF.loc[i][EVENT_ID]
            e_name = trainDF.loc[i][EVENT_NAME]

            #Add this event to the correct history
            if case in self.histories:
                self.histories[case] += e_name + ','
            else:
                self.histories[case] = e_name + ','
            hist = self.histories[case]

            #If we just had the last event in this case, the case is no longer relevant
            if e_id in self.final_ids_list:
                self.histories.pop(case)

            self.updateDictionaries(hist, rt)
            
            self.progress.updateProgress()

    def getPrediction(self, hist, naive):
        if hist in self.historyAvg:
            return self.historyAvg[hist]
        else:
            return naive
    
def write_estimation(trainAnalyzer, testAnalyzer, verbose = False):

    trainingDF = trainAnalyzer.df
    testDF = testAnalyzer.df
    train_final_ids = trainAnalyzer.final_events
    test_final_ids = testAnalyzer.final_events
    stolen = 0

    filter = trainingDF[trainingDF[EVENT_TIME] <= testDF.iloc[-1][EVENT_TIME]]
    progress = ProgressPrinter(len(testDF) + len(filter) + len(train_final_ids), "Computing history estimation...          ")
    ht = HistoryTrainer(filter, progress, train_final_ids)

    histories = {}

    for i in testDF.index:

        case = testDF.loc[i][CASE_NAME]
        ts = testDF.loc[i][EVENT_TIME]
        e_id = testDF.loc[i][EVENT_ID]
        e_name = testDF.loc[i][EVENT_NAME]
        if case in histories:
            histories[case] += e_name + ','
        else:
            histories[case] = e_name + ','
        hist = histories[case]

        ht.trainUntilTime(ts)
        naive = testDF.loc[i][AVG_PRED]
        prediction = ht.getPrediction(hist, naive)
        testDF.at[i, HIST_PRED] = prediction
        testDF.at[i, 'History stole from naive'] = (prediction == naive)
        if prediction == naive:
            stolen += 1

        #If we just had the last event in this case, the case is no longer relevant
        if e_id in test_final_ids:
            histories.pop(case)
        progress.updateProgress()
    if verbose:
        print('{} values used from Naive Estimator, {} exact history matches found.                 '.format(stolen, len(testDF) - stolen))
