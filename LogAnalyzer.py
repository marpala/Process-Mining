import numpy as np
import pandas as pd
from constants import *
from scipy import stats

class LogAnalyzer:

    def __init__(self, dataframe, dataset_name, verbose=True):
        self.df = dataframe
        self.dataset_name = dataset_name
        self.concluding_attr = {} # dict, {column name: {values}, ...}
        self.final_events = [] # list of all final eventID's
        self.start_times = {} # dict, {case name: Timestamp of first event, ...}
        self.case_times = {} # dict, {case name: total time of case, in sec, ...}
        self.verbose = verbose
        self.__pre_analyze()

    # Modifies the instance dataframe *in place* so that it only contains finished cases.
    # It also finds the eventID of every concluding last event in a case.
    def filter_concluded(self):
        keys = list(self.concluding_attr.keys())
        last_by_case = self.df.groupby(CASE_NAME).last() # last event of each case
        sliced = last_by_case[[EVENT_ID] + keys] # slice dataframe by keys
        
        for k in keys:
            sliced = sliced[sliced[k].isin(self.concluding_attr[k])]
        
        self.final_events = sliced[EVENT_ID].values
        self.df = self.df[self.df[CASE_NAME].isin(sliced.index)].copy()

    # Fixes weird column names
    def __pre_analyze(self):
        self.df.columns = [i.strip() for i in self.df.columns.tolist()]

    # Calculates the start times and total times of each case in the log and
    # modifies the dataframe *in place* to include the remaining time as well as the time since start for each event
    def analyze(self):
        se = self.df.groupby(CASE_NAME)[EVENT_TIME].agg(['first','last']) # start and end times
        cases = sorted(self.df[CASE_NAME].unique())
        time_diff = (se['last'].values - se['first'].values) / np.timedelta64(1, 's')
        self.start_times = dict(zip(cases, se['first'].values))
        self.case_times = dict(zip(cases, time_diff))
        
        times = self.df[[CASE_NAME, EVENT_TIME]].values
        event_times_since = [(x[1]-self.start_times.get(x[0])).total_seconds() for x in times]
        self.df[TIME_SINCE_START] = event_times_since
        times = self.df[[CASE_NAME, TIME_SINCE_START]].values
        event_times_until = [self.case_times.get(x[0]) - x[1] for x in times]
        self.df[TIME_UNTIL_FINISHED] = event_times_until
    
    def filterOutliers(self):
        self.df = self.df[self.df[TIME_UNTIL_FINISHED] < DAYS_FILTER_TIME * 24 * 3600]
        zScores = self.df[TIME_UNTIL_FINISHED].values
        zScores = stats.zscore(zScores)
        zScoreDF = self.df
        zScoreDF['ZScore'] = zScores
        outlierDataframe = zScoreDF[abs(zScoreDF['ZScore']) > ZSCORE_FILTER_LIMIT]
        outlierCases = outlierDataframe[CASE_NAME].unique()
        preLen = len(self.df)
        self.df = self.df[~self.df[CASE_NAME].isin(outlierCases)]
        if self.verbose:
            print('ZScore Limit = {}              '.format(ZSCORE_FILTER_LIMIT))
            print('{} - {} = {}'.format(preLen, len(self.df), preLen - len(self.df)))


class TrainAnalyzer(LogAnalyzer):

    def __init__(self, dataframe, dataset_name, verbose=True, runnerUpPercentage=0.30):
        super().__init__(dataframe, dataset_name, verbose)
        self.runnerUpPercentage = runnerUpPercentage
        self.analyze()
        super().filterOutliers()
    
    # Calulates the concluding events for the instance dataframe (self.df)
    def find_concluding_attr(self):
        if self.dataset_name in CONCLUDING_ATTR:
            print('Recognized dataset {}. Using pre-computed concluding attributes'.format(self.dataset_name))
            self.concluding_attr = CONCLUDING_ATTR[self.dataset_name]
            if self.verbose:
                print('Concluding attributes:', self.concluding_attr)
            return
        
        print('Unrecognized dataset {}. The concluding attributes will be estimated'.format(self.dataset_name))
        for e in [EVENT_NAME]:
            last_event_case = self.df.groupby(CASE_NAME)[e].last()
            l = int(len(last_event_case)*2/3) # ignore the last 33.4% of cases
            allconcluding_attr = last_event_case[:l].value_counts()
            if self.verbose:
                print('Last event frequency:\n', allconcluding_attr)
            maxDifference = allconcluding_attr.iloc[0] * self.runnerUpPercentage
            minValue = allconcluding_attr.iloc[0] - maxDifference
            allconcluding_attr = allconcluding_attr[lambda x : x >= minValue]
            self.concluding_attr[e] = set(allconcluding_attr.index)
            if self.verbose:
                print('The computed concluding attributes are:\n', self.concluding_attr)

    def analyze(self):
        self.find_concluding_attr()
        print('Analyzing training data...', end='\r')
        self.filter_concluded()
        super().analyze()


class TestAnalyzer(LogAnalyzer):

    def __init__(self, dataframe, dataset_name, concluding_attr, verbose=True):
        super().__init__(dataframe, dataset_name, verbose)
        self.concluding_attr = concluding_attr
        print("Analyzing test data...     ", end='\r')
        self.filter_concluded()
        self.analyze()
        super().filterOutliers()