import time
import numpy as np
import pandas as pd
import naive_estimator
import HistoryEstimator
import Random_Forest
from helper_functions import *
from LogAnalyzer import *
from constants import *
from Plots import *



time_start = time.time()

# Parses the command line arguments
verbose, manual, sample, plot, training_file, test_file, output_file = parseargs()
dataset_name = training_file.replace('\\','/').split('/')[-1].split('-')[0].replace(" ", "_")

# Prompts the user to input the final attributes manually
if manual:
    concl_attr = prompt_for_final_attr()
    CONCLUDING_ATTR[dataset_name] = concl_attr

print("Parsing CSV files...", end='\r')
time_parse = time.time()
date_parser = lambda x: pd.datetime.strptime(x, DATE_FORMAT)
df_train = pd.read_csv(training_file, parse_dates=[EVENT_TIME], date_parser=date_parser, encoding="ISO-8859-1")
df_test = pd.read_csv(test_file, parse_dates=[EVENT_TIME], date_parser=date_parser, encoding="ISO-8859-1")
end_parse = time.time() - time_parse
print('Parsing CSV files \u2713 ({} seconds)'.format(round(end_parse,2)))

# Analyze training set
time_train = time.time()
anlz_train = TrainAnalyzer(df_train, dataset_name, verbose)
concluding_events = anlz_train.concluding_attr
end_train = time.time() - time_train
print('Analyzing training data \u2713 ({} seconds)'.format(round(end_train,2)))

if sample:
    print("Sampling test data...", end='\r')
    df_test = sample_cases(df_test, sample, verbose)
    print("Sampling test data \u2713     ", end='\r')

# Analyze training set
time_test = time.time()
anlz_test = TestAnalyzer(df_test, dataset_name, concluding_events, verbose)
end_test = time.time() - time_test
print('Analyzing test data \u2713 ({} seconds)'.format(round(end_test,2)))

# Start computing naive estimation
time_naive = time.time()
naive_estimator.write_estimation(anlz_train, anlz_test)
end_naive = time.time() - time_naive
print('Computing naive estimation \u2713 ({} seconds)               \
    '.format(round(end_naive,2)))

# Start computing history
time_hist = time.time()
HistoryEstimator.write_estimation(anlz_train, anlz_test, verbose)
end_hist = time.time() - time_hist
print('Computing history estimation \u2713 ({} seconds)               \
    '.format(round(end_hist,2)))

# Start computing Random Forest
time_forest = time.time()
Random_Forest.random_forest_regression(anlz_train, anlz_test, [EVENT_ID, CASE_PD, TIME_SINCE_START], verbose)
end_forest = time.time()-time_forest
print('Computing forest estimation \u2713 ({} seconds)               \
    '.format(round(end_forest,2)))

# Export the dataframe to a CSV file
df_test = anlz_test.df
df_test.to_csv(output_file, index=False, encoding="ISO-8859-1")
time_end = time.time() - time_start

if plot:
    print("Plotting MSE of predictors in a graph...", end='\r')
    Mseplotting(df_test, [AVG_PRED, HIST_PRED, FOREST_PRED]) #To plot additional estimators, add them to the list. --> [AVG_PRED, HIST_PRED]
    DotPlot(df_test, AVG_PRED)
    DotPlot(df_test, HIST_PRED)
    DotPlot(df_test, FOREST_PRED)

print("Done! Output file is in {}. Total time: {} seconds".format(output_file, round(time_end,2)))