import sys
import argparse
import numpy as np
import pandas as pd
from constants import *


class ProgressPrinter:
    """
    Initialize it with the len(df_test) of the test dataset.\
    Call updateProgress() method after an event has been predicted.
    """
    def __init__(self, totalEvents, msg='computing... '):
        self.total = totalEvents
        self.percent = 0.0
        self.stepSize = 100 / totalEvents
        self.msg = msg
        print('{}% {}'.format(round(self.percent, 1), self.msg), end='\r')

    def printProgress(self):
        print('{}% {}'.format(round(self.percent, 1), self.msg), end='\r')

    def updateProgress(self):
        try:
            self.percent = self.percent + self.stepSize
            self.printProgress()
        except KeyboardInterrupt:
            exit()
            
# Samples size amount of cases from the dataset
# if size <= 0, returns the entire dataset unchanged
# if size > 0, returns toParse amount of cases from the dataset
def sample_cases(df_test, sample, verbose=True):
    if sample <= 0 or not sample:
        return df_test

    case_names = df_test[CASE_NAME].unique()
    if isinstance(sample,bool):
        sampled_cases = prompt_for_sample()
    else:
        sampled_cases = sorted(np.random.choice(case_names, size=sample, replace=False))
    if verbose:
        print('The following cases have been sampled:')
        print(sampled_cases)
    return df_test[df_test[CASE_NAME].isin(sampled_cases)]

# Command line argument parser
def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sample', metavar='int', nargs='?', const=True, type=int,
                        default=0, help='positive integer indicating the number of cases to sample; default 2')
    parser.add_argument('-v', '--verbose',
                        help='increase output verbosity', action='store_true')
    parser.add_argument('-m', '--manual',
                        help='prompts the user to manually enter the final events', action='store_true')
    parser.add_argument("-p", '--plot',
                        help="plot the MSE of the predictors in a graph", action='store_true')
    parser.add_argument("train_data", type=str,
                        help="path to the training data file (must end with *-training.csv)")
    parser.add_argument("test_data", type=str,
                        help="path to the test data file (must end with *-test.csv)")
    parser.add_argument("output_data", type=str,
                        help="path where the output file will be stored (must end with *.csv)")

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    print(welcome_msg)

    if args.verbose:
        print('Verbose output turned on')
        verbose = True
    else:
        verbose = False

    if args.manual:
        manual = True
    else:
        manual = False

    if args.plot:
        plot = True
    else:
        plot = False

    sample = args.sample
    if isinstance(sample, bool):
        print('Sampling turned on')
    elif sample > 0:
        print('Sampling turned on. {} cases will be sampled'.format(sample))

    if args.train_data.endswith('-training.csv'):
        training_file = args.train_data
    else:
        print('Path to training file must end with *-training.csv')
        parser.exit()

    if args.test_data.endswith('-test.csv'):
        test_file = args.test_data
    else:
        print('Path to test file must end with *-test.csv')
        parser.exit()

    if args.output_data.endswith('.csv'):
        output_file = args.output_data
    else:
        print('Path to output file must end with *.csv')
        parser.exit()

    return verbose, manual, sample, plot, training_file, test_file, output_file

# Prompts the user to manually input the final attributes of a log
def prompt_for_final_attr():
    while True:
        print('You have chosen to manually input the concluding attributes.')
        columns = input('First, enter the attribute columns where the attributes appear (comma-separated), e.g.\n\
event concept:name, event lifecycle:transition\n-> ')
        columns = [c.strip() for c in columns.split(',')]
        concluding_attr = dict.fromkeys(columns)
        for c in columns:
            attr = input('Now enter the concluding attributes that appear in column: ' +
                           str(c) + ' (comma-separated)\n-> ')
            concluding_attr[c] = {e.strip() for e in attr.split(',')}
        print('Is this correct?', concluding_attr, '\ny / n')
        ans = input('-> ')
        if ans == 'y':
            break
        else:
            continue
    return concluding_attr

def prompt_for_sample():
    while True:
        print('You have chosen to manually input the cases to sample from the test data.')
        cases = input('Enter the names of the cases (comma-separated), e.g.\n\
206324, 206327\n-> ')
        cases = [c.strip() if not c.strip().isdigit() else int(c.strip()) for c in cases.split(',')]
        print('Is this correct?', cases, '\ny / n')
        ans = input('-> ')
        if ans == 'y':
            break
        else:
            continue
    return cases