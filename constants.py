# Constant values. These should never be changed through code
# Columns
EVENT_TIME = 'event time:timestamp'
EVENT_NAME = 'event concept:name'
EVENT_ID = 'eventID'
CASE_NAME = 'case concept:name'
CASE_PD = 'case Purchasing Document'
TIME_UNTIL_FINISHED = 'actual remaining time (sec)' # Time until the case is finished
TIME_SINCE_START = 'time since case start (sec)' # Time passed from the start of the case until the current event
AVG_PRED = 'naive prediction (sec)'
HIST_PRED = 'history-based prediction (sec)'
FOREST_PRED = 'Random Forest prediction (sec)'

DATE_FORMAT = '%d-%m-%Y %H:%M:%S.%f'
DAYS_SINCE_START = 'Days since start'
DAYS_FILTER_TIME = 1500
ZSCORE_FILTER_LIMIT = 1.5
# Hard-coded or user-input concluding attributes for known datasets
CONCLUDING_ATTR = {
   'BPI_Challenge_2012': {
      'event concept:name': {'A_DECLINED', 'A_CANCELLED', 'W_Valideren aanvraag'}, 
      'event lifecycle:transition': {'COMPLETE'}
   },
   'Artificial_Digital_Photo_Copier_Event_Log': {
      'event concept:name': {'Job'},
      'event lifecycle:transition' : {'complete'}
   },
   'BPI_Challenge_2019': {
      'event concept:name': {'Clear Invoice'}
   }
   ,
   'BPI_Demo_2019': {
      'event concept:name': {'Clear Invoice'}
   }
}

welcome_msg = """
         ____  _       _  ___           
        |___ \(_) ___ (_)/ _ \          
          __) | |/ _ \| | | | |         
   ____  / __/| | (_) | | |_| | _ ____  
  / ___||_____|_|\___/|_|\___/ / |___ \ 
 | |  _| '__/ _ \| | | | '_ \  | | __) |
 | |_| | | | (_) | |_| | |_) | | |/ __/ 
  \____|_|  \___/ \__,_| .__/  |_|_____|
                       |_|                      
"""