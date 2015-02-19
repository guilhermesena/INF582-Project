#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# ============================================================================
#       Filename:  pre_processor.py
#    Description:  Pre processor
#        Created:  2015-02-11 19:45 
#         Author:  Tiago Lobato Gimenes        (tlgimenes@gmail.com)
#     Co-Authors:  <-- put your name here -->
#      Copyright:  Tiago Lobato Gimenes
#       Lincense:  MIT
# ============================================================================

"""
Pre-process the data from TITANIC. Creates new data where it's missing
"""

###############################################################################

import numpy as np
import scipy.stats as stats 
import sys

###############################################################################
## Parsed file data type
types_parsed_train_file = [('PassengerId', np.float64),('Survived', np.float64),
        ('Pclass', np.float64),('Sex', np.float64),('Age', np.float64), 
        ('SibSp', np.float64), ('Parch', np.float64),('Fare', np.float64),
        ('Embarked', np.float64)];

types_parsed_test_file = [('PassengerId', np.float64),('Pclass', np.float64),
        ('Sex', np.float64),('Age', np.float64),('SibSp', np.float64), 
        ('Parch', np.float64),('Fare', np.float64),('Embarked', np.float64)];

###############################################################################
## String definitions
sex = {'male':0, 'female': 1}
pClass = {1: 0, 2: 1, 3: 2}
embarked = {'S': 0, 'C': 1, 'Q': 2, '': 3}

###############################################################################
## @arg0: Train data based on the data type defined in main
## @returns: New created ndarray with the data types specified above
def parse(train_data, type='train') :

    # Creates new array for the parsed data
    if type == 'train' :
        parsed_data = np.ndarray(shape=(np.shape(train_data)[0],), 
            dtype=types_parsed_train_file);
        # Parses Survived column
        parsed_data['Survived'] = train_data['Survived'];

    else:
        parsed_data = np.ndarray(shape=(np.shape(train_data)[0],), 
            dtype=types_parsed_test_file);

    # Parses each column
     
    # Parses PassengerId
    copy_replace_nan(parsed_data['PassengerId'], train_data['PassengerId']);
    # Parses Passenger class column
    parse_pClass_col(parsed_data['Pclass'], train_data['Pclass']);
    # Parses Sex column
    parse_sex_col(parsed_data['Sex'], train_data['Sex']);
    # Parses Age column
    copy_replace_nan(parsed_data['Age'], train_data['Age']);
    # Parses SibSp column
    copy_replace_nan(parsed_data['SibSp'], train_data['SibSp']);
    # Parses Parch column
    copy_replace_nan(parsed_data['Parch'], train_data['Parch']);
    # Parses Fare
    copy_replace_nan(parsed_data['Fare'], train_data['Fare']);
    # Parses embarked column
    parse_embarked_col(parsed_data['Embarked'], train_data['Embarked']);

    # Checks if nan left on table
    if is_nan(parsed_data, type) :
        print ("Error Fatal: NaNs left on parsed data :(");
        sys.exit(1);

    return parsed_data;

###############################################################################
## @arg0: column of the parsed data to be filled
## @arg1: column of the raw file to be parsed
def parse_sex_col(parsed_col, raw_col) :
    for i in range(np.shape(raw_col)[0]) :
        parsed_col[i] = sex[raw_col[i]];

###############################################################################
## @arg0: column of the parsed data to be filled
## @arg1: column of the raw file to be parsed
def parse_pClass_col(parsed_col, raw_col) :
    for i in range(np.shape(raw_col)[0]) :
        parsed_col[i] = pClass[raw_col[i]];

###############################################################################
## @arg0: column of the parsed data to be filled
## @arg1: column of the raw file to be parsed
def parse_embarked_col(parsed_col, raw_col) :
    for i in range(np.shape(raw_col)[0]) :
        parsed_col[i] = embarked[raw_col[i]];

###############################################################################
## Substitutes the NaN for the mean of the args
## @arg0: column of the parsed data to be filled
## @arg1: raw file to be parsed
def copy_replace_nan(parsed_col, train_col) :
    # Sets parsed_col with raw values
    for i in range(np.shape(train_col)[0]) :
        parsed_col[i] = train_col[i];

    # Takes the mean of the values 
    mean = stats.nanmean(train_col, axis=0);

    # Find indicies that you need to replace (NaN)
    inds_nan = np.where(np.isnan(train_col));

    # Replaces NaN with mean
    parsed_col[inds_nan] = mean;


###############################################################################
## Checks if there is any NaN left on data
def is_nan(parsed_data, type) :
    if type == 'train' :
        dtype = types_parsed_train_file;
    else :
        dtype = types_parsed_test_file;

    # Foreach column checks if there is a NaN
    for i in range(np.shape(dtype)[0]) :
        if np.shape(np.where(np.isnan(parsed_data[dtype[i][0]]))[0])[0] > 0 :
            print(parsed_data[np.where(np.isnan(parsed_data[dtype[i][0]]))[0]]);
            return True;

    return False;

###############################################################################
