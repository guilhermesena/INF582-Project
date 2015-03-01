#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# ============================================================================
#       Filename:  main.py
#    Description:  Main
#        Created:  2015-02-11 19:45 
#         Author:  Tiago Lobato Gimenes        (tlgimenes@gmail.com)
#     Co-Authors:  <-- put your name here -->
#      Copyright:  Tiago Lobato Gimenes
#       Lincense:  MIT
# ============================================================================

"""
Main file !
"""

###############################################################################

import sys
import os
import numpy as np

from pre_processor import parse
from ada_boost import ada_boost
from perceptron import perceptron

###############################################################################
## C-style defines
TRAIN_PATH_DATA_DEFAULT = 'Data/train.csv'
TEST_PATH_DATA_DEFAULT = 'Data/test.csv'

OUTPUT_FILE_NAME = 'survived.csv'

###############################################################################
## Raw file data type
types_train_raw_file = [('PassengerId', np.int64),('Survived', np.int64),('Pclass', 
        np.int64),('SurName', np.str_, 99), ('Name', np.str_, 99), ('Sex', 
        np.str_, 99),('Age', np.float64), ('SibSp', np.int64), ('Parch', 
        np.int64),('Ticket', np.str_, 99), ('Fare', np.float64),('Cabin', 
        np.str_, 99),('Embarked', np.str_, 99)];

types_test_raw_file = [('PassengerId', np.int64),('Pclass', np.int64),
        ('SurName', np.str_, 99), ('Name', np.str_, 99), ('Sex', np.str_, 99),
        ('Age', np.float64), ('SibSp', np.int64), ('Parch', np.int64),
        ('Ticket', np.str_, 99), ('Fare', np.float64),('Cabin', np.str_, 99),
        ('Embarked', np.str_, 99)];

###############################################################################
## Main function !
def main():
    # Parses command line
    train_path, test_path = cli_parser();

    # Loads train data
    train_data = np.genfromtxt(train_path, dtype=types_train_raw_file, 
                               delimiter=',', skip_header=1);
    # Loads test data 
    test_data = np.genfromtxt(test_path, dtype=types_test_raw_file, 
                              delimiter=',', skip_header=1);

    # Calls the pre_processor
    parsed_train_data = parse(train_data, type='train');
    parsed_test_data = parse(test_data, type='test');

    # Calls the learning algorithm
    #survived = ada_boost(parsed_train_data, parsed_test_data);
    
    survived = perceptron(parsed_train_data, parsed_test_data);

    # Formats the output data
    out_data = np.array(zip(parsed_test_data['PassengerId'], survived), 
                dtype=[('PassengerId', np.int64), ('Survived', np.int64)]);

    # Saves data in file for sanity check 
    np.savetxt(OUTPUT_FILE_NAME, out_data, delimiter=',', fmt=['%d', '%d'],
              header='PassengerId,Survived', comments='');

###############################################################################
## A simple parser for the command line interface (CLI)
def cli_parser():
    train_path = TRAIN_PATH_DATA_DEFAULT;
    test_path = TEST_PATH_DATA_DEFAULT;

    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-s' and i < len(sys.argv)-1:
            train_path = sys.argv[i+1];
        elif sys.argv[i] == '-d' and i < len(sys.argv)-1:
            test_path = sys.argv[i+1];
        else:
            print_help();
            sys.exit(1);

    return (train_path, test_path);

###############################################################################
## Print help and usage
def print_help():
    print('Usage: python2.7 main.py [options] <path>')
    print('Options:')
    print('  -s     Sets the trainning data (sample) path')
    print('  -d     Sets the test data (data) path')
    print('  -h     Displays this menu')
    print('')
    print('Thanks !')

###############################################################################
## Runs main function
main();
