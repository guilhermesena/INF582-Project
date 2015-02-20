#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# ============================================================================
#       Filename:  pre_processor.py
#    Description:  Pre processor
#        Created:  2015-02-18 19:45 
#         Author:  Tiago Lobato Gimenes        (tlgimenes@gmail.com)
#     Co-Authors:  <-- put your name here -->
#      Copyright:  Tiago Lobato Gimenes
#       Lincense:  MIT
# ============================================================================

"""
Learning part using the adaBoost algorithm
"""

###############################################################################

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

###############################################################################
## Choose here the columns of the training data that will be used by Ada Boost
types_clf = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 
             'Fare', 'Embarked'];


###############################################################################
## Does the adaBoost and returns an array containning died or not 
## @arg0: Train data
## @arg1: Test data
## @returns: Array with 1 if survived 0 if not
def ada_boost(train_data, test_data) :
    # Create and fit an AdaBoosted decision tree
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8), n_estimators=100);

    # Select columns of train_data 
    survived_training = train_data['Survived'];
    learning_data = np.matrix(train_data[types_clf]).T;

    # Fits model
    bdt.fit(learning_data, survived_training);

    # Predicts
    survived_test = bdt.predict(np.matrix(test_data[types_clf]).T);

    # Converts -1,1 space to 0 1 space
    to_01_space(survived_test);

    return survived_test;

###############################################################################
## Converts -1 1 range to 0 1 range of a column
def to_01_space(survived_test) :
    hash = {-1: 0, 1: 1};
    for i in range(np.shape(survived_test)[0]) :
        survived_test[i] = hash[survived_test[i]];

###############################################################################
