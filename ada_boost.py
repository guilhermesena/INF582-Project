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
types_clf = ['Pclass', 'Age', 'Sex'];

###############################################################################
## Does the adaBoost and returns an array containning died or not 
## @arg0: Train data
## @arg1: Test data
## @returns: Array with 1 if survived 0 if not
def ada_boost(train_data, test_data) :
    # Create and fit an AdaBoosted decision tree
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=10);

    # Select columns of train_data 
    survived_training = train_data['Survived'];
    learning_data = to_ndarray(train_data, types_clf);

    # Fits model
    bdt.fit(learning_data, survived_training);

    # Predicts
    survived_test = bdt.predict(to_ndarray(test_data, types_clf));

    # Converts -1,1 space to 0 1 space
    survived_test = [{-1:0, 1:1}[survived] for survived in survived_test];

    return survived_test;

###############################################################################
## Converts a tupple to an ndarray
def to_ndarray(tupple, tupple_type):
    i0 = np.shape(tupple[tupple_type])[0];
    j0 = np.shape(tupple_type)[0];
    mat = np.ndarray((i0,j0));

    for i in range(i0):
        for j in range(j0):
            mat[i][j] = tupple[tupple_type][i][j];

    return mat;
###############################################################################
