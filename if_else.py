#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# ============================================================================
#       Filename:  if_else.py
#    Description:  If-Else hand-made machine learning
#        Created:  2015-02-11 19:45 
#         Author:  Tiago Lobato Gimenes        (tlgimenes@gmail.com)
#     Co-Authors:  <-- put your name here -->
#      Copyright:  Tiago Lobato Gimenes
#       Lincense:  MIT
# ============================================================================

"""
If else like file from the internet
"""

import numpy as np
import pandas
#import statsmodels.api as sm

def custom_heuristic(file_path):
    '''
    You are given a list of Titantic passengers and their associating
    information. More information about the data can be seen at the link below:
    http://www.kaggle.com/c/titanic-gettingStarted/data

    For this exercise, you need to write your custom heuristic that will take
    in some combination of the passenger's attributes and predict if the passenger
    survived the titanic diaster.

    Can your custom heuristic beat 80% accuracy?

    The available attributes are:
    Pclass          Passenger Class
                    (1 = 1st; 2 = 2nd; 3 = 3rd)
    Name            Name
    Sex             Sex
    Age             Age
    SibSp           Number of Siblings/Spouses Aboard
    Parch           Number of Parents/Children Aboard
    Ticket          Ticket Number
    Fare            Passenger Fare
    Cabin           Cabin
    Embarked        Port of Embarkation
                    (C = Cherbourg; Q = Queenstown; S = Southampton)

    SPECIAL NOTES:
    Pclass is a proxy for socio-economic status (SES)
    1st ~ Upper; 2nd ~ Middle; 3rd ~ Lower

    Age is in Years; Fractional if Age less than One (1)
    If the Age is Estimated, it is in the form xx.5

    With respect to the family relation variables (i.e. SibSp and Parch)
    some relations were ignored.  The following are the definitions used
    for SibSp and Parch.

    Sibling:  Brother, Sister, Stepbrother, or Stepsister of Passenger Aboard Titanic
    Spouse:   Husband or Wife of Passenger Aboard Titanic (Mistresses and Fiances Ignored)
    Parent:   Mother or Father of Passenger Aboard Titanic
    Child:    Son, Daughter, Stepson, or Stepdaughter of Passenger Aboard Titanic

    Write your prediction back into the "predictions" dictionary. The
    key of the dictionary should be the Passenger's id (which can be accessed
    via passenger["PassengerId"]) and the associating value should be 1 if the
    passenger survied or 0 otherwise.

    For example, if a passenger survived:
    passenger_id = passenger['PassengerId']
    predictions[passenger_id] = 1

    Or if a passenger perished in the disaster:
    passenger_id = passenger['PassengerId']
    predictions[passenger_id] = 0

    You can also look at the titantic data that you will be working with
    at the link below:
    https://www.dropbox.com/s/r5f9aos8p9ri9sa/titanic_data.csv
    '''

    predictions = {}
    df = pandas.read_csv(file_path)
    for passenger_index, passenger in df.iterrows():
        #
        # your code here
        #
        if (passenger['Sex'] == 'female' and passenger['Pclass'] == 1) or (
passenger['Sex'] == 'female' and passenger['Pclass'] == 2) or (
passenger['Sex'] == 'female' and passenger['Pclass'] == 3 and passenger
['Age'] < 28) or (passenger['Age'] < 16 and passenger['Pclass'] == 2) or (
passenger['Age'] < 29 and passenger['Pclass'] == 1) or (passenger['Parch'] > 5) or (
passenger['Age'] < 1 and passenger['Pclass'] == 3):
            predictions[passenger['PassengerId']] = 1
        else:
            predictions[passenger['PassengerId']] = 0

    return np.asarray(predictions.values());

