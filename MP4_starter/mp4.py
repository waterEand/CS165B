# Starter code for CS 165B MP3
import random

import math
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

np.random.seed(0)

def compute_metric(labels, expected):
    tp = np.sum(labels[expected == 1])
    fp = np.sum(labels[expected == 0])
    tn = np.sum(1-labels[expected == 0])
    fn = np.sum(1-labels[expected == 1])
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    error_rate = (fp+fn)/(tp+fp+tn+fn)
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    precision = tp/(tp+fp)
    f1 = 2*tp/(2*tp+fp+fn)

    return {
        "f1": f1,
        "accuracy": accuracy,
        "precision": precision,
        "tpr": tpr,
        "fpr": fpr,
        "error_rate": error_rate,
    }

def preprocess(training_data):

    # training_data['NAME_INCOME_TYPE'] = training_data['NAME_INCOME_TYPE'].replace(
    #     ['Pensioner', 'Student', 'Working', 'Commercial associate', 'State servant'], [0, 1, 2, 3, 4])
    # training_data['NAME_EDUCATION_TYPE'] = training_data['NAME_EDUCATION_TYPE'].replace(
    #     ['Lower secondary', 'Secondary / secondary special', 'Incomplete higher', 'Higher education',
    #      'Academic degree'], [0, 1, 2, 3, 4])
    # training_data['NAME_FAMILY_STATUS'] = training_data['NAME_FAMILY_STATUS'].replace(
    #     ['Single / not married', 'Widow', 'Separated', 'Married', 'Civil marriage'], [0, 1, 2, 3, 4])
    # training_data['NAME_HOUSING_TYPE'] = training_data['NAME_HOUSING_TYPE'].replace(
    #     ['With parents', 'Rented apartment', 'Co-op apartment', 'Office apartment', 'Municipal apartment',
    #      'House / apartment'], [0, 1, 2, 3, 4, 5])
    # training_data['QUANTIZED_INC'] = training_data['QUANTIZED_INC'].replace(
    #     ['lowest', 'low', 'medium', 'high', 'highest'], [0, 1, 2, 3, 4])
    # training_data['QUANTIZED_AGE'] = training_data['QUANTIZED_AGE'].replace(
    #     ['lowest', 'highest', 'low', 'high', 'medium'], [0, 0, 1, 1, 2])
    # training_data['QUANTIZED_WORK_YEAR'] = training_data['QUANTIZED_WORK_YEAR'].replace(
    #     ['lowest', 'low', 'medium', 'high', 'highest'], [0, 1, 2, 3, 4])

    le = LabelEncoder()
    training_data['NAME_INCOME_TYPE'] = le.fit_transform(training_data['NAME_INCOME_TYPE'])
    training_data['NAME_EDUCATION_TYPE'] = le.fit_transform(training_data['NAME_EDUCATION_TYPE'])
    training_data['NAME_FAMILY_STATUS'] = le.fit_transform(training_data['NAME_FAMILY_STATUS'])
    training_data['NAME_HOUSING_TYPE'] = le.fit_transform(training_data['NAME_HOUSING_TYPE'])
    training_data['QUANTIZED_INC'] = le.fit_transform(training_data['QUANTIZED_INC'])
    training_data['QUANTIZED_AGE'] = le.fit_transform(training_data['QUANTIZED_AGE'])
    training_data['QUANTIZED_WORK_YEAR'] = le.fit_transform(training_data['QUANTIZED_WORK_YEAR'])
    training_data['OCCUPATION_TYPE'] = le.fit_transform(training_data['OCCUPATION_TYPE'])


    sc = StandardScaler()
    training_data[['NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','QUANTIZED_INC','QUANTIZED_AGE',
                   'QUANTIZED_WORK_YEAR','OCCUPATION_TYPE','CNT_CHILDREN','AMT_INCOME_TOTAL','DAYS_BIRTH','DAYS_EMPLOYED','CNT_FAM_MEMBERS']] =\
        sc.fit_transform(training_data[['NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','QUANTIZED_INC','QUANTIZED_AGE',
                   'QUANTIZED_WORK_YEAR','OCCUPATION_TYPE','CNT_CHILDREN','AMT_INCOME_TOTAL','DAYS_BIRTH','DAYS_EMPLOYED','CNT_FAM_MEMBERS']])

    # training_data[['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','FLAG_MOBIL','FLAG_WORK_PHONE','FLAG_PHONE','FLAG_EMAIL']] = \
    #     sc.fit_transform(training_data[['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','FLAG_MOBIL','FLAG_WORK_PHONE','FLAG_PHONE','FLAG_EMAIL']])

    return training_data

def run_train_test(training_data: pd.DataFrame, testing_data: pd.DataFrame) -> List[int]:
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.

    Inputs:
        training_data: 
        testing_data: the same as training_data with "target" removed.

    Output:
        testing_prediction: List[int]
    Example output:
    return random.choices([0, 1, 2], k=len(testing_data))
    """


    training_data = preprocess(training_data)
    testing_data = preprocess(testing_data)

    x_train = training_data.iloc[:,0:-1].values
    y_train = training_data.iloc[:,-1].values
    x_test = testing_data.values

    # x_train = np.delete(x_train, [10], 1)
    # x_test = np.delete(x_test, [10], 1)
    # print(x_train[0])
    clf = MLPClassifier(hidden_layer_sizes=(256,128,64,8),batch_size=256,learning_rate_init=0.001,max_iter=300,activation = 'relu',solver='adam',random_state=1)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    # print(pred)

    # predict = np.zeros(len(testing_data))

    return pred


if __name__ == '__main__':

    training = pd.read_csv('data/train.csv')
    development = pd.read_csv('data/dev.csv')

    target_label = development['target']
    development.drop('target', axis=1, inplace=True)
    prediction = run_train_test(training, development)
    target_label = target_label.values
    status = compute_metric(prediction, target_label)
    print(status)

    


    


