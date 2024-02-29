# Starter code for CS 165B MP3
import math
import random
import numpy as np
import pandas as pd

from typing import List

def run_train_test(training_data: pd.DataFrame, testing_data: pd.DataFrame) -> List[int]:
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.

    Inputs:
        training_data: pd.DataFrame
        testing_data: the same as training_data with "label" removed.

    Output:
        testing_prediction: List[int]
    Example output:
    return random.choices([0, 1, 2], k=len(testing_data))
    """
    # w = np.zeros(shape=6)
    # w = np.array([0.44,0.54,0.45,0.39,0.28,-0.75]) # [0.52,0.68,0.55,0.47,0.32,-0.9]
    w = np.zeros(shape = 6)

    eta = 0.01
    cvg = False
    epoch = 0

    training_data['target'] = training_data['target'].replace(to_replace=0,value=-1)

    while not cvg:
        cvg = True
        for idx, row in training_data.iterrows():
            nprow = row[['x1','x2','x3','x4','x5']].to_numpy()
            nprow = np.append(nprow, 1.0)

            judge = row['target'] * (w.T @ nprow)

            if judge <= 0.0:
                w = w + eta * row['target'] * nprow
                cvg = False
        epoch += 1
        if epoch > 300:
            break

    # print(w)


    res = []
    for idx, row in testing_data.iterrows():
        nprow = row[['x1', 'x2', 'x3', 'x4', 'x5']].to_numpy()
        nprow = np.append(nprow, 1.0)
        val = w.T @ nprow
        if val > 0:
            res.append(1)
        else:
            res.append(0)

    return res
    # return random.choices([0, 1, 2], k=len(testing_data))

    #TODO implement your model and return the prediction

if __name__ == '__main__':
    # load data
    training = pd.read_csv('data/train.csv')
    testing = pd.read_csv('data/dev.csv')
    target_label = testing['target']
    testing.drop('target', axis=1, inplace=True)

    # run training and testing
    prediction = run_train_test(training, testing)

    # check accuracy
    target_label = target_label.values
    print("Dev Accuracy: ", np.sum(prediction == target_label) / len(target_label))
    


    


