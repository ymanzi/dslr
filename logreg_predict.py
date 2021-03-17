import sys
import pandas as pd
import numpy as np
from lib.metrics import *
from lib.my_logistic_regression import MyLogisticRegression as MLR
from lib.utils import minmax_normalization

def one_versus_all_test(x_test, y_test, theta_list):
    cat_unique = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
    y_predict = np.ones(y_test.shape[0])
    tmp_prob = np.zeros(y_test.shape[0])
    for j, school in enumerate(cat_unique):
        mlr = MLR(theta_list[:,j])
        y_hat = mlr.predict_(x_test)
        for i, school_prob in enumerate(y_hat.tolist()):
            if tmp_prob[i] < school_prob[0]:
                y_predict[i] = j
                tmp_prob[i] = school_prob[0]
    return y_predict

def prediction_function(filename):
    x_test = pd.read_csv(filename, index_col=0).values
    for i in range(1, x_test.shape[1]):
        x_test[:,i] = minmax_normalization(x_test[:,i])
    y_test = pd.read_csv("resources/y_test.csv", index_col=0).values
    theta_list = pd.read_csv("theta.csv", index_col=0).values
    y_predict = one_versus_all_test(x_test, y_test, theta_list)
    print(accuracy_score_(y_test, y_predict))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("incorrect number of arguments")
    elif sys.argv[1][-3:] != "csv":
        print("wrong data file extension")
    else:
        try:
            prediction_function(sys.argv[1])
        except:
            print("That filename doesn't exist")