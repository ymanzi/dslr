import sys
import pandas as pd
import numpy as np
from lib.metrics import *
from lib.my_logistic_regression import MyLogisticRegression as MLR
from lib.utils import minmax_normalization

def export_predict_to_file(y_predict):
    dic_cat = {1:'Ravenclaw', 2:'Slytherin', 3:'Gryffindor', 4:'Hufflepuff'}
    for i, elem in enumerate(y_predict):
        y_predict[i] = [i, dic_cat[y_predict[i]]]
    pd.DataFrame(y_predict, columns = ['Index','Hogwarts House']).to_csv('houses.csv', index=False)

def one_versus_all_test(x_test, theta_list):
    dic_cat = {1:'Ravenclaw', 2:'Slytherin', 3:'Gryffindor', 4:'Hufflepuff'}
    y_predict = np.zeros(x_test.shape[0])
    tmp_prob = np.zeros(x_test.shape[0])
    for j, school in dic_cat.items():
        mlr = MLR(theta_list[:,j - 1])
        y_hat = mlr.predict_(x_test)
        for i, school_prob in enumerate(y_hat.tolist()):
            if tmp_prob[i] < school_prob[0]:
                y_predict[i] = j
                tmp_prob[i] = school_prob[0]
    return y_predict

def prediction_function(data_file, theta_file):
    x_test = pd.read_csv(data_file, index_col=0).drop(columns=['Hogwarts House','First Name','Last Name','Birthday', 'Arithmancy'])
    x_test["Best Hand"].replace({"Left":1, "Right":2}, inplace=True)
    x_test.fillna(x_test.median(), inplace=True)
    x_test = x_test.values
    for i in range(1, x_test.shape[1]):
        x_test[:,i] = minmax_normalization(x_test[:,i])
    theta = pd.read_csv(theta_file , index_col=0).values
    y_predict = one_versus_all_test(x_test, theta)
    export_predict_to_file(y_predict.tolist())

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("incorrect number of arguments")
    elif sys.argv[1][-3:] != "csv" or sys.argv[2][-3:] != "csv":
        print("wrong data file extension")
    else:
        try:
            prediction_function(sys.argv[1], sys.argv[2])
        except:
            print("That filename doesn't exist")