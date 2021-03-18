import sys
import numpy as np
import pandas as pd
from lib.utils import *
from lib.metrics import *
from lib.my_logistic_regression import MyLogisticRegression as MLR

def change_cat_to_int(df, cat):
    if cat == 'Hogwarts House':
        list_val = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
    else:
        list_val = df[cat].unique()
    for i, elem in enumerate(list_val):
        df.loc[df[cat] == elem, cat] = i
    return df

def one_versus_all_train(x_train, y_train):
    cat_unique = np.unique(y_train)
    val = np.ones(x_train.shape[0])
    prob = np.zeros(x_train.shape[0])
    test = val.astype(float)
    theta_list = []
    for planete in cat_unique:
        verif = y_train == planete
        y_zero_train = verif.astype(float)
        mlr = MLR(np.ones(x_train.shape[1] + 1), alpha= 1e-3, n_cycle=100000) #100000
        mlr.fit_(x_train, y_zero_train)
        y_hat = mlr.predict_(x_train)
        for i, planete_prob in enumerate(y_hat.tolist()):
            # print(prob[i],"\n", planete_prob)
            if prob[i] < planete_prob[0]:
                val[i] = planete
                prob[i] = planete_prob[0]
        theta_list.append(pd.DataFrame(mlr.theta.reshape(-1,)))
    ret_list = []
    ret_list.append(val)
    theta_array = pd.concat(theta_list, axis=1)
    ret_list.append(theta_array.values)
    return ret_list

def init_exo(filename):
    res = pd.read_csv(filename, index_col=0).drop(columns=['First Name', 'Last Name', 'Birthday']).drop_duplicates().dropna()
    res = change_cat_to_int(res, "Best Hand")
    res = change_cat_to_int(res, 'Hogwarts House')
    y_train = res['Hogwarts House']
    x_train = res.drop(columns = ['Hogwarts House'])
    data = data_spliter(np.array(x_train), np.array(y_train).reshape(-1, 1), 0.6)
    pd.DataFrame(data[0]).to_csv("resources/x_train.csv", header = x_train.columns)
    pd.DataFrame(data[1]).to_csv("resources/y_train.csv", header = ['Hogwarts House'])
    pd.DataFrame(data[2]).to_csv("resources/x_test.csv", header = x_train.columns)
    pd.DataFrame(data[3]).to_csv("resources/y_test.csv", header = ['Hogwarts House'])

def logreg_train():
    x_train = pd.read_csv("resources/x_train.csv", index_col=0).values
    for i in range(1, x_train.shape[1]):
        x_train[:,i] = minmax_normalization(x_train[:,i])
    y_train = pd.read_csv("resources/y_train.csv", index_col=0).values
    ret = one_versus_all_train(x_train, y_train)
    pd.DataFrame(ret[1]).to_csv("theta.csv")
    y_predict = ret[0].reshape(-1, 1)
    print("F1 Score: ", f1_score_(y_train, y_predict, 1.0))
    print("Accuracy Score: ", accuracy_score_(y_train, y_predict))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("incorrect number of arguments")
    elif sys.argv[1][-3:] != "csv":
        print("wrong data file extension")
    else:
        try:
            # init_exo(sys.argv[1])
            logreg_train()
        except:
            print("That filename doesn't exist")