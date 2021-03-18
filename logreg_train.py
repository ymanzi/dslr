import sys
import numpy as np
import pandas as pd
from lib.utils import *
from lib.metrics import *
from lib.my_logistic_regression import MyLogisticRegression as MLR

def change_cat_to_int(df, cat):
    """
        Change for example the Best Hand categorie, to 0 and 1
        or the Hogwarts house to 0, 1, 2, or 3
        To allow us to use the date in thoses categories
    """
    if cat == 'Hogwarts House':
        list_val = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
    else:
        list_val = df[cat].unique()
    for i, elem in enumerate(list_val):
        df.loc[df[cat] == elem, cat] = i
    return df

def one_versus_all_train(x_train, y_train, gradient_type):
    """
        for each school, it will calculate the probability of the student to go to that
        school instead of any other.
        At the end it will keep the biggest probability  
    """
    cat_unique = np.unique(y_train)
    y_predict = np.ones(x_train.shape[0])
    prob = np.zeros(x_train.shape[0])
    theta_list = []
    for school in cat_unique:
        verif = y_train == school
        y_zero_train = verif.astype(float)
        mlr = MLR(np.ones(x_train.shape[1] + 1), alpha= 1e-3, n_cycle=100000, grandient_type= gradient_type) #100000
        mlr.fit_(x_train, y_zero_train)
        y_hat = mlr.predict_(x_train)
        for i, school_prob in enumerate(y_hat.tolist()):
            if prob[i] < school_prob[0]:
                y_predict[i] = school
                prob[i] = school_prob[0]
        theta_list.append(pd.DataFrame(mlr.theta.reshape(-1,)))
    ret_list = list([y_predict, pd.concat(theta_list, axis=1) ])
    return ret_list

def split_data(filename):
    res = pd.read_csv(filename, index_col=0)
    x_train = res
    y_train = res['Hogwarts House']
    data = data_spliter(x_train.values, y_train.values.reshape(-1, 1), 0.6) # , np.array(y_train).reshape(-1, 1), 0.6)
    pd.DataFrame(data[0]).to_csv("resources/x_train.csv", header = x_train.columns)
    pd.DataFrame(data[1]).to_csv("resources/y_train.csv", header = ['Hogwarts House'])
    pd.DataFrame(data[2]).to_csv("resources/x_test.csv", header = x_train.columns)
    pd.DataFrame(data[3]).to_csv("resources/y_test.csv", header = ['Hogwarts House'])

def init_exo(filename):
    res = pd.read_csv(filename, index_col=0).drop(columns=['First Name', 'Last Name', 'Birthday', 'Defense Against the Dark Arts']).drop_duplicates().dropna()
    res = change_cat_to_int(res, "Best Hand")
    res = change_cat_to_int(res, 'Hogwarts House')
    y_train = res['Hogwarts House'].values.reshape(-1, 1)
    x_train = res.drop(columns = ['Hogwarts House']).values
    return list([x_train, y_train])

def logreg_train(data, gradient_type='batch'):
    x_train = data[0].astype(float)
    for i in range(1, x_train.shape[1]):
        x_train[:,i] = minmax_normalization(x_train[:,i])
    y_train = data[1]
    ret = one_versus_all_train(x_train, y_train, gradient_type)
    pd.DataFrame(ret[1]).to_csv("theta.csv")
    y_predict = ret[0].reshape(-1, 1)
    print("F1 Score: ", f1_score_(y_train, y_predict, 1.0))
    print("Accuracy Score: ", accuracy_score_(y_train, y_predict))

if __name__ == "__main__":
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("incorrect number of arguments")
    elif sys.argv[1][-3:] != "csv":
        print("wrong data file extension")
    elif len(sys.argv) == 3 and sys.argv[2] not in ['stochastic', 'batch', 'mini_batch']:
        print("wrong argument for gradient-type")
    else:
        try:
            split_data(sys.argv[1])
            data = init_exo(sys.argv[1])  # data = [x_train, y_train]
            if len(sys.argv) == 2:
                logreg_train(data)
            else:
                logreg_train(data, sys.argv[2])# The second parameter is the gradient type
        except:
            print("That filename doesn't exist")