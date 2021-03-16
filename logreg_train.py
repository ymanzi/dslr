import numpy as np
import pandas as pd
from lib.my_logistic_regression import data_spliter
from lib.my_logistic_regression import MyLogisticRegression as MLR
from lib.my_logistic_regression import minmax_normalization


def check_positive_negative(y: np.ndarray, y_hat: np.ndarray, categorie):
    dic_pos_neg = { "true positives" : 0,
                    "false positives": 0,
                    "true negatives": 0,
                    "false negatives": 0}
    for e_real, e_predict in zip(y, y_hat):
        if e_real == e_predict and e_real == categorie:
            dic_pos_neg["true positives"] += 1
        elif e_real == e_predict and e_real != categorie:
            dic_pos_neg["true negatives"] += 1
        elif e_real != e_predict and e_real == categorie:
            dic_pos_neg["false negatives"] += 1
        elif e_real != e_predict and e_predict == categorie:
            dic_pos_neg["false positives"] += 1
    return dic_pos_neg

def accuracy_score_(y: np.ndarray, y_hat: np.ndarray):
    result = np.array([e1 == e2 for e1, e2 in zip(y, y_hat)]).astype(int)
    return np.sum(result) / result.size

def precision_score_(y: np.ndarray, y_hat: np.ndarray, pos_label=1):
    """
    Compute the precision score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns: 
        The precision score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    dic_pos_neg = check_positive_negative(y, y_hat, pos_label)
    return dic_pos_neg["true positives"] / (dic_pos_neg["true positives"] + dic_pos_neg["false positives"])

def recall_score_(y, y_hat, pos_label=1):
    """
    Compute the recall score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns: 
        The recall score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    dic_pos_neg = check_positive_negative(y, y_hat, pos_label)
    return dic_pos_neg["true positives"] / (dic_pos_neg["true positives"] + dic_pos_neg["false negatives"])

def f1_score_(y, y_hat, pos_label=1):
    """
    Compute the f1 score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns: 
        The f1 score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    dic_pos_neg = check_positive_negative(y, y_hat, pos_label)
    return (2 * precision_score_(y, y_hat, pos_label) * recall_score_(y, y_hat, pos_label)) /\
         (precision_score_(y, y_hat, pos_label) + recall_score_(y, y_hat, pos_label))


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
        mlr = MLR(np.ones(x_train.shape[1] + 1), alpha= 1e-3, n_cycle=100000, penalty='none')
        mlr.fit_(x_train, y_zero_train)
        y_hat = mlr.predict_(x_train)
        for i, planete_prob in enumerate(y_hat.tolist()):
            # print(prob[i],"\n", planete_prob)
            if prob[i] < planete_prob[0]:
                val[i] = planete
                prob[i] = planete_prob[0]
        theta_list.append(mlr.theta)
    ret_list = []
    ret_list.append(val)
    # print(theta_list)
    theta_array = pd.DataFrame(theta_list[0], columns=str(0))
    for i in range(1, len(theta_list)):
        theta_array[str(i)] = pd.Series(theta_list[i], index=theta_array.index)  #concat([theta_array, pd.DataFrame(theta_list[i])])
        # theta_array = theta_array.append(pd.DataFrame(theta_list[i])).reset(columns)
    print(theta_array)
    ret_list.append(theta_array.values)
    return ret_list

def init_exo():
    res = pd.read_csv("resources/dataset_train.csv", index_col=0).drop(columns=['First Name', 'Last Name', 'Birthday']).drop_duplicates().dropna()
    res = change_cat_to_int(res, "Best Hand")
    res = change_cat_to_int(res, 'Hogwarts House')
    y_train = res['Hogwarts House']
    x_train = res.drop(columns = ['Hogwarts House'])
    data = data_spliter(np.array(x_train), np.array(y_train).reshape(-1, 1), 0.6)
    pd.DataFrame(data[0]).to_csv("x_train.csv", header = x_train.columns)
    pd.DataFrame(data[1]).to_csv("y_train.csv", header = ['Hogwarts House'])
    pd.DataFrame(data[2]).to_csv("x_test.csv", header = x_train.columns)
    pd.DataFrame(data[3]).to_csv("y_test.csv", header = ['Hogwarts House'])

def logreg_train():
    x_train = pd.read_csv("x_train.csv", index_col=0).values
    for i in range(1, x_train.shape[1]):
        x_train[:,i] = minmax_normalization(x_train[:,i])
    y_train = pd.read_csv("y_train.csv", index_col=0).values
    # print(x_train)
    ret = one_versus_all_train(x_train, y_train)
    pd.DataFrame(ret[1]).to_csv("theta.csv")
    y_predict = ret[0].reshape(-1, 1)
    print(f1_score_(y_train, y_predict, 1.0))
    print(accuracy_score_(y_train, y_predict))

    # print(ret[1])

# init_exo()
logreg_train()

# print(np.array(y_train).reshape(-1, 1).shape)
# print(np.array(x_train).shape)


# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("incorrect number of arguments")
#     else:
