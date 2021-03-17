import sys
import pandas as pd
import numpy as np
from lib.visualization import Komparator as KP

def change_cat_to_int(df, cat):
    if cat == 'Hogwarts House':
        list_val = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
    else:
        list_val = df[cat].unique()
    for i, elem in enumerate(list_val):
        df.loc[df[cat] == elem, cat] = i
    return df

def init_data(filename):
    res = pd.read_csv(filename, index_col=0).drop(columns=['First Name', 'Last Name', 'Birthday']).drop_duplicates().dropna()
    res = change_cat_to_int(res, "Best Hand")
    visu = KP(res)
    visu.compare_histograms("Hogwarts House", res.drop(columns = ['Hogwarts House']).columns[3:5])
    
    # y_train = res['Hogwarts House']
    # x_train = res.drop(columns = ['Hogwarts House'])
    # data = data_spliter(np.array(x_train), np.array(y_train).reshape(-1, 1), 0.6)
    # pd.DataFrame(data[0]).to_csv("resources/x_train.csv", header = x_train.columns)
    # pd.DataFrame(data[1]).to_csv("resources/y_train.csv", header = ['Hogwarts House'])
    # pd.DataFrame(data[2]).to_csv("resources/x_test.csv", header = x_train.columns)
    # pd.DataFrame(data[3]).to_csv("resources/y_test.csv", header = ['Hogwarts House'])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("incorrect number of arguments")
    elif sys.argv[1][-3:] != "csv":
        print("wrong data file extension")
    else:
        # try:
        init_data(sys.argv[1])
        # except:
        print("That filename doesn't exist")