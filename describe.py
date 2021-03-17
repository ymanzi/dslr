import sys
import numpy as np
import pandas as pd
from lib.TinyStatistician import *

def change_cat_to_int(df, cat):
    if cat == 'Hogwarts House':
        list_val = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
    else:
        list_val = df[cat].unique()
    for i, elem in enumerate(list_val):
        df.loc[df[cat] == elem, cat] = i
    return df

def get_stat(df):
    list_columns = df.columns
    list_rows = ['Count', 'Mean','Std','Min','25%','50%','75%','Max']
    list_data = []
    for col in list_columns:
        data = df[col].values
        list_data.append(np.array([df[col].size \
            ,  mean_(data)\
            , std_(data)\
            , min(data.tolist())\
            , quartiles_(data, 25)\
            , quartiles_(data, 50)\
            , quartiles_(data, 75)\
            , max(data)]))
    print(pd.DataFrame(np.array(list_data).transpose(), columns= list_columns, index = list_rows))

def init_data(filename):
    res = pd.read_csv(filename, index_col=0).drop(columns=['First Name', 'Last Name', 'Birthday']).drop_duplicates().dropna()
    res = change_cat_to_int(res, "Best Hand")
    res = res.drop(columns = ['Hogwarts House'])
    get_stat(res)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("incorrect number of arguments")
    elif sys.argv[1][-3:] != "csv":
        print("wrong data file extension")
    else:
        try:
            init_data(sys.argv[1])
        except:
            print("That filename doesn't exist")