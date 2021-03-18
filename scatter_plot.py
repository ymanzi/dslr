import sys
import pandas as pd
import numpy as np
from lib.visualization import Komparator as KP

def init_data(filename):
    res = pd.read_csv(filename, index_col=0).drop(columns=['First Name', 'Last Name', 'Birthday', 'Best Hand']).drop_duplicates().dropna()
    visu = KP(res)
    visu.scatterplot_("Hogwarts House", res.drop(columns = ['Hogwarts House']).columns)

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