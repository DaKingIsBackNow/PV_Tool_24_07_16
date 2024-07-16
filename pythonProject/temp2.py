import pandas as pd
import datetime
import logging
import numpy as np
import os 
from main_pv.standard_variables import *
import main_pv.solar_functions as solar_functions
from numba import njit

DEBUGGING = False

if DEBUGGING:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

@njit
def get_first_index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx
    # If no item was found return None, other return types might be a problem due to
    # numbas type inference.
    return None


arr1 = [np.ones((100)) for _ in range(3)]
arr1[0][2] = 0
arr1[1][5] = 0
arr1[2][12] = 0

first_indexes_zero = [get_first_index(column, 0) for column in arr1]
for i in range(len(first_indexes_zero)):
    end_index = first_indexes_zero[i][0]
    arr1[i][: end_index] = 0

for i in range(len(first_indexes_zero)):
    print(f"{i} : \n")
    print(arr1[i])

