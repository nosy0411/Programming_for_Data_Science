import csv
import os
import pandas as pd
from getdata import dataset
from sklearn.model_selection import train_test_split

path="./assignment6/dataset"
# print(os.listdir(path))
# print(os.path.join(path,'*.csv'))

for input_file in os.listdir(path):
    data, target = dataset(input_file)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=0)
    print("=============================")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    print("=============================")
    break