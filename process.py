from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets, linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve 
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import numpy as np

import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

def process_data(df):
    values = np.array(df.iloc[:, 0])
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    print(integer_encoded)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse_output=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    onehot_encoded = pd.DataFrame(onehot_encoded, columns = ["F", "I", "M"])
    print(onehot_encoded)
    df = df.drop(columns = "sex")
    df = pd.concat([onehot_encoded, df], axis=1)
    df = np.array(df)
    df[:, -1] = np.where(df[:, -1] <= 7, 1, np.where(df[:, -1] <= 10, 2, np.where(df[:, -1] <= 15, 3, 4)))
    return df 
    
def main():
    df = pd.read_csv('abalone.data', header=None, names=["sex", "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight", "rings"])
    df = process_data(df)
    print(df)
    print(np.unique(df[:, -1]))
    
    for i in range(0, 4):
        print(df[df[:, -1] == i+1].shape[0])
   
if __name__ == '__main__':
    main()

