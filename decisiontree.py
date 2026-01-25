#6 trees f1 score = 0.5998
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import datasets, linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA

import xgboost as xgb
from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l2

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import export_graphviz
from subprocess import call

import numpy as np

import pandas as pd
from pandas import DataFrame

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
    df[:, -1] = np.where(df[:, -1] <= 7, 0, np.where(df[:, -1] <= 10, 1, np.where(df[:, -1] <= 15, 2, 3)))
    return df 

def prepare(df, i):
    X = df[:, 0:-1]
    y = df[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = i)
    
    transformer = StandardScaler()
    x_train = transformer.fit_transform(x_train)
    x_test = transformer.transform(x_test)

    return x_train, x_test, y_train, y_test

def decision(x_train, x_test, y_train, y_test, depth):

    model = DecisionTreeClassifier(random_state = 0, max_depth = depth)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_pred_train = model.predict(x_train)

    acc = accuracy_score(y_test, y_pred)
    acc_train = accuracy_score(y_train, y_pred_train)

    f1 = f1_score(y_test, y_pred, average = 'weighted')
    f1_train = f1_score(y_train, y_pred_train, average = 'weighted') 
    
    return acc, acc_train, f1, f1_train
    
def main():
    df = pd.read_csv('abalone.data', header=None, names=["sex", "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight", "rings"])
    df = process_data(df)
    print(df)

    accmean = []
    accstd = []
    acc_trainmean = []
    acc_trainstd = []

    f1mean = []
    f1std = []
    f1_trainmean = []
    f1_trainstd = []
    for depth in range(1, 11):
        acc_all = np.zeros(10)
        acc_train_all = np.zeros(10)
        f1_all = np.zeros(10)
        f1_train_all = np.zeros(10)
        for i in range(10):
            x_train, x_test, y_train, y_test = prepare(df, i)
            acc, acc_train, f1, f1_train = decision(x_train, x_test, y_train, y_test, depth)        
            acc_all[i] = acc
            acc_train_all[i] = acc_train
            f1_all[i] = f1
            f1_train_all[i] = f1_train
        print(acc_all, 'acc') 
        print(acc_train_all, 'acc_train') 
        print(f1_all, 'f1') 
        print(f1_train_all, 'f1_train') 

        accmean.append(round(acc_all.mean(), 4)), accstd.append(round(acc_all.std(), 4)) #accuracy for each minleaf
        acc_trainmean.append(round(acc_train_all.mean(), 4)), acc_trainstd.append(round(acc_train_all.std(), 4))
        f1mean.append(round(f1_all.mean(), 4)), f1std.append(round(f1_all.std(), 4)) 
        f1_trainmean.append(round(f1_train_all.mean(), 4)), f1_trainstd.append(round(f1_train_all.std(), 4))
    print(accmean, 'accmean') 
    print(acc_trainmean, 'acc_trainmean') 
    print(accstd, 'accstd')
    print(acc_trainstd, 'acc_trainstd')
    print(f1mean, 'f1mean') 
    print(f1_trainmean, 'f1_trainmean') 
    print(f1std, 'f1std')
    print(f1_trainstd, 'f1_trainstd')

    index = np.argmax(accmean)
    print(index, max(accmean))
    index2 = np.argmax(f1mean)
    print(index2, max(f1mean))

    x_train, x_test, y_train, y_test = prepare(df, 10)
    best_tree = DecisionTreeClassifier(random_state = 0, max_depth = index2 + 1)
    best_tree.fit(x_train, y_train)
    
    export_graphviz(best_tree, out_file='tree.dot', 
                    feature_names = ("F", "I", "M", "Length", "Diameter", "Height", "Whole weight", "Schucked weight", "Viscvera weight", "Shell weight"),
                    class_names = ("0-7 years", "8-10 years", "11-15 years", ">15 years"),
                    rounded = True, proportion = False, 
                    precision = 3, filled = True)

    call(['dot', '-Tpdf', 'tree.dot', '-o', 'best_tree.pdf', '-Gdpi=600'])

if __name__ == '__main__':
    main()

