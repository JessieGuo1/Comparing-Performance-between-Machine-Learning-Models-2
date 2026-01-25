#using treelist gradboost: best accmean - 0.629, xg: 0.6126 nest = 100; using range(1, 10): 0.617; 0.6281,
#using treelist gradboost: best f1mean - 0.6192, xg: 0.6068, nestimators = 100

from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import datasets, linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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
    x_train[:, 3:] = transformer.fit_transform(x_train[:, 3:])
    x_test[:, 3:] = transformer.transform(x_test[:, 3:])

    return x_train, x_test, y_train, y_test

def boost(x_train, x_test, y_train, y_test, xggrad, trees):
    
    if xggrad == 'Gradient':
        model = GradientBoostingClassifier(n_estimators=trees, random_state = 0)

    elif xggrad == 'XG':
        model = xgb.XGBClassifier(n_estimators=trees, random_state = 0)

    model.fit(x_train, y_train)    
    y_pred = model.predict(x_test)
    y_pred_train = model.predict(x_train)

    accb = accuracy_score(y_test, y_pred)
    acc_trainb = accuracy_score(y_train, y_pred_train)

    f1 = f1_score(y_test, y_pred, average = 'weighted')
    f1_train = f1_score(y_train, y_pred_train, average = 'weighted') 

    return accb, acc_trainb, f1, f1_train

def main():
    df = pd.read_csv('abalone.data', header=None, names=["sex", "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight", "rings"])
    df = process_data(df)
    print(df)

    treelist = [100, 110, 120, 130, 140, 150]
    accmean = []
    accstd = []
    acc_trainmean = []
    acc_trainstd = []
    f1mean = []
    f1std = []
    f1_trainmean = []
    f1_trainstd = []

    accmeanx = []
    accstdx = []
    acc_trainmeanx = []
    acc_trainstdx = []
    f1meanx = []
    f1stdx = []
    f1_trainmeanx = []
    f1_trainstdx = []
    for trees in treelist:
        acc_all = np.zeros(10)
        acc_train_all = np.zeros(10)
        acc_allx = np.zeros(10)
        acc_train_allx = np.zeros(10)
        f1_all = np.zeros(10)
        f1_train_all = np.zeros(10)
        f1_allx = np.zeros(10)
        f1_train_allx = np.zeros(10)        
        for j in range(10):
            x_train, x_test, y_train, y_test = prepare(df, j)
            acc, acc_train, f1, f1_train = boost(x_train, x_test, y_train, y_test, 'Gradient', trees)
            acc_all[j] = acc
            acc_train_all[j] = acc_train
            f1_all[j] = f1
            f1_train_all[j] = f1_train

            accx, acc_trainx, f1x, f1_trainx = boost(x_train, x_test, y_train, y_test, 'XG', trees)
            acc_allx[j] = accx
            acc_train_allx[j] = acc_trainx
            f1_allx[j] = f1x
            f1_train_allx[j] = f1_trainx
        print(acc_all, 'acc') 
        print(acc_train_all, 'acc_train') 
        print(acc_allx, 'accx') 
        print(acc_train_allx, 'acc_trainx') 
        print(f1_all, 'f1') 
        print(f1_train_all, 'f1_train') 
        print(f1_allx, 'f1x') 
        print(f1_train_allx, 'f1_trainx') 
        accmean.append(round(acc_all.mean(), 4)), accstd.append(round(acc_all.std(), 4))
        acc_trainmean.append(round(acc_train_all.mean(), 4)), acc_trainstd.append(round(acc_train_all.std(), 4))
        accmeanx.append(round(acc_allx.mean(), 4)), accstdx.append(round(acc_allx.std(), 4))
        acc_trainmeanx.append(round(acc_train_allx.mean(), 4)), acc_trainstdx.append(round(acc_train_allx.std(), 4))
        f1mean.append(round(f1_all.mean(), 4)), f1std.append(round(f1_all.std(), 4)) 
        f1_trainmean.append(round(f1_train_all.mean(), 4)), f1_trainstd.append(round(f1_train_all.std(), 4))
        f1meanx.append(round(f1_allx.mean(), 4)), f1stdx.append(round(f1_allx.std(), 4)) 
        f1_trainmeanx.append(round(f1_train_allx.mean(), 4)), f1_trainstdx.append(round(f1_train_allx.std(), 4))
    print(accmean, 'accmean') 
    print(acc_trainmean, 'acc_trainmean') 
    print(accstd, 'accstd')
    print(acc_trainstd, 'acc_trainstd')
    print(f1mean, 'f1mean') 
    print(f1_trainmean, 'f1_trainmean') 
    print(f1std, 'f1std')
    print(f1_trainstd, 'f1_trainstd')   

    print(accmeanx, 'accmeanx') 
    print(acc_trainmeanx, 'acc_trainmeanx') 
    print(accstdx, 'accstdx')
    print(acc_trainstdx, 'acc_trainstdx')
    print(f1meanx, 'f1meanx') 
    print(f1_trainmeanx, 'f1_trainmeanx') 
    print(f1stdx, 'f1stdx')
    print(f1_trainstdx, 'f1_trainstdx')

    index = np.argmax(accmean)
    print(index, max(accmean))
    index2 = np.argmax(f1mean)
    print(index2, max(f1mean))
    indexx = np.argmax(accmeanx)
    print(indexx, max(accmeanx))
    index2x = np.argmax(f1meanx)
    print(index2, max(f1meanx))

if __name__ == '__main__':
    main()
