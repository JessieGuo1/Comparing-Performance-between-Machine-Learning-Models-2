#no difference in accmean 0.5791, std 0.006, similarly for training
#same with 0.5469 f1mean
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
    x_train = transformer.fit_transform(x_train)
    x_test = transformer.transform(x_test)

    return x_train, x_test, y_train, y_test

def pca2(df, var, i):
    
    if var == 0.95:
        pca = PCA(n_components=0.95)
    elif var == 0.98:
        pca = PCA(n_components=0.98)
    
    X = df[:, 0:-1]
    y = df[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = i)
    pca.fit(x_train)
    print(pca.fit(x_train), 'fit')
    x_reduced_train = pca.transform(x_train)
    print(x_reduced_train, 'x_reduced_train')
    x_reduced_test = pca.transform(x_test)
    print(x_reduced_test, 'x_reduced_test')
    explained_variance = pca.explained_variance_ratio_
    print(f'\nReduced data shape: {x_reduced_train.shape}, Total variance explained: {explained_variance.sum():.3f}')
    print(f'Variance explained by principle components: {explained_variance}')

    transformer = StandardScaler()
    x_reduced_train = transformer.fit_transform(x_reduced_train)
    x_reduced_test = transformer.transform(x_reduced_test)

    model = GradientBoostingClassifier(n_estimators=100, random_state = 0)
    model.fit(x_reduced_train, y_train)    
    y_pred = model.predict(x_reduced_test)
    y_pred_train = model.predict(x_reduced_train)

    acc = accuracy_score(y_test, y_pred)
    acc_train = accuracy_score(y_train, y_pred_train)

    f1 = f1_score(y_test, y_pred, average = 'weighted')
    f1_train = f1_score(y_train, y_pred_train, average = 'weighted') 

    return acc, acc_train, f1, f1_train
    
def main():
    df = pd.read_csv('abalone.data', header=None, names=["sex", "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight", "rings"])
    df = process_data(df)
    print(df)

    acc_all = np.zeros(10)
    acc_train_all = np.zeros(10)
    acc_all2 = np.zeros(10)
    acc_train_all2 = np.zeros(10)
    f1_all = np.zeros(10)
    f1_train_all = np.zeros(10)
    f1_all2 = np.zeros(10)
    f1_train_all2 = np.zeros(10)  

    for i in range(10):

        acc, acc_train, f1, f1_train = pca2(df, 0.95, i)
        acc_all[i] = acc
        acc_train_all[i] = acc_train
        f1_all[i] = f1
        f1_train_all[i] = f1_train

        acc2, acc_train2, f12, f1_train2 = pca2(df, 0.98, i)
        acc_all2[i] = acc2
        acc_train_all2[i] = acc_train2
        f1_all2[i] = f12
        f1_train_all2[i] = f1_train2

    print(round(acc_all.mean(), 4), 'accmean') 
    print(round(acc_train_all.mean(), 4), 'acc_trainmean') 
    print(round(acc_all.std(), 4), 'accstd')
    print(round(acc_train_all.std(), 4), 'acc_trainstd')
    print(round(f1_all.mean(), 4), 'f1mean') 
    print(round(f1_train_all.mean(), 4), 'f1_trainmean') 
    print(round(f1_all.std(), 4), 'f1std')
    print(round(f1_train_all.std(), 4), 'f1_trainstd')
        
    print(round(acc_all2.mean(), 4), 'accmean2') 
    print(round(acc_train_all2.mean(), 4), 'acc_trainmean2') 
    print(round(acc_all2.std(), 4), 'accstd2')
    print(round(acc_train_all2.std(), 4), 'acc_trainstd2')
    print(round(f1_all2.mean(), 4), 'f1mean2') 
    print(round(f1_train_all2.mean(), 4), 'f1_trainmean2') 
    print(round(f1_all2.std(), 4), 'f1std2')
    print(round(f1_train_all2.std(), 4), 'f1_trainstd2')
        
if __name__ == '__main__':
    main()
