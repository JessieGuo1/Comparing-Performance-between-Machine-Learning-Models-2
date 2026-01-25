#epochs = 10 for both functions still killed program with 10 ite of trainsplit
#20 in hidden layer accmean 0.6039 (10 iter), accmean2 0.6053, accmean3 0.6091, accmean4 0.6074
#f1mean 0.5735 (10 iter), f1mean2 0.5796, f1mean3 0.5819, f1mean4 0.5766
#100: 0.6356 accmean (10 iter), 0.6369 accmean2, 0.628 accmean3, 0.6166 accmean4
#0.6199 f1mean (10 iter), 0.622 f1mean2, 0.6081 f1mean3, 0.5879 f1mean4
#20 0.6053 accmean (5 iter)
#0.5798 f1mean (5 iter)
#100: (5 iter) 0.6375 accmean; 0.6369 accmean2; 0.628 accmean3; 0.6166 accmean4
#0.623 (5 iter) f1mean; 0.622 f1mean2; 0.6081 f1mean3; 0.5879 f1mean4
#100: 5 iter 0.6367 accmeanka; 0.6363 accmean2; 0.6296 accmean3; 0.6166 accmean4
#0.623 f1meanka; 0.622 f1mean2; 0.6111 f1mean3; 0.5891 f1mean4
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
from keras import backend as K
from keras.utils import set_random_seed

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

def reg(x_train, x_test, y_train, y_test):
    set_random_seed(0)
    model = Sequential()
    model.add(Dense(100, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(4, activation = 'softmax'))
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, verbose=0)
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    _, acc_train = model.evaluate(x_train, y_train, verbose=1)
    
    y_pred = model.predict(x_test)
    print(y_pred)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred_train = model.predict(x_train)
    y_pred_train = np.argmax(y_pred_train, axis=1)
    print(y_pred)
    print(y_pred_train)
    f1 = f1_score(y_test, y_pred, average = 'weighted')
    f1_train = f1_score(y_train, y_pred_train, average = 'weighted')
    K.clear_session()

    return acc, acc_train, f1, f1_train

def regul(x_train, x_test, y_train, y_test, weight):
    set_random_seed(0)
    model = Sequential()
    model.add(Dense(100, input_dim=x_train.shape[1], activation='relu', kernel_regularizer=l2(weight)))
    model.add(Dense(4, activation = 'softmax'))
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, verbose=0)
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    _, acc_train = model.evaluate(x_train, y_train, verbose=1)
    
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred_train = model.predict(x_train)
    y_pred_train = np.argmax(y_pred_train, axis=1)

    f1 = f1_score(y_test, y_pred, average = 'weighted')
    f1_train = f1_score(y_train, y_pred_train, average = 'weighted')
    K.clear_session()

    return acc, acc_train, f1, f1_train

def main():
    df = pd.read_csv('abalone.data', header=None, names=["sex", "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight", "rings"])
    df = process_data(df)
    print(df)

    acc_all = np.zeros(5)
    acc_train_all = np.zeros(5)
    f1_all = np.zeros(5)
    f1_train_all = np.zeros(5)

    weights = [0.001, 0.01, 0.05]
    for i in range(5):
        x_train, x_test, y_train, y_test = prepare(df, i)

        acc, acc_train, f1, f1_train = reg(x_train, x_test, y_train, y_test)
        acc_all[i] = acc
        acc_train_all[i] = acc_train
        f1_all[i] = f1
        f1_train_all[i] = f1_train

    acc_all2 = np.zeros(5)
    acc_train_all2 = np.zeros(5)
    f1_all2 = np.zeros(5)
    f1_train_all2 = np.zeros(5)

    acc_all3 = np.zeros(5)
    acc_train_all3 = np.zeros(5)
    f1_all3 = np.zeros(5)
    f1_train_all3 = np.zeros(5)

    acc_all4 = np.zeros(5)
    acc_train_all4 = np.zeros(5)
    f1_all4 = np.zeros(5)
    f1_train_all4 = np.zeros(5) 

    for weight in weights:
        for j in range(5):
            x_train, x_test, y_train, y_test = prepare(df, j)
            acc2, acc_train2, f12, f1_train2 = regul(x_train, x_test, y_train, y_test, weight)
            if weight == 0.001:
                acc_all2[j] = acc2
                acc_train_all2[j] = acc_train2
                f1_all2[j] = f12
                f1_train_all2[j] = f1_train2
            elif weight == 0.01:
                acc_all3[j] = acc2
                acc_train_all3[j] = acc_train2
                f1_all3[j] = f12
                f1_train_all3[j] = f1_train2
            elif weight == 0.05:
                acc_all4[j] = acc2
                acc_train_all4[j] = acc_train2
                f1_all4[j] = f12
                f1_train_all4[j] = f1_train2 

    print(acc_all, 'acc') 
    print(acc_train_all, 'acc_train')
    print(f1_all, 'f1') 
    print(f1_train_all, 'f1_train')

    print(acc_all2, 'acc2') 
    print(acc_train_all2, 'acc_train2') 
    print(f1_all2, 'f12') 
    print(f1_train_all2, 'f1_train2')

    print(acc_all3, 'acc3') 
    print(acc_train_all3, 'acc_train3') 
    print(f1_all3, 'f13') 
    print(f1_train_all3, 'f1_train3') 

    print(acc_all4, 'acc4') 
    print(acc_train_all4, 'acc_train4') 
    print(f1_all4, 'f14') 
    print(f1_train_all4, 'f1_train4') 
    
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

    print(round(acc_all3.mean(), 4), 'accmean3') 
    print(round(acc_train_all3.mean(), 4), 'acc_trainmean3') 
    print(round(acc_all3.std(), 4), 'accstd3')
    print(round(acc_train_all3.std(), 4), 'acc_trainstd3')
    print(round(f1_all3.mean(), 4), 'f1mean3') 
    print(round(f1_train_all3.mean(), 4), 'f1_trainmean3') 
    print(round(f1_all3.std(), 4), 'f1std3')
    print(round(f1_train_all3.std(), 4), 'f1_trainstd3')

    print(round(acc_all4.mean(), 4), 'accmean4') 
    print(round(acc_train_all4.mean(), 4), 'acc_trainmean4') 
    print(round(acc_all4.std(), 4), 'accstd4')
    print(round(acc_train_all4.std(), 4), 'acc_trainstd4')
    print(round(f1_all4.mean(), 4), 'f1mean4') 
    print(round(f1_train_all4.mean(), 4), 'f1_trainmean4') 
    print(round(f1_all4.std(), 4), 'f1std4')
    print(round(f1_train_all4.std(), 4), 'f1_trainstd4')

if __name__ == '__main__':
    main()


