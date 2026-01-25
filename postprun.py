# optimal-ccp = 0.00133 for both f1 and acc, if max-depth = 4 then both 0
# accmean 0.6157, f1mean 0.6038
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import datasets, linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeClassifier
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
    df[:, -1] = np.where(df[:, -1] <= 7, 1, np.where(df[:, -1] <= 10, 2, np.where(df[:, -1] <= 15, 3, 4)))
    return df  
 
def prepare(df, i):
    X = df[:, 0:-1]
    y = df[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = i)
    
    transformer = StandardScaler()
    x_train = transformer.fit_transform(x_train)
    x_test = transformer.transform(x_test)

    return x_train, x_test, y_train, y_test

def postprun(x_train, x_test, y_train, y_test):

    clf = DecisionTreeClassifier(random_state=0, max_depth = 4)
    path = clf.cost_complexity_pruning_path(x_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities #alphas, total leaf impurities at each step
    
    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=0, max_depth = 4, ccp_alpha=ccp_alpha)
        clf.fit(x_train, y_train)
        clfs.append(clf)
    
    train_scores = [clf.score(x_train, y_train) for clf in clfs] #training accuracy of model at the specified alpha
    test_scores = [clf.score(x_test, y_test) for clf in clfs]
    
    f1train_scores = []
    f1_scores = []
    for clf in clfs:
        y_pred = clf.predict(x_test)
        y_pred_train = clf.predict(x_train)
        f1 = f1_score(y_test, y_pred, average = 'weighted')
        f1_train = f1_score(y_train, y_pred_train, average = 'weighted') 
 
        f1train_scores.append(f1_train)
        f1_scores.append(f1)

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    plt.show()
    plt.savefig("Accuracy vs alpha.png")

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("f1 score")
    ax.set_title("F1 Score vs alpha for training and testing sets")
    ax.plot(ccp_alphas, f1train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, f1_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    plt.show()
    plt.savefig("F1 Score vs alpha.png")

    return ccp_alphas, test_scores, f1_scores

def main():
    df = pd.read_csv('abalone.data', header=None, names=["sex", "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight", "rings"])
    df = process_data(df)
    print(df)

    x_train, x_test, y_train, y_test = prepare(df, 10)
    ccp_alphas, test_scores, f1_scores = postprun(x_train, x_test, y_train, y_test)
    index = np.argmax(test_scores)
    index2 = np.argmax(f1_scores) 
    optimal_ccp = ccp_alphas[index]
    optimal_ccp2 = ccp_alphas[index2]
    
    print(round(optimal_ccp, 5))
    print(round(optimal_ccp2, 5))

    acc_all = np.zeros(10)
    acc_train_all = np.zeros(10)
    f1_all = np.zeros(10)
    f1_train_all = np.zeros(10)
    for i in range(10):
        x_train, x_test, y_train, y_test = prepare(df, i)
        model = DecisionTreeClassifier(random_state = 0, max_depth = 6, ccp_alpha = 0.00133)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        y_pred_train = model.predict(x_train)

        acc = accuracy_score(y_test, y_pred)
        acc_train = accuracy_score(y_train, y_pred_train)

        f1 = f1_score(y_test, y_pred, average = 'weighted')
        f1_train = f1_score(y_train, y_pred_train, average = 'weighted')        
            
        acc_all[i] = acc
        acc_train_all[i] = acc_train
        f1_all[i] = f1
        f1_train_all[i] = f1_train

    print(round(acc_all.mean(), 4), 'accmeanp')
    print(round(acc_all.std(), 4), 'accstdp')
    print(round(acc_train_all.mean(), 4), 'acc_trainmeanp')
    print(round(acc_train_all.std(), 4), 'acc_trainstdp')
    print(round(f1_all.mean(), 4), 'f1meanp')
    print(round(f1_all.std(), 4), 'f1stdp') 
    print(round(f1_train_all.mean(), 4), 'f1_trainmeanp')
    print(round(f1_train_all.std(), 4), 'f1_trainstdp')

if __name__ == '__main__':
    main()

