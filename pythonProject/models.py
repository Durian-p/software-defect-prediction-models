from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Conv1D, Flatten, MaxPool2D
import matplotlib.pyplot as plt
import numpy as np


# 随机森林
def random_forest(original_data, original_X, original_Y, combined_training_data, x_train, x_test, y_train, y_test):
    # 初始化模型，n_estimators:决策树的数量,max_depth：树的最大深度,random_state：随机数种子
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
    # 对训练集进行拟合
    clf.fit(x_train, y_train.values.ravel())
    return clf


# 支持向量机
def svm(original_data, original_X, original_Y, combined_training_data, x_train, x_test, y_train, y_test):
    # 核函数使用默认函数，即特征值平均
    clf = SVC(gamma='auto')
    clf.fit(x_train, y_train.values.ravel())
    return clf


# 决策树模型
def decision_tree(original_data, original_X, original_Y, combined_training_data, x_train, x_test, x_val, y_train, y_test,y_val):
    clf = tree.DecisionTreeClassifier(max_depth=5)
    clf.fit(x_train, y_train.values.ravel())
    return clf


# 朴素贝叶斯算法
def nb(original_data, original_X, original_Y, combined_training_data, x_train, x_test, y_train, y_test):
    clf = MultinomialNB()
    # MultinomialNB要求训练集中不出现负值,将训练数据归一化，使用preprocessing.MinMaxScaler来处理。
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    X_train_scaled = scaler.transform(x_train)
    clf.fit(X_train_scaled, y_train.values.ravel())
    # clf.fit(x_train, y_train.values.ravel())
    return clf


# KNN
def knn(original_data, original_X, original_Y,combined_training_data,x_train1,x_train2,x_train,x_test,x_val,y_train1,y_train2,y_train,y_test,y_val):
    clf = KNeighborsClassifier(n_neighbors=4)
    clf.fit(x_train, y_train.values.ravel())
    return clf

# Adaboost
def ada(original_data, original_X, original_Y,combined_training_data,x_train1,x_train2,x_train,x_test,x_val,y_train1,y_train2,y_train,y_test,y_val):
    clf = AdaBoostClassifier()
    clf.fit(x_train, y_train.values.ravel())
