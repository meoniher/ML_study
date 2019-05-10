#!/usr/bin/python3
#-*-coding:utf-8 -*-
# __Source__  : task from suixin
# __Author__  : chenjiahui from internet
# __File__    : LogisticRegression.py
# __Software__: PyCharm
'''
    逻辑回归的实现及注意事项:
    1、要求label服从伯努利分布,属于广义线性模型中的一种
    2、使用多种优化方法实现逻辑回归(二分类 and 多分类)，使用以下数据集
    def getData():
        from sklearn.datasets import make_classification
        X1, y1 = make_classification(n_samples=200,n_features=10,n_classes=2,n_clusters_per_class=1,random_state=0)
        X2, y2 = make_classification(n_samples=200, n_features=10, n_classes=3, n_clusters_per_class=1, random_state=0)
        return X1,y1,X2,y2
'''

def accuracy_score(y_true, y_predict):
    '''
        计算y_true和y_predict之间的准确率
    '''
    assert len(y_true) == len(y_predict)
    return np.sum(y_true == y_predict) / len(y_true)

def train_test_split(X, y, test_ratio=0.2, seed=None):
    ''' 
        切分原始数据
    '''
    assert X.shape[0] == y.shape[0]
    assert 0.0 <= test_ratio <= 1.0
    if seed:
        np.random.seed(seed)
    shuffled_indexes = np.random.permutation(len(X))    
    test_size = int(len(X) * test_ratio)
    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]
    X_train = X[train_indexes]
    y_train = y[train_indexes]
    X_test = X[test_indexes]
    y_test = y[test_indexes]
    return X_train, X_test, y_train, y_test

class LogisticRegression:
    '''
        逻辑回归
    '''
    #初始化模型
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self._theta = None
        
    #sigmod函数
    def _sigmod(self, t):
        return 1. / (1. + np.exp(-t))
    
    #梯度下降训练训练数据
    def fit(self, X_train, y_train, eta=0.01, n_iters=1e4):
        assert X_train.shape[0] == y_train.shape[0]

        def J(theta, X_b, y):
            y_hat = self._sigmod(X_b.dot(theta))
            try:
                return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / len(y)
            except:
                return float("inf")

        def dJ(theta, X_b, y):
            return X_b.T.dot(self._sigmod(X_b.dot(theta)) - y) / len(X_b)

        def gradient_descent(X_b, y, inital_theta, eta, n_inters=1e4, epsilon=1e-8):
            theta = initial_theta
            i_inter = 0

            while i_inter < n_inters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient

                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break

                i_inter += 1
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    
    #预测的概率函数
    def predict_proba(self, X_predict):
        assert self.intercept_ is not None and self.coef_ is not None
        assert X_predict.shape[1] == len(self.coef_)
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return self._sigmod(X_b.dot(self._theta))

    #预测结果的函数
    def predict(self, X_predict):
        assert self.intercept_ is not None and self.coef_ is not None
        assert X_predict.shape[1] == len(self.coef_)
        proba = self.predict_proba(X_predict)
        return np.array(proba >= 0.5, dtype='int')
    
    #准确率
    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        
        return "LogisticRegression()"


import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=200, n_features=10, n_classes=2, n_clusters_per_class=1, random_state=0)
print(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
print(log_reg.predict(X_test))
print(log_reg.score(X_test, y_test))
