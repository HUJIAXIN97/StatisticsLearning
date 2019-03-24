#!/usr/bin/env python
# coding: utf-8

#Hu Jiaxin

# This is for prostate data analysis. 
# The data is from the book The Elements of Statistic Learning
# which is https://web.stanford.edu/~hastie/ElemStatLearn/index.html

# I used three models to fit the regression: ElasticNet, Lasso and Ridge. 


#import the data
import pandas as pd
import numpy as np
df = pd.read_csv('prostate.csv',header=None,sep=',', names=["col",'lcavol','lweight','age','lbph','svi','lcp','gleason','pgg45','lpsa','train'])

#check if the data is correct
#pandas will show the first 5 and the last 5 rows as default
print (df.head())
print (df.tail())

#split the 67 train data and the 30 test data
train = df.loc[df['train']=='T']
test = df.loc[df['train']=='F']
#check
print(train.shape)
print(test.shape)

X_train = train[['lcavol','lweight','age','lbph','svi','lcp','gleason','pgg45']]
Y_train = train[['lpsa']]
print(X_train.shape)
print(Y_train.shape)

X_test = test[['lcavol','lweight','age','lbph','svi','lcp','gleason','pgg45']]
Y_test = test[['lpsa']]
print(X_test.shape)
print(Y_test.shape)

X = df[['lcavol','lweight','age','lbph','svi','lcp','gleason','pgg45']]
Y = df[['lpsa']]
print(X.shape)
print(Y.shape)

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1, test_size=0.2)

model_1 = ElasticNet()

model_1.fit(X_train, Y_train)

Y_test_predict_1 = model_1.predict(X_test)
MSE_test_1 = mean_squared_error(Y_test_predict_1, Y_test)

plt.scatter(Y_test, Y_test_predict_1)
plt.plot( [0,4],[0,4] )
plt.title("test data : MSE is %s"%(MSE_test_1))
plt.xlabel('Y value in testing data')
plt.ylabel('Y value of prediction')
plt.text(0.5, 3, 'Model 1: ElesticNet')

Y_train_predict_1 = model_1.predict(X_train)
MSE_train_1 = mean_squared_error(Y_train_predict_1, Y_train)

plt.scatter(Y_train, Y_train_predict_1)
plt.plot( [0,6],[0,6] )
plt.title("train data : MSE is %s"%(MSE_train_1))
plt.xlabel('Y value in training data')
plt.ylabel('Y value of prediction')
plt.text(0.5, 4, 'Model 1: ElesticNet')

model_2 = Lasso()

#use X_train and Y_train to train the model
model_2.fit(X_train, Y_train)

#put the X_test data into the model and get the predicted Y values
Y_test_predict_2 = model_2.predict(X_test)
MSE_test_2 = mean_squared_error(Y_test_predict_2, Y_test)

plt.scatter(Y_test, Y_test_predict_2)
plt.plot( [0,4],[0,4] )
plt.title("test data : MSE is %s"%(MSE_test_2))
plt.xlabel('Y value in testing data')
plt.ylabel('Y value of prediction')
plt.text(0.5, 3, 'Model 2: Lasso')

Y_train_predict_2 = model_2.predict(X_train)
MSE_train_2 = mean_squared_error(Y_train_predict_2, Y_train)

plt.scatter(Y_train, Y_train_predict_2)
plt.plot( [0,6],[0,6] )
plt.title("train data : MSE is %s"%(MSE_train_2))
plt.xlabel('Y value in training data')
plt.ylabel('Y value of prediction')
plt.text(0.5, 5, 'Model 2: Lasso')

model_3 = Ridge()

model_3.fit(X_train, Y_train)

Y_test_predict_3 = model_3.predict(X_test)
MSE_test_3 = mean_squared_error(Y_test_predict_3, Y_test)

plt.scatter(Y_test, Y_test_predict_3)
plt.plot( [0,4],[0,4] )
plt.title("test data : MSE is %s"%(MSE_test_3))
plt.xlabel('Y value in testing data')
plt.ylabel('Y value of prediction')
plt.text(0.5, 3, 'Model 3: Ridge')

Y_train_predict_3 = model_3.predict(X_train)
MSE_train_3 = mean_squared_error(Y_train_predict_3, Y_train)

plt.scatter(Y_train, Y_train_predict_3)
plt.plot( [0,6],[0,6] )
plt.title("train data : MSE is %s"%(MSE_train_3))
plt.xlabel('Y value in training data')
plt.ylabel('Y value of prediction')
plt.text(0.5, 4, 'Model 3: Ridge')

