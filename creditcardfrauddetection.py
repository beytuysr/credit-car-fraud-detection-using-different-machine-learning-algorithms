# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 20:36:28 2020
credit card fraud detection
@author: Beytu
"""

#1.import libraries
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow.keras
import keras
import seaborn as sns
import random

np.random.seed(2)

#2. import data
data=pd.read_csv('creditcard.csv')

#3.data exploration and preprocessing
dh=data.head()
dt=data.tail()

from sklearn.preprocessing import StandardScaler
data['normalizedAmount']=StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data=data.drop(['Amount'],axis=1)
data=data.drop(['Time'],axis=1)
dhn=data.head()
dtn=data.tail()


X=data.iloc[:,data.columns!='Class']
xh=X.head()
y=data.iloc[:,data.columns=='Class']
yh=y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3, random_state=0)
xts=X_test.shape

X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)

#4.deep neural network
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

model=Sequential()
model.add(Dense(units=16,input_dim=29,activation='relu'))
model.add(Dense(units=24,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(20,activation='relu'))
model.add(Dense(24,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


ms=model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=5,batch_size=15,verbose=1,validation_split=0.2)
y_pred=model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred.round())
print('deep learning confusion matrix')
print(cm)
sns.heatmap(cm,annot=True)
plt.show()

#5.random forest
from sklearn.ensemble import RandomForestClassifier
random_forest=RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train,y_train.ravel())
y_pred=random_forest.predict(X_test)

score=random_forest.score(X_test,y_test)
print('random forest score')
print(score)
cm=confusion_matrix(y_test,y_pred.round())
print('Random forest confusion matrix')
print(cm)
sns.heatmap(cm,annot=True)
plt.show()

#6.Decision tree
from sklearn.tree import DecisionTreeClassifier
decision_tree=DecisionTreeClassifier()
decision_tree.fit(X_train,y_train.ravel())
y_pred=decision_tree.predict(X_test)
decisionscore=decision_tree.score(X_test,y_test)
print('decison tree score')
print(decisionscore)
cm=confusion_matrix(y_test,y_pred.round())
print('decision tree confusion matrix')
print(cm)
sns.heatmap(cm,annot=True)
plt.show()

#7.undersampling
fraud_indices=np.array(data[data.Class==1].index)
number_records_fraud=len(fraud_indices)
print('number of records fraud')
print(number_records_fraud)

normal_indices=np.array(data[data.Class==0].index)
random_normal_indices=np.random.choice(normal_indices,number_records_fraud,replace=False)
random_normal_indices=np.array(random_normal_indices)
print('random mormal indices')
print(len(random_normal_indices))

under_sample_indices=np.concatenate([fraud_indices,random_normal_indices])
print('under sample indices')
print(len(under_sample_indices))

under_sample_data=data.iloc[under_sample_indices,:]
X_undersample=under_sample_data.iloc[:,under_sample_data.columns!='Class']
y_undersample=under_sample_data.iloc[:,under_sample_data.columns=='Class']
X_train, X_test, y_train, y_test=train_test_split(X_undersample,y_undersample,test_size=0.3)

X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)

model.sum=model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=5,batch_size=15,verbose=1,validation_split=0.2)
y_pred=model.predict(X_test)
y_expected=pd.DataFrame(y_test)

cm=confusion_matrix(y_expected,y_pred.round())
print('under sampling confusion matrix')
print(cm)
sns.heatmap(cm,annot=True)
plt.show()

#8.smote
from imblearn.over_sampling import SMOTE
X_resample, y_resample = SMOTE().fit_sample(X,y.values.ravel())
y_resample = pd.DataFrame(y_resample)
X_resample = pd.DataFrame(X_resample)
X_train, X_test, y_train, y_test = train_test_split(X_resample,y_resample,test_size=0.3)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=5,batch_size=15,verbose=1,validation_split=0.2)
y_pred=model.predict(X_test)
y_expected=pd.DataFrame(y_test)

cm=confusion_matrix(y_expected,y_pred.round())
print('under sampling confusion matrix')
print(cm)
sns.heatmap(cm,annot=True)
plt.show()
