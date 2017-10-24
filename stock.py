"""
file stock.py
author: Petri Lamminaho

"""

import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
df = quandl.get("WIKI/TSLA")
#df_rovio = quandl.get(("WIKI/ROVI"))
#df = quandl.get("SSE/NOA3", authtoken="oPEXdFiDhzkwWTQN9wVv")

#df = df[['Adj. Open', 'Adj. High',  'Adj. Low', 'Adj. Close', 'Adj. Volume']]

#HL_PCT = High low percent. High ja low:n suhde
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) /  df['Adj. Close'] * 100
#Päivän muutos prosenteina
df['Daily_Change'] = (df['Adj. Close'] - df['Adj. Open']) /  df['Adj. Open'] * 100

df = df[['Adj. Close', 'HL_PCT', 'Daily_Change', 'Adj. Volume']]
print(df.tail(1))
forecast_col  = 'Adj. Close'
df.fillna(-99999, inplace=True) #if value is missing value is: -99999
forecast_out = int(math.ceil(0.01 * len(df))) #timescale is 34 days
#print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
X = np.array(df.drop(['label'],1))
y = np.array(df['label'])
x = preprocessing.scale(X) # ei ehkä tarvita
y = np.array(df['label'])
X_Lately = X[-forecast_out:]
#X = X[:-forecast_out:]


#print(len(X), len(y))

x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
clf = LinearRegression()
clf.fit(x_train, y_train)# training the classifier
accuracy = clf.score(x_test, y_test)
#print(accuracy)

forecast_set = clf.predict(X_Lately)


print('Forecast for',forecast_out,"days.")
print("Accuracy is:",accuracy*100,"%")
index = 1
for  forecast in forecast_set:
    print("Day",index, ":", forecast)
    index+=1

#print(forecast_set[-1])

