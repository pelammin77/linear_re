"""
file stock.py
author: Petri Lamminaho

"""

import pandas as pd
import quandl, datetime
import math
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression


style.use('ggplot')
df = quandl.get("WIKI/GOOGL")
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) /  df['Adj. Close'] * 100  #HL_PCT = High low percent. High ja low:n suhde
df['Daily_Change'] = (df['Adj. Close'] - df['Adj. Open']) /  df['Adj. Open'] * 100 #Päivän muutos prosenteina
df = df[['Adj. Close', 'HL_PCT', 'Daily_Change', 'Adj. Volume']]
print(df.tail(5))
forecast_col  = 'Adj. Close'
df.fillna(-99999, inplace=True) #if value is missing value is: -99999
forecast_out = int(math.ceil(0.0037 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
X = np.array(df.drop(['label'],1))
y = np.array(df['label'])
x = preprocessing.scale(X) # ei ehkä tarvita
y = np.array(df['label'])
X_Lately = X[-forecast_out:]
#X = X[:-forecast_out:]
x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
clf = LinearRegression()
clf.fit(x_train, y_train)# training the classifier
accuracy = clf.score(x_test, y_test)
forecast_set = clf.predict(X_Lately)
print('Forecast for',forecast_out,"days.")
print("Accuracy is:",accuracy*100,"%")
index = 1

for  forecast in forecast_set:
    print("Day",index, ":", forecast)
    index+=1

df['forecast'] = np.nan

last_date = df.iloc[-1].name
print(last_date)
last_unix = last_date.timestamp()
one_day_to_seconds = 86400
next_unix = last_unix + one_day_to_seconds

for i in forecast_set:
    next_day = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day_to_seconds
    df.loc[next_day] = [np.nan for _ in range(len(df.columns)-1 )] + [i]



df['Adj. Close'].plot()
df['forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

