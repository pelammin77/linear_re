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
df = quandl.get("WIKI/GOOGL")



#df = df[['Adj. Open', 'Adj. High',  'Adj. Low', 'Adj. Close', 'Adj. Volume']]

#HL_PCT = High low percent. High ja low:n suhde
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) /  df['Adj. Close'] * 100
#Päivän muutos prosenteina
df['Daily_Change'] = (df['Adj. Close'] - df['Adj. Open']) /  df['Adj. Open'] * 100

df = df[['Adj. Close', 'HL_PCT', 'Daily_Change', 'Adj. Volume']]
forecast_col  = 'Adj. Close'
df.fillna(-99999, inplace=True) #jos arvo puutuu niin sijoitetaan -99999
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df.tail())