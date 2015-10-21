import csv as csv 
import numpy as np
import pandas as pd
import pylab as py
import matplotlib.pyplot as plt
import math
import datetime
from time import time

# Open up the csv file in to a Python object
data = pd.DataFrame.from_csv(open('/Users/dadda/Dropbox (MIT)/Kaggle Competitions/RSS/train.csv', 'rb',), index_col=None) 
data = data[data.Sales > 1]


t0 = time()

data.loc[(data.Store.isnull()), 'Store'] = 0
data.loc[(data.SchoolHoliday.isnull()), 'SchoolHoliday'] = 0
data.loc[(data.StateHoliday.isnull()), 'StateHoliday'] = 0
data.loc[(data.Promo.isnull()), 'Promo'] = 0
data['Date'] = pd.to_datetime(data['Date']).dt.month#.dtype='datetime64[ns]
data['Sales'] = np.log(data['Sales']+1)

print(data.info())

print(len(data))
print('The average sale is :' + str(data['Sales'].mean() ))
#print(data.info())

# Creates the label array
labels = data['Sales'].values

#print data.groupby('Date')['Date'].mean()

y = data.groupby('Store')['Sales'].mean()
x = data.groupby('Store')['Store'].mean()
#plt.scatter(x, y, color = 'red')
#py.show()

#data['StateHoliday2'] = data['StateHoliday'].map( {'a': 1, 'b': 2, 'c': 3, '0': 0, 0:0} ).astype(int)
#data['NewVar_1'] = np.where((data['Open'] == 0), 0, data['StateHoliday2'] * data['Open'] + 1)

data = data.drop(['StateHoliday', 'Customers', 'Sales', 'Open', 'SchoolHoliday', 'Date'], axis=1) 

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler

print(data.columns.values)

features = data.values

#features = maxmin.fit_transform(features)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.1, random_state=32)


from sklearn import neighbors, datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import grid_search, svm


#clf = LinearRegression()
clf = RandomForestRegressor(n_estimators = 20, n_jobs = -1, min_samples_split = 8 )
#clf = AdaBoostRegressor(n_estimators = 100)
#clf = GradientBoostingRegressor(n_estimators = 100)
#clf = SVR()

#parameters = {'min_samples_split':[2, 4, 6, 8, 10, 20, 50, 100]}
#svr = RandomForestRegressor(n_jobs = -1, n_estimators = 20)
#clf = grid_search.GridSearchCV(svr, parameters)
clf = clf.fit(features_train, labels_train)
#print clf.best_estimator_

print(100*(1-clf.score(features_train, labels_train)))
print('Trainingtime is: ' + str(time() - t0))
#pred = clf.predict(features_test)

index=features_test[:,0]
day = features_test[:,1]

#data2 = pd.DataFrame(dict(day=day, index=index, pred=pred))
#data2.columns = ['DayOfWeek', 'Store', 'Sales']

#y_2 = data2.groupby('Store')['Sales'].mean()
#x_2 = data2.groupby('Store')['Store'].mean()
#plt.scatter(x, y, color = 'red')
#plt.scatter(x_2, y_2, color = 'blue')
#py.show()

#y = data3.groupby('DayOfWeek')['Sales'].mean()
#x = data3.groupby('DayOfWeek')['DayOfWeek'].mean()
#y_2 = data2.groupby('DayOfWeek')['Sales'].mean()
#x_2 = data2.groupby('DayOfWeek')['DayOfWeek'].mean()
#plt.scatter(x, y, color = 'red')
#plt.scatter(x_2, y_2, color = 'blue')
#py.show()

test = pd.DataFrame.from_csv('/Users/dadda/Dropbox (MIT)/Kaggle Competitions/RSS/test.csv', index_col=None)
#test = test[test.Open == 1]

PassID = test['Id']

test.loc[(test.Store.isnull()), 'Store'] = 0
test.loc[(test.Open.isnull()), 'Open'] = 0
test.loc[(test.SchoolHoliday.isnull()), 'SchoolHoliday'] = 0
#test['StateHoliday2'] = test['StateHoliday'].map( {'a': 1, 'b': 2, 'c': 3, '0': 0, 0:0 } ).astype(int)
test['Date'] = pd.to_datetime(test['Date']).dt.dayofyear#.dtype='datetime64[ns]


# Drops useless columns
test = test.drop(['StateHoliday', 'Id', 'Open','SchoolHoliday', 'Date'], axis=1) 

features_test2 = test.values
print(test.columns.values)
pred = clf.predict(features_test2)
pred = np.exp(pred) - 1

solution = pd.DataFrame(dict(pred = pred, PassID = PassID))
solution.columns = ['Id', 'Sales']
solution.to_csv('/Users/dadda/Dropbox (MIT)/Kaggle Competitions/RSS/solution.csv', index = False)
#print(solution.tail(50))

