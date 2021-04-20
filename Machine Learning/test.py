
import read as rd

import pandas as pd

readAddress = 'C:/Users/whcl3/Desktop/학교/4학년 2학기/인공지능론/term project/dataset.csv'

rawData = rd.read_csv(readAddress)
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)

# print(EDA.head(rawData))
# EDA.info(rawData)
# EDA.value_counts(rawData)
# EDA.describe(rawData)
# corr = EDA.correlation(rawData)

# print(corr)



# rawData.plot(kind = "scatter", x= "Temperature", y ="Humidity", alpha = 0.1, label = "HumidityRatio", c = "CO2",  cmap = plt.get_cmap("jet"), colorbar = True)
# plt.show()


# attributes = ["Temperature", "Light" , "Humidity" ]
# scatter_matrix(rawData[attributes])
# plt.show()
# print(EDA.VIF(rawData[["Temperature", "Humidity", "Light", "CO2", "HumidityRatio", "Occupancy"]]))

# import statsmodels.formula.api as smf

# model = smf.ols(formula='Occupancy ~ HumidityRatio + Temperature', data = rawData)
# result = model.fit()
# print(result.summary())
# print(y_new)

# import statsmodels.api as sm

# model = sm.Logit(rawData['Occupancy'], rawData[["Temperature", "Humidity", "Light", "CO2"]])
# result = model.fit()

# model.predict()

# print(y_new)
# print(result.summary())


X = rawData[["Temperature", "Humidity", "Light", "CO2"]]
Y = rawData[['Occupancy']]
print(Y.mean(), Y.median())

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, shuffle=False, random_state=1004)

# from sklearn.linear_model import LinearRegression
# import pandas as pd
# import numpy as np

# from sklearn.model_selection import train_test_split
# from sklearn import metrics
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, shuffle=False, random_state=1004)

# line_fitter = LinearRegression()
# line_fitter.fit(X_train, Y_train)

# print(line_fitter.coef_)
# print(line_fitter.intercept_)

# plt.plot(X, Y, 'o')
# plt.plot(Y_test, line_fitter.predict(X_test))
# print(line_fitter.score(X_test, Y_test))
# plt.show()

# from sklearn.linear_model import LogisticRegression
# from sklearn import metrics

# line_fitter = LogisticRegression()
# line_fitter.fit(X_train, Y_train)

# print(line_fitter.coef_)
# print(line_fitter.intercept_)

# plt.plot(X, Y, 'o')
# plt.plot(Y_test, line_fitter.predict(X_test))
# print(metrics.accuracy_score(Y_test, line_fitter.predict(X_test)))
# plt.show()

# dnn.dnn(X_train, X_test, Y_train, Y_test)

# lstm.lstm(X_train, X_test, Y_train, Y_test)

import gradientboosting

gradientboosting.xgb(X_train, X_test, Y_train, Y_test)
