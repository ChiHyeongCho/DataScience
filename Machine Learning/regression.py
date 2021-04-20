
# 선형 회귀

#1#
# def regression(rawdata, y, x):

#    import statsmodels.formula.api as smf

#    model = smf.ols(formula='y ~ x[:]', data=rawdata)
#    result = model.fit()
#    result.summary()



#2#

# from sklearn.linear_model import LinearRegression
# import pandas as pd

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


# 로지스틱 회귀

#1#
# def logisticregression(rawdata, y, x):

#    import statsmodels.api as sm

#    model = sm.Logit(rawData['Occupancy'], rawData[["Temperature", "Humidity", "Light", "CO2"]])
#    result = model.fit()
#    print(result.summary())

#2#
#from sklearn.linear_model import LogisticRegression
#from sklearn import metrics

#line_fitter = LogisticRegression()
#line_fitter.fit(X_train, Y_train)

#print(line_fitter.coef_)
#print(line_fitter.intercept_)

#plt.plot(X, Y, 'o')
#plt.plot(Y_test, line_fitter.predict(X_test))
#print(metrics.accuracy_score(Y_test, line_fitter.predict(X_test)))
#plt.show()