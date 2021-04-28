
##################################################################

# 해당 소스파일의 저작권은 없습니다.
# 필요하신 분들은 언제나 사용하시길 바라며, 해당 소스코드의 부족한 부분에 대해서는
# whcl303@hanmail.net으로 언제든지 피드백 주시길 바랍니다:)
# 소스설명 : 데이터분석(EDA 등) 후, 선형회귀를 이용한 예측 함수 입니다.
# 1. statsmodel 활용 2. sklearn

##################################################################

# 선형 회귀

#1
def regression_statsmodels(rawdata, x, y):

    import statsmodels.formula.api as smf

    model = smf.ols(formula='y ~ x[:]', data=rawdata)
    result = model.fit()
    result.summary()



#2 (일반적인 경우 sklearn 사용)

def regression(x, y):

    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(x, y)

    print(model.coef_)
    print(model.intercept_)

    return model