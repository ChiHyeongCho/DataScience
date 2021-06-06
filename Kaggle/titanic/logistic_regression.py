
# 로지스틱 회귀

#################################################################

# 해당 소스파일의 저작권은 없습니다.
# 필요하신 분들은 언제나 사용하시길 바라며, 해당 소스코드의 부족한 부분에 대해서는
# whcl303@hanmail.net으로 언제든지 피드백 주시길 바랍니다:)
# 소스설명 : 데이터분석(EDA 등) 후, 선형 분류회귀를 이용한 예측 함수 입니다.
# 1. statsmodel 활용 2. sklearn

##################################################################


#1#
def regression_statsmodels(rawdata, x, y):

    import statsmodels.api as sm

    model = sm.Logit(y, x)
    result = model.fit()
    print(result.summary())

#2#
def regression(x, y):

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.fit(x, y)

    #print(model.coef_)
    #print(model.intercept_)
    #print(model.score(x, y))

    return model
