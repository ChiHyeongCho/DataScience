
##################################################################

# 해당 소스파일의 저작권은 없습니다.
# 필요하신 분들은 언제나 사용하시길 바라며, 해당 소스코드의 부족한 부분에 대해서는
# whcl303@hanmail.net으로 언제든지 피드백 주시길 바랍니다:)
# 소스설명 : 다양한 차원과 값을 조합해가며 특이한 점이나 의미 있는 사실을 도출하고
# 분석의 최종 목적을 달성해가는 과정으로, 데이터의 특징과 구조적 관계를 알아내기 위한 기법들

##################################################################


def head(rawdata):

    return (rawdata.head())

    return

def info(rawdata):

    return (rawdata.info())



def value_counts(rawdata):

    return (rawdata.value_counts())



def describe(rawdata):

    return (rawdata.describe())



def correlation(rawdata):

    return (rawdata.corr())

def VIF(xdata):

    import pandas as pd

    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(xdata.values, i) for i in range(xdata.shape[1])]
    vif["features"] = xdata.columns

    return vif




