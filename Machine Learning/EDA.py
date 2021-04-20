
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




