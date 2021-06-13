# KNN Classifier

#################################################################

# 해당 소스파일의 저작권은 없습니다.
# 필요하신 분들은 언제나 사용하시길 바라며, 해당 소스코드의 부족한 부분에 대해서는
# whcl303@hanmail.net으로 언제든지 피드백 주시길 바랍니다:)
# 소스설명 : 데이터분석(EDA 등) 후, KNN 알고리즘을 이용한 예측 함수 입니다.
# 1. statsmodel 활용 2. sklearn

##################################################################


# knn_regression

def KNN_reg(n_neighbors_num, x, y):
    from sklearn import neighbors

    reg = neighbors.KNeighborsRegressor(n_neighbors = n_neighbors_num)
    reg.fit(x, y)

    return reg

#################################################

# knn_classfication

def KNN_clf(n_neighbors_num, x, y):
    from sklearn import neighbors

    clf = neighbors.KNeighborsClassifier(n_neighbors = n_neighbors_num)
    clf.fit(x, y)

    return clf
