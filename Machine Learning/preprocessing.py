
##################################################################

# 해당 소스파일의 저작권은 없습니다.
# 필요하신 분들은 언제나 사용하시길 바라며, 해당 소스코드의 부족한 부분에 대해서는
# whcl303@hanmail.net으로 언제든지 피드백 주시길 바랍니다.
# 소스설명 : 데이터 전처리를 하는 소스로 Feature Scaling, 문자열 처리 기능을 포함하고 있습니다.
# 포함함수 : train_test_split - 트레이닝, 테스트 셋 분리

##################################################################


# rawdata.dropna(subset [""])
# rawdata[""].fillna(x, inplace = True)

def train_test_split(x, y, suffleYN, testSize):

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=testSize, shuffle=suffleYN, random_state=1004)
    return X_train, X_test, y_train, y_test