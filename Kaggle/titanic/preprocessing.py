
##################################################################

# 해당 소스파일의 저작권은 없습니다.
# 필요하신 분들은 언제나 사용하시길 바라며, 해당 소스코드의 부족한 부분에 대해서는
# whcl303@hanmail.net으로 언제든지 피드백 주시길 바랍니다.
# 소스설명 : 데이터 전처리를 하는 소스로 Feature Scaling, 문자열 처리 기능을 포함하고 있습니다.
# 포함함수 : train_test_split - 트레이닝, 테스트 셋 분리

##################################################################


# rawdata.dropna(subset [""])
# rawdata[""].fillna(x, inplace = True)

def preprocessing(raw_data):

    import copy

    import pandas as pd
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy as stats
    from matplotlib import rc
    import missingno as mano

    # 데이터 처리 시작

    plt.style.use("ggplot")
    mpl.rcParams["axes.unicode_minus"] = False


    # 콘솔창 출력 column 확대
    pd.set_option('display.max_columns', None)

    data = copy.deepcopy(raw_data)

    # 2번째 column Name -> Title

    data["Title"] = data['Name'].str.extract('([A-Za-z]+)\.', expand=False)

    # 불필요한 Column 제거

    # axis = 1 이면 column 삭제, 0이면 Row 삭제
    data.drop('Name', axis=1, inplace=True)

    # 데이터 Mapping
    title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 0, "Dr": 3, "Rev": 3,
                     "Col": 3, "Major": 3, "Mlle": 3, "Countess": 3, "Ms": 3, "Lady": 3,
                     "Jonkheer": 3, "Don": 3, "Dona": 3, "Mme": 3, "Capt": 3, "Sir": 0}

    sex_mapping = {"male": 0, "female": 1}

    data["Title"] = data["Title"].map(title_mapping)
    data["Sex"] = data["Sex"].map(sex_mapping)

    # 데이터 Null값 제거
    data["Age"].fillna(data.groupby("Title")["Age"].transform("median"), inplace=True)

    # binning 작업 (age)

    data["AgeBand"] = pd.cut(data["Age"], 5)

    data.loc[data["Age"] <= 16, "Age"] = 0
    data.loc[(data["Age"] > 16) & (data["Age"] <= 32), "Age"] = 1
    data.loc[(data["Age"] > 32) & (data["Age"] <= 48), "Age"] = 2
    data.loc[(data["Age"] > 48) & (data["Age"] <= 64), "Age"] = 3
    data.loc[(data["Age"] > 64), "Age"] = 4

    # Embarked

    embarked_mapping = {"S": 0, "Q": 1, "C": 2}

    data['Embarked'] = data['Embarked'].fillna('S')
    data["Embarked"] = data["Embarked"].map(embarked_mapping)

    # Fare

    data["Fare"].fillna(data.groupby("Pclass")["Fare"].transform("median"), inplace=True)
    # test["FareBand"] = pd.cut(test["Fare"], 5)

    # print(train[["FareBand", "Survived"]].groupby("FareBand", as_index = False).mean().sort_values(by = "FareBand"))

    data.loc[data["Fare"] <= 102, "Fare"] = 0
    data.loc[(data["Fare"] > 102) & (data["Fare"] <= 204), "Fare"] = 1
    data.loc[(data["Fare"] > 204) & (data["Fare"] <= 307), "Fare"] = 2
    data.loc[data["Fare"] > 307, "Fare"] = 3

    data.drop("AgeBand", axis=1, inplace=True)

    # Cabin & Ticket
    data.drop("Cabin", axis=1, inplace=True)

    data.drop("Ticket", axis=1, inplace=True)

    # family Size 생성

    data["FamilySize"] = data["SibSp"] + data['Parch'] + 1

    # PassengerId 삭제
    data.drop("PassengerId", axis=1, inplace = True)

    return data


def train_test_split(x, y, suffleYN, testSize):

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=testSize, shuffle=suffleYN, random_state=1004)
    return X_train, X_test, y_train, y_test