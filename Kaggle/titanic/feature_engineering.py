
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as stats
from matplotlib import rc
import missingno as mano

plt.style.use("ggplot")
mpl.rcParams["axes.unicode_minus"] = False

# data load

import read_data

# 콘솔창 출력 column 확대
pd.set_option('display.max_columns', None)

train = read_data.read_csv("C:/Users/whcl3/PycharmProjects/DataScience/Kaggle/titanic/train.csv")
test = read_data.read_csv("C:/Users/whcl3/PycharmProjects/DataScience/Kaggle/titanic//test.csv")

# 2번째 column Name -> Title

train_test_data = [train, test]

for dataset in train_test_data:
    dataset["Title"] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand = False)


# 불필요한 Column 제거

# axis = 1 이면 column 삭제, 0이면 Row 삭제
test.drop('Name', axis = 1, inplace = True)
train.drop('Name', axis = 1, inplace = True)


# 데이터 Mapping
title_mapping = {"Mr" : 0, "Miss" : 1, "Mrs" : 2, "Master" : 0, "Dr" : 3, "Rev" : 3,
                 "Col" : 3, "Major" : 3, "Mlle" : 3, "Countess" : 3, "Ms" : 3, "Lady" : 3,
                 "Jonkheer" : 3, "Don" : 3, "Dona" : 3, "Mme" : 3, "Capt" : 3, "Sir" : 0}


sex_mapping = {"male" : 0, "female" : 1}

for dataset in train_test_data:
    dataset["Title"] = dataset["Title"].map(title_mapping)
    dataset["Sex"] = dataset["Sex"].map(sex_mapping)


# 데이터 Null값 제거
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace = True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace = True)


# binning 작업 (age)

train["AgeBand"] = pd.cut(train["Age"], 5)
# test["AgeBand"] = pd.cut(test["Age"], 5)

for dataset in train_test_data:
    dataset.loc[dataset["Age"] <= 16, "Age"] = 0
    dataset.loc[(dataset["Age"] > 16) & (dataset["Age"] <= 32), "Age"] = 1
    dataset.loc[(dataset["Age"] > 32) & (dataset["Age"] <= 48), "Age"] = 2
    dataset.loc[(dataset["Age"] > 48) & (dataset["Age"] <= 64), "Age"] = 3
    dataset.loc[(dataset["Age"] > 64), "Age"] = 4

# Embarked

embarked_mapping = { "S" : 0, "Q" : 1, "C" : 2}

for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset["Embarked"] = dataset["Embarked"].map(embarked_mapping)


# Fare

test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace = True)
train["FareBand"] = pd.cut(train["Fare"], 5)
# test["FareBand"] = pd.cut(test["Fare"], 5)

# print(train[["FareBand", "Survived"]].groupby("FareBand", as_index = False).mean().sort_values(by = "FareBand"))

for dataset in train_test_data:
    dataset.loc[dataset["Fare"] <= 102, "Fare"] = 0
    dataset.loc[(dataset["Fare"] > 102) & (dataset["Fare"] <= 204), "Fare"] = 1
    dataset.loc[(dataset["Fare"] > 204) & (dataset["Fare"] <= 307), "Fare"] = 2
    dataset.loc[dataset["Fare"] > 307, "Fare"]  = 3

train.drop("AgeBand", axis = 1, inplace = True)
train.drop("FareBand", axis = 1, inplace = True)

# Cabin & Ticket
train.drop("Cabin", axis = 1, inplace = True)
test.drop("Cabin", axis = 1, inplace = True)

train.drop("Ticket", axis = 1, inplace = True)
test.drop("Ticket", axis = 1, inplace = True)


# family Size 생성

train["FamilySize"] = train["SibSp"] + train['Parch'] + 1
test["FamilySize"] = test["SibSp"] + test['Parch'] + 1

# PassengerId 삭제
train_data = train.drop(["Survived", "PassengerId"], axis = 1)


print(train_data.head(5))