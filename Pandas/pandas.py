
# Pandas 사용법에 관련한 코드입니다.


import pandas as pd

address = " "

train_data = pd.read_csv(address)

# 데이터 모양 확인하기
print(train_data.shape)

# 앞의 n개 데이터만 확인해보기
n = 5
print(train_data.head(n))

# data의 datatype, null 여부 확인하기
print(train_data.info())

# data의 null 값 Check
print(train_data.isnull().sum())
