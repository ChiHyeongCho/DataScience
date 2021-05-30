
# 데이터 분석에 필요한 라이브러리

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from  matplotlib import rc

# Jupyter notebook 사용시 그래프 출력을 위한 코드
# %matplotlib

# 그래프를 격자 스타일로
plt.style.use("ggplot")

# 그래프에서 마이너스 폰트 깨지는 문제 해결
mpl.rcParams["axes.unicode_minus"] = False

# Null인 데이터 시각화 하기

import miassingno as mano
mano.matrix(train, figsize = (12, 5))