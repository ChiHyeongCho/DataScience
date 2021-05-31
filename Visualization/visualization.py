
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
def missingno_graph(train):
    import missingno as mano
    mano.matrix(train, figsize = (12, 5))

def bar_chart(train, feature, ax):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked=True, ax=ax)


def draw_facetgrid(train, feature):
    # train에 저장된 DataFrame을 FacetGrid를 통해 그래프로 그려줍니다.
    # hue="Survived"는 그래프의 범례(legend)의 이름을 설정합니다.
    # aspect=5 는 그래프의 종횡비를 설정해줍니다.
    facet = sns.FacetGrid(train, hue="Survived", aspect=5)

    # facet.map()은 kedplot 방식을 사용하여 주어진 데이터 feature를 plotting 하는
    # 즉, 그래프를 그리는 기능을 합니다.
    facet.map(sns.kdeplot, feature, shade=True)
    # 0 부터 값의 주어진 데이터의 최대 값까지를 x축의 범위로 설정합니다.
    facet.set(xlim=(0, train[feature].max()))
    # 지정된 범례(legend)를 표시.
    facet.add_legend()


def main():
    train = pd.read_csv('C:/Users/whcl3/PycharmProjects/DataScience/Visualization/train.csv')
    missingno_graph(train)

    mpl.rc('font', family='NanumGothic')
    figure, ((ax1,ax2,ax3), (ax4,ax5,ax6)) = plt.subplots(nrows=2, ncols=3)
    figure.set_size_inches(18, 12)
    bar_chart(train, 'Sex', ax1)
    bar_chart(train, 'Pclass', ax2)
    bar_chart(train, 'SibSp', ax3)
    bar_chart(train, 'Parch', ax4)
    bar_chart(train, 'Embarked', ax5)
    ax1.set(title="성별 생사정보")
    ax2.set(title="티켓 class")
    ax3.set(title="형제 수")
    ax4.set(title="부모 자식의 수")
    ax5.set(title="승선 장소")

    draw_facetgrid(train, "Age")
    draw_facetgrid(train, "Fare")

    plt.show()

main()