
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

data = read_data.read_csv("C:/Users/whcl3/PycharmProjects/DataScience/Kaggle/kobe/Input/data.csv")

# datatype -> category, object (메모리 절약 이점)

data['action_type'] = data['action_type'].astype('object')
data['combined_shot_type'] = data['combined_shot_type'].astype('category')
data['game_event_id'] = data['game_event_id'].astype('category')
data['game_id'] = data['game_id'].astype('category')
data['period'] = data['period'].astype('object')
data['playoffs'] = data['playoffs'].astype('category')
data['season'] = data['season'].astype('category')
data['shot_made_flag'] = data['shot_made_flag'].astype('category')
data['shot_type'] = data['shot_type'].astype('category')
data['team_id'] = data['team_id'].astype('category')


data.set_index("shot_id", inplace = True)

# print(data.describe(include = ['number']))
# print(data.describe(include = ['category', 'object']))

train = data.dropna(how = 'any')

def bar_chart(train, feature, ax):
    success = train[train['shot_made_flag']==1][feature].value_counts()
    fail = train[train['shot_made_flag']==0][feature].value_counts()
    df = pd.DataFrame([success, fail])
    df.index = ['Success', 'Fail']
    df.plot(kind='bar', stacked=True, ax=ax)

bar_chart(train, 'shot_made_flag', plt.axes())
print(plt.show())

print(train['shot_made_flag'].value_counts()/len(train.index))

# sns.pairplot(train, vars = ['loc_x', 'loc_y', 'lat', 'lon', 'shot_distance'], hue = "shot_made_flag", size=3)
# print(plt.show())

def count_plot(column, ax):
    sns.countplot(x= column, hue = "shot_made_flag", data = train, ax = ax)

f, axrr = plt.subplots(8, figsize = (15, 30))

categorical_data = ["combined_shot_type", "season","period", "playoffs", "shot_type", "shot_zone_area",
                    "shot_zone_basic", "shot_zone_range"]

for idx, category_data in enumerate(categorical_data, 0):
    count_plot(category_data, axrr[idx])
    axrr[idx].set_title(category_data)

plt.tight_layout()
plt.show()

def print_probability(colum):
    print(train[train["shot_made_flag"] == 1][colum].value_counts() / (train[train["shot_made_flag"] == 1][colum].value_counts() + train[train["shot_made_flag"] == 0][colum].value_counts()))

for category_data in categorical_data:
    print_probability(category_data)

def draw_facetgrid(feature):
    facet = sns.FacetGrid(train, hue = "shot_made_flag", aspect=5)
    facet.map(sns.kdeplot, feature, shade = True)
    facet.set(xlim = (0, train[feature].max()))
    facet.add_legend()
    plt.show()

draw_facetgrid('minutes_remaining')
draw_facetgrid('seconds_remaining')

train['shot_made_flag'] = train["shot_made_flag"].astype('int64')
print(train.groupby(['season', 'combined_shot_type'])['shot_made_flag'].sum()/train.groupby(['season', 'combined_shot_type'])['shot_made_flag'].count())

data_cp = data.copy()
target = data_cp['shot_made_flag'].copy()

data_cp.drop('team_id', axis = 1, inplace =True)
data_cp.drop('team_name', axis = 1, inplace =True)
data_cp.drop('lat', axis = 1, inplace =True)
data_cp.drop('lon', axis = 1, inplace =True)
data_cp.drop('game_id', axis = 1, inplace =True)
data_cp.drop('game_event_id', axis = 1, inplace =True)
data_cp.drop('shot_made_flag', axis = 1, inplace =True)

data_cp['seconds_from_period_end'] = 60 * data_cp['minutes_remaining'] + data_cp['seconds_remaining']

data_cp['last_5_sec_in_period'] = data_cp['seconds_from_period_end'] < 5

data_cp.drop('seconds_from_period_end', axis = 1, inplace =True)
data_cp.drop('minutes_remaining', axis = 1, inplace =True)
data_cp.drop('seconds_remaining', axis = 1, inplace =True)

data_cp["home_away"] = data_cp['matchup'].str.contains('vs').astype('int')
data_cp.drop('matchup', axis = 1, inplace =True)

data_cp['game_date'] = pd.to_datetime(data_cp['game_date'])
data_cp['game_year'] = data_cp['game_date'].dt.year
data_cp['game_month'] = data_cp['game_date'].dt.month

data_cp.drop('game_date', axis = 1, inplace =True)

# loc_X, loc_y

data_cp['loc_x'] = pd.cut(data_cp['loc_x'], 25)
data_cp['loc_y'] = pd.cut(data_cp['loc_y'], 25)

rare_action_types = data_cp['action_type'].value_counts().sort_values().index.values[:20]
data_cp.loc[data_cp['action_type'].isin(rare_action_types), 'action_type'] = 'Other'

categorical_col = ['action_type', 'combined_shot_type', 'period', 'season', 'shot_type',
                   'shot_zone_area', 'shot_zone_basic', 'shot_zone_range', 'game_year',
                   'game_month', 'opponent', 'loc_x', 'loc_y' ]

for column in categorical_col:
    dummies = pd.get_dummies(data_cp[column])
    dummies = dummies.add_prefix("{}#".format(column))
    data_cp.drop(column, axis = 1, inplace = True)
    data_cp = data_cp.join(dummies)

unknown_mask = data["shot_made_flag"].isnull()

Y = target[-unknown_mask]
X = data_cp[-unknown_mask]

from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

threshold = 0.90
vt = VarianceThreshold().fit(X)
# Find feature names
feat_var_threshold = data_cp.columns[vt.variances_ > threshold * (1-threshold)]
print(feat_var_threshold)

model = RandomForestClassifier()
model.fit(X, Y)
feature_imp = pd.DataFrame(model.feature_importances_, index=X.columns, columns=["importance"])
feat_imp_20 = feature_imp.sort_values("importance", ascending=False).head(20).index
print(feat_imp_20)

featrues = np.hstack([feat_var_threshold, feat_imp_20])
featrues = np.unique(featrues)

print(featrues)

print(X.shape)

components = 8
pca = PCA(n_components=components).fit(X)
pca_variance_explained_df= pd.DataFrame(
    {     "component": np.arange(1, components+1),     "variance_explained": pca.explained_variance_ratio_                 })
ax = sns.barplot(x='component', y='variance_explained', data=pca_variance_explained_df)
ax.set_title("PCA - Variance explained")

plt.show()


