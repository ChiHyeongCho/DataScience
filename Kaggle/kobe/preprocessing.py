
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

    # 데이터 처리 시작

    plt.style.use("ggplot")
    mpl.rcParams["axes.unicode_minus"] = False


    # 콘솔창 출력 column 확대
    pd.set_option('display.max_columns', None)

    data = copy.deepcopy(raw_data)

    # 데이터 타입 변경
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
    data.set_index("shot_id", inplace=True)

    # Null값 삭제
    data = data.dropna(how='any')

    data['shot_made_flag'] = data["shot_made_flag"].astype('int64')


    data.drop('team_id', axis=1, inplace=True)
    data.drop('team_name', axis=1, inplace=True)
    data.drop('lat', axis=1, inplace=True)
    data.drop('lon', axis=1, inplace=True)
    data.drop('game_id', axis=1, inplace=True)
    data.drop('game_event_id', axis=1, inplace=True)

    data['seconds_from_period_end'] = 60 * data['minutes_remaining'] + data['seconds_remaining']

    data['last_5_sec_in_period'] = data['seconds_from_period_end'] < 5

    data.drop('seconds_from_period_end', axis=1, inplace=True)
    data.drop('minutes_remaining', axis=1, inplace=True)
    data.drop('seconds_remaining', axis=1, inplace=True)

    data["home_away"] = data['matchup'].str.contains('vs').astype('int')
    data.drop('matchup', axis=1, inplace=True)

    data['game_date'] = pd.to_datetime(data['game_date'])
    data['game_year'] = data['game_date'].dt.year
    data['game_month'] = data['game_date'].dt.month

    data.drop('game_date', axis=1, inplace=True)

    # loc_X, loc_y

    data['loc_x'] = pd.cut(data['loc_x'], 25)
    data['loc_y'] = pd.cut(data['loc_y'], 25)

    rare_action_types = data['action_type'].value_counts().sort_values().index.values[:20]
    data.loc[data['action_type'].isin(rare_action_types), 'action_type'] = 'Other'

    categorical_col = ['action_type', 'combined_shot_type', 'period', 'season', 'shot_type',
                       'shot_zone_area', 'shot_zone_basic', 'shot_zone_range', 'game_year',
                       'game_month', 'opponent', 'loc_x', 'loc_y']

    for column in categorical_col:
        dummies = pd.get_dummies(data[column])
        dummies = dummies.add_prefix("{}#".format(column))
        data.drop(column, axis=1, inplace=True)
        data = data.join(dummies)

    unknown_mask = data["shot_made_flag"].isnull()

    Y = data["shot_made_flag"][-unknown_mask]
    X = data[-unknown_mask]
    X_test = data[unknown_mask]
    X.drop('shot_made_flag', axis=1, inplace=True)


    from sklearn.feature_selection import VarianceThreshold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.decomposition import PCA

    threshold = 0.90
    vt = VarianceThreshold().fit(X)
    # Find feature names
    feat_var_threshold = X.columns[vt.variances_ > threshold * (1 - threshold)]

    model = RandomForestClassifier()
    model.fit(X, Y)
    feature_imp = pd.DataFrame(model.feature_importances_, index=X.columns, columns=["importance"])
    feat_imp_20 = feature_imp.sort_values("importance", ascending=False).head(20).index

    features = np.hstack([feat_var_threshold, feat_imp_20])
    features = np.unique(features)
    list(features)
    X = X.loc[:, features]
    X_test = X_test.loc[:, features]

    return X, Y, X_test

def train_test_split(x, y, suffleYN, testSize):

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=testSize, shuffle=suffleYN, random_state=1004)
    return X_train, X_test, y_train, y_test