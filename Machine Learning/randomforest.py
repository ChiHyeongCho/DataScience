
def randomforest(x, y, max_depth, random_state):

    from sklearn import ensemble
    import numpy as np

    clf = ensemble.RandomForestClassifier(criterion='entropy', n_estimators=100, max_depth=max_depth, random_state=random_state)
    clf.fit(x, np.ravel(y))
