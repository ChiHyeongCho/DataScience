
def randomforest(x, y, depth, state):

    from sklearn import ensemble
    import numpy as np

    model = ensemble.RandomForestClassifier(criterion='entropy', n_estimators=100, max_depth=depth, random_state=state)
    model.fit(x, np.ravel(y))

    return model