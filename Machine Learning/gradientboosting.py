
def xgb(X_train, Y_train, lr, depth):
    import numpy as np
    import xgboost as xgb

    # reg:linear, binary:logistic, multi=softmax
    model = xgb.XGBClassifier(n_estimators = 500, learning_rate = lr, max_depth = depth, objective = 'binary:logistic')

    model.fit(X_train, Y_train)

    return model


