
def xgb(X_train, X_test, Y_train, Y_test):
    import numpy as np
    import xgboost as xgb

    # reg:linear, binary:logistic, multi=softmax
    model = xgb.XGBClassifier(n_estimators = 500, learning_rate = 0.2, max_depth = 1, early_stoppings = 20, objective = 'binary:logistic')

    model.fit(X_train, Y_train)
    xgb_pred = model.predict(X_test)
    xgb.plot_importance(model)
    import matplotlib.pyplot as plt
    plt.show()
    print(model.score(X_train, Y_train))

    import matplotlib.pyplot as plt
    plt.plot(Y_test, xgb_pred)
    plt.show()


