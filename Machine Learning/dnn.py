
def dnn(X_train, X_test, Y_train, Y_test):

    import keras
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    Y_train = Y_train
    Y_test = Y_test

    from keras.models import Sequential
    from keras.layers import Dense, Dropout

    model = Sequential([
        Dense(units=16, kernel_initializer='uniform', input_dim= len(X_train[0]), activation='relu'),
        Dense(units=18, kernel_initializer='uniform', activation='relu'),
        Dropout(0.25),
        Dense(20, kernel_initializer='uniform', activation='relu'),
        Dense(24, kernel_initializer='uniform', activation='relu'),
        Dense(1, kernel_initializer='uniform', activation='sigmoid')
    ])

    print(model.summary())

    callbacks = [keras.callbacks.EarlyStopping(patience=5)]
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=15, epochs=10, callbacks = callbacks)

    score = model.evaluate(X_test, Y_test, verbose=0)
    print(model.metrics_names)
    print(score)