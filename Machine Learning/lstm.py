
def LSTM(X_train, X_test, Y_train, Y_test):

    import keras
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import pandas as pd

    sc = StandardScaler()

    from keras.preprocessing.sequence import pad_sequences
    from keras.utils import to_categorical

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    Y_train = sc.fit_transform(Y_train)
    Y_test = sc.transform(Y_test)

    ls_X_train = pd.DataFrame()
    ls_Y_train = pd.DataFrame()

    for i in range(len(X_train)-3):
        x = []
        for j in range(3):
            x.append(X_train[i+j])

        ls_X_train.append([x])



    for i in range(len(Y_train)-3):
        x = []
        for j in range(3):
            x.append(Y_train[i+j])
        ls_Y_train.append([x])

    X_train = np.array(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))

    Y_train = to_categorical(Y_train)[3 :]
    Y_test = to_categorical(Y_test)[3 :]



    print(X_train.shape)
    print(X_test)
    print(Y_train.shape)
    print(Y_test.shape)



    from keras.models import Sequential
    from keras.layers import Dense, Dropout, SimpleRNN, LSTM

    np.random.seed(0)
    model = Sequential()
    model.add(LSTM(10, input_dim=1, input_length=3))  # 3개의 값이 각각 1개의 뉴런에 1개씩 들어가고 총 10개 데이터임
    model.add(Dense(1))

    print(model.summary())

    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
                 keras.callbacks.ModelCheckpoint(filepath='best_model.h5',monitor='val_loss', save_best_only=True)]

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=15, epochs=10, callbacks = callbacks)

    score = model.evaluate(X_test, Y_test, verbose=0)
    print(model.metrics_names)
    print(score)