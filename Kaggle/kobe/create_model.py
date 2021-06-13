

##################################################################

# 해당 소스파일의 저작권은 없습니다.
# 필요하신 분들은 언제나 사용하시길 바라며, 해당 소스코드의 부족한 부분에 대해서는
# whcl303@hanmail.net으로 언제든지 피드백 주시길 바랍니다:)
# 소스설명 : 데이터분석(EDA 등) 후, 데이터 모델링을 위한 Main 함수 구조입니다.

##################################################################

# 가변길이 argument를 받을 때 *args 사용 (*만 있으면 사용가능)

########################### args 리스트 ###########################

# Local환경 모델

#0 : Data 경로,  #1 : Log File 경로, #2 : Model 결과 저장 위치

# (ex, #0 :  Data 경로(train), #1 : Data 경로(test), #2 : Output #3 : 모델)

##################################################################

def main(*args):

    import pandas as pd
    import logging
    import numpy as np

    ##################################################################

    # preparations : arguments 변수 정의 및 logging 설정
    # logging level : DEBUG, INFO, WARNING, ERROR, CRITICAL

    data_adrress = args[0]
    log_adrress = args[1]
    model_output_adrress = args[2]

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("log")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_adrress + "/log.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logging.basicConfig()

    logger.info("Program Start..")

    ##################################################################

    # Frist : Read data from the directory -> rawData

    import read_data

    logger.info("read data start..")

    rawData = read_data.read_csv(data_adrress)

    # 콘솔창 출력 column 확대
    pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)

    #print(rawData_test)

    logger.info("read data end..")

    ##################################################################

    # Second : Preprocess data from rawData to Train & Test set
    # Step : 1.remove Outlier 2.Feature Scaling 3.Test/Train Split & Shuffle

    import preprocessing

    logger.info("preprocessing start..")

    # Proprocess rawData according to data characteristics

    # step 1 : remove outlier & feature scaling (from EDA or domain Knowledge)

    # X (Independent variable) & Y (Dependent variable) Split

    preprocessing_Data_X, preprocessing_Data_Y, x_test = preprocessing.preprocessing(rawData)


    x_train = preprocessing_Data_X
    y_train = preprocessing_Data_Y

    # print(x_train.shape, y_train.shape)

    logger.info("preprocessing end..")

    ##################################################################

    # Third : Build Model

    from sklearn.model_selection import KFold

    logger.info("build Model start..")

    import knn
    import logistic_regression
    import randomforest
    import gradientboosting
    from sklearn.model_selection import KFold

    num_folds = 5
    num_instance = len(y_train)
    k_fold = KFold(n_splits = num_folds, shuffle=True)


    KFold(num_folds)
    x_train = x_train.values
    y_train = y_train.values.ravel()

    knn_model = knn.KNN_clf(10, x_train, y_train)
    logistic_model = logistic_regression.regression(x_train, y_train)
    randomforest_model = randomforest.randomforest(x_train, y_train, 20, 0)
    xgboost_model = gradientboosting.xgb(x_train, y_train, 0.01, 100)

    logger.info("build Model end..")

    ##################################################################

    # Fourth : Test & Tuning Model

    logger.info("test start..")

    from sklearn import metrics
    from sklearn.model_selection import cross_val_score

    y_pred_knn_model = knn_model.predict(x_train)
    print('knn_model 정확도 :', metrics.accuracy_score(y_train, y_pred_knn_model))
    print('knn_model 정확도 (K-Fold) :', np.mean(cross_val_score(knn_model, x_train, y_pred_knn_model, cv = k_fold, scoring= 'neg_log_loss')))

    y_pred_logistic_model = logistic_model.predict(x_train)
    print('logistic_model 정확도 :', metrics.accuracy_score(y_train, y_pred_logistic_model))
    print('logistic_model 정확도 (K-Fold) :', np.mean(cross_val_score(logistic_model, x_train, y_pred_logistic_model, cv = k_fold, scoring= 'neg_log_loss')))

    y_pred_randomforest_model = randomforest_model.predict(x_train)
    print('randomforest_model 정확도 :', metrics.accuracy_score(y_train, y_pred_randomforest_model))
    print('randomforest_model 정확도 (K-Fold) :', np.mean(cross_val_score(randomforest_model, x_train, y_pred_randomforest_model, cv = k_fold, scoring= 'neg_log_loss')))

    y_pred_xgboost_model = xgboost_model.predict(x_train)
    print(' xgboost_model 정확도 :', metrics.accuracy_score(y_train, y_pred_xgboost_model))
    print(' xgboost_model 정확도 (K-Fold) :', np.mean(cross_val_score( xgboost_model, x_train, y_pred_xgboost_model, cv = k_fold, scoring= 'neg_log_loss')))

    y_pred_xgboost_model_final = xgboost_model.predict(x_test)

    logger.info("test end..")

    ##################################################################

    # Fifth : clear memory & Save Output

    logger.info("save start..")

    import joblib
    joblib.dump(xgboost_model, model_output_adrress+"./xgboost_model.pkl")

    logger.info("save end..")
    logger.info("Program End..")


main("C:/Users/whcl3/PycharmProjects/DataScience/Kaggle/kobe/Input/data.csv", "C:/Users/whcl3/PycharmProjects/DataScience/Kaggle/kobe/Output", "C:/Users/whcl3/PycharmProjects/DataScience/Kaggle/kobe/Model")