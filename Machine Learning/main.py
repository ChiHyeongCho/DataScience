

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

# (ex, #0 : 사용자 Source 경로, #1 : Data 경로, #2 : Data 명 ...)

##################################################################

def main(*args):

    import pandas as pd
    import logging

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

    #print(rawData)

    logger.info("read data end..")

    ##################################################################

    # Second : Preprocess data from rawData to Train & Test set
    # Step : 1.remove Outlier 2.Feature Scaling 3.Test/Train Split & Shuffle

    import preprocessing

    logger.info("preprocessing start..")

    # Proprocess rawData according to data characteristics

    # For regression (ex. Demand/Time forecasting) --> preprocessingData

    # step 1 : remove outlier (from EDA or domain Knowledge)
    preprocessing_Data = rawData

    # step 2 : Feature scaling



    # For classification (ex. Image, Natural language) --> preprocessingData

    preprocessingData = rawData

    # X (Independent variable) & Y (Dependent variable) Split

    independentVar = preprocessingData.loc[:, ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]]
    dependentVar = preprocessingData.loc[:, ["Occupancy"]]

    # Train & Test Split (독립변수, 의존변수, Shuffle 유무, Test Set 사이즈)

    x_train, x_test, y_train, y_test = preprocessing.train_test_split(independentVar, dependentVar, True, 0.2)

    logger.info("preprocessing end..")

    ##################################################################

    # Third : Build Model

    ##################################################################

    # Fourth : Test & Tuning Model

    ##################################################################

    # Fifth : clear memory & Save Output


    logger.info("Program End..")


main("C:/Users/whcl3/Desktop/DS/Input/dataset.csv", "C:/Users/whcl3/Desktop/DS/Output", "C:/Users/whcl3/Desktop/DS/Model")