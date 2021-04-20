
##################################################################

# 해당 소스파일의 저작권은 없습니다.
# 필요하신 분들은 언제나 사용하시길 바라며, 해당 소스코드의 부족한 부분에 대해서는
# whcl303@hanmail.net으로 언제든지 피드백 주시길 바랍니다.
# 소스설명 : 데이터 읽기 함수로 EDA 등 데이터분석이 끝난 데이터를 읽어오는 함수입니다.
# 포함함수 : csv 파일 읽기, excel 파일 읽기

##################################################################


def read_csv(address):

    import pandas as pd

    data = pd.read_csv(address)

    return data

def read_excel(address):

    import pandas as pd

    data = pd.read_excel(address)

    return data
