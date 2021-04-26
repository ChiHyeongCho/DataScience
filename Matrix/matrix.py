
##################################################################

# 해당 소스파일의 저작권은 없습니다.
# 필요하신 분들은 언제나 사용하시길 바라며, 해당 소스코드의 부족한 부분에 대해서는
# whcl303@hanmail.net으로 언제든지 피드백 주시길 바랍니다:)
# 소스설명 : Machine Learning 에서 일반적으로 사용되는 Matrix 연산 (Numpy 사용)

##################################################################

# matrix 생성

import numpy as np

X = np.matrix([[1], [1]])
Y = np.matrix([[2], [0]])
print(X)
print(Y)

# Transpose 연산 (Vector.T)

A = Y.T*Y

print(A)

# Vector to Scaler
print(float(A))


# Vector Projection (2 차원, X -> Y)

omega = (X.T*Y)/(Y.T*Y)
omega = float(omega)
W = omega*Y

# Vector Projection (3 차원, B -> A)

A = np.matrix([[1,0],[0,1],[0,0]])
B = np.matrix([[1],[1],[1]])

X = (A.T*A).I*A.T*B
Bstar = A*X
print(Bstar)