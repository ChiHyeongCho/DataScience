import numpy as np
import matplotlib.pyplot as plt

# np.linspace : 1차원 배열 만들기, 그래프 그리기
# np.linspace(start, stop, num) start : 배열 시작값, stop : 배열 끝값, num : 간격(Default 50)

x = np.linspace(0, 1, 100)

print(x)

# entropy 함수
y = -x*np.log2(x) - (1-x)*np.log2(1-x)

plt.figure(figsize = (10, 8))
plt.plot(x, y, linewidth = 3)
plt.xlabel(r'$x$', fontsize = 15)
plt.axis('equal')
plt.grid(alpha = 0.3)
plt.show()