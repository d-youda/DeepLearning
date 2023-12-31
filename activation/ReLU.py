import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0,x) #0과 x중 큰 값 출력하기

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()