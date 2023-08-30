import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    if x>=0 : return 1
    elif x<1: return 0

def step_function_np(x):
    '''x가 0보다 크면 true, 작으면 False 반환 -> 숫자로 바꾸면 True:1 False:0이기 때문에 이렇게도 사용 가능함
        이 경우 위에 정의한 일반 계단 함수와는 다르게 배열이 들어가도 출력할 수 있다는 장점이 있음'''
    y = x>0
    return y.astype(np.int)

x = np.arange(-5.0, 5.0, 0.1) # -5.0~5.0 사이를 0.1 간격의 숫자로 잘라서 넘파이 배열에 저장함
y = step_function_np(x)

plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show()