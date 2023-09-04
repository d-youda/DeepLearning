import numpy as np

#1d array
a = np.array([1,2,3,4])
print("-"*5, "a info", "*"*5)
print(np.ndim(a))#a의 dimension 출력
print(a.shape) #a의 모양 출력
print(a.shape[0])

#2d array
b = np.array([[1,2], [2,3], [3,4]])
print("-"*5, "b info", "*"*5)
print(b)
print(np.ndim(b))
print(b.shape)