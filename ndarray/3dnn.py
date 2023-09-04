import numpy as np

X = np.random.randint(1,10,size=(1,2))
W = np.array([[1,3,5], [2,4,6]])
Y = np.dot(X,W)

print(X)
print()

print(W)
print()

print(Y)

