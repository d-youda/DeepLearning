import numpy as np
a = np.array([[1,2], [3,4]])
b = np.array([[5,6], [7,8]])

multiply1 = np.dot(a,b)
multiply2 = np.dot(b,a)

print(multiply1)
print(multiply2)