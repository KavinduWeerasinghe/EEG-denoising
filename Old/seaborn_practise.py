import numpy as np

a=np.array([0,1,1,1,0])
b=np.array([1,0,1,1,0])
c=np.array([0,1,1,1,1])

print((a&b)|(a&c)|(b&c))