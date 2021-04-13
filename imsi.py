a = [1,2,-3]
b = [2,-3,4]
import numpy as np

a = np.array([a, a])
b = np.array([b, b])

print(np.abs(a - b))
print(np.sum(np.abs(a - b)))