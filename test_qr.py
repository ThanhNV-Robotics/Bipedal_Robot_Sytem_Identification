import numpy as np
from scipy import linalg

A = np.array([[12, -51, 4, 4*2], 
              [6, 167, -68, -68*2], 
              [-4, 24, -41, -41*2],
              [-1, 2, 3, 3*2],
              [2, 1, 1, 1*2],
              [3, 2, 2, 2*2]])

Q, R, p = linalg.qr(A, pivoting=True)
print("pivots:", p)
P = np.eye(p.size)[p]
print(P)
print(R)
print(R@P)
print(np.allclose(A@P.T, np.dot(Q,R)))
