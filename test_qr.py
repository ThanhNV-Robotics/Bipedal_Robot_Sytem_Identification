import numpy as np
from scipy import linalg, signal

A = np.array([[12, -51, 4, 4*2], 
              [6, 167, -68, -68*2], 
              [-4, 24, -41, -41*2],
              [-1, 2, 3, 3*2],
              [2, 1, 1, 1*2],
              [3, 2, 2, 2*2]])

Q, R, p = linalg.qr(A, pivoting=True)
print("pivots:", p)
P = np.eye(p.size)[p]

print(R@P)

print(np.allclose(A@P.T, np.dot(Q,R)))

print(R)


# print("\nPermutation matrix P:\n", P)
# print("rearanged R\n", R @ P)
# print("shape of A:", A.shape)
# print("Shape of Q:", Q.shape)
# print("Shape of R:", R.shape)
# print("Permutation matrix P:\n")
# print(P)
#print(Q.T@A)  # should be equal to R
# print("Matrix Q (Orthogonal):\n", Q)
# print("\nMatrix R (Upper Triangular):\n", R)

# # Verification: Check that Q * R is approximately equal to A
# print("\nVerification (Q @ R):\n", Q @ R)