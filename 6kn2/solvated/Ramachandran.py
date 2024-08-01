import numpy as np

# A, B, C and D are (N, 3) tensors
def __compute_torsions__(A, B, C, D):
    b1 = A - B
    b2 = C - B
    b3 = D - C
    b2_n = b2 / np.sqrt(np.sum(b2 * b2, axis=1))

    n1 = np.cross(b1, b2, axisa=1, axisb=1)
    n1 = n1 / np.sqrt(np.sum(n1 * n1, axis=1))
    n2 = np.cross(b2, b3, axisa=1, axisb=1)
    n2 = n2 / np.sqrt(np.sum(n2 * n2, axis=1))
    m = np.cross(n1, b2_n, axisa=1, axisb=1)
    
    xp = np.sum(n1 * n2, axis=1)
    yp = np.sum( m * n2, axis=1)
    phi = np.arctan2((-1)*yp, (-1)*xp)

    return -phi