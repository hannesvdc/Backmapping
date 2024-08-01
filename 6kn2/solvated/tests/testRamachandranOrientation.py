import sys
sys.path.append('../')

import numpy as np

import Ramachandran as rm

A = np.array([[-1, 1, 0], [-1, 1, 0], [-1, 1, 0], [-1, 1, 0]], dtype=float)
B = np.array([[ 0, 0, 0], [ 0, 0, 0], [ 0, 0, 0], [ 0, 0, 0]], dtype=float)
C = np.array([[ 1, 0, 0], [ 1, 0, 0], [ 1, 0, 0], [ 1, 0, 0]], dtype=float) # Should yield a positive torsion angle
D = np.array([[ 1, 1, 0], [ 1,-1, 0], [ 2, 0,-1], [ 2, 0, 1]], dtype=float)

print(rm.__compute_torsions__(A, B, C, D))