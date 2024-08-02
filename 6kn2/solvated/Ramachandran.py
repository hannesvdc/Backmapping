import numpy as np

# A, B, C and D are (N, 3) tensors
def __compute_torsions__(A, B, C, D):
    b1 = A - B
    b2 = C - B
    b3 = D - C
    b2_n = b2 / np.sqrt(np.sum(b2 * b2, axis=1, keepdims=True))

    n1 = np.cross(b1, b2, axisa=1, axisb=1)
    n1 = n1 / np.sqrt(np.sum(n1 * n1, axis=1, keepdims=True))
    n2 = np.cross(b2, b3, axisa=1, axisb=1)
    n2 = n2 / np.sqrt(np.sum(n2 * n2, axis=1, keepdims=True))
    m = np.cross(n1, b2_n, axisa=1, axisb=1)
    
    xp = np.sum(n1 * n2, axis=1)
    yp = np.sum( m * n2, axis=1)
    phi = np.arctan2((-1)*yp, (-1)*xp)

    return phi

class Ramachandran:
    def __init__(self, amino_acids):
        self.amino_acids = amino_acids
        self.lens = {"GLY": 9, 
                     "PHE": 20, 
                     "ARG": 24, 
                     "SER": 11, 
                     "PRO": 14, 
                     "CYS": 10}

    # We assume that x is a (N_data, N_atoms, 3) - numpy tensor
    def computeDihedrals(self, x):
        ram_angles = []

        total_len = 0
        for i in range(len(self.amino_acids)):
            N = x[:, total_len + 0, :]
            Ca= x[:, total_len + 1, :]
            C = x[:, total_len + 2, :]

            # Boundary Conditions
            if i == 0:
                Cprev = x[:, 4, :] # This is the C in the CH3 initial group
            else:
                Cprev = x[: total_len - self.lens[self.amino_acids[i-1]] + 2, :] # C atom from the previous amino-acid

            if i == len(self.amino_acids)-1:
                Nnext = x[:, total_len + 4, :] # This is the terminal N atom
            else:
                Nnext = x[:, total_len + self.lens[self.amino_acids[i]] + 0, :]

            phi = __compute_torsions__(Cprev, N, Ca, C) # According to Wim, Ca must be the 'C' atom on the central bond
            psi = __compute_torsions__(Nnext, C, Ca, N) # So both looking directions are opposite
            ram_angles.append((phi, psi))

            # Bookkeeping for the next amino-acid
            total_len = total_len + self.lens[self.amino_acids[i]]
		
        return ram_angles