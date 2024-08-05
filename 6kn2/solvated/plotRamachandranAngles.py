import numpy as np
import matplotlib.pyplot as plt

import Ramachandran as rm
import Util as util

# Load the microscopic positions
store_directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Backmapping for Proteins/6kn2/solvated/'
data_filename = 'microscopicHMC_N=10000_dt=3e_5_.npy'
positions = np.load(store_directory + data_filename).T
print(positions.shape)

# Convert these positions to the (N_data, N_atoms, 3) tensor format
N_data = positions.shape[0]
N_atoms = positions.shape[1] // 3
x = np.reshape(positions, (N_data, N_atoms, 3))

# Compute the ramachandran angles of 6kn2
amino_acids = ["GLY", "PHE", "ARG", "SER", "PRO", "CYS", "PRO", "PRO", "PHE", "CYS"]
ram_calculator = rm.Ramachandran(amino_acids)
ram_angles = ram_calculator.computeDihedrals(x)

# Plot these Ramachandran angles on a grid per amino acid
for i in range(0, len(ram_angles)):
    plt.figure()
    plt.scatter(ram_angles[i][0], ram_angles[i][1])
    plt.xlabel(r"$\phi$")
    plt.ylabel(r"$\psi$")
    plt.xlim((-np.pi, np.pi))
    plt.ylim((-np.pi, np.pi))
    plt.title(amino_acids[i])
plt.show()