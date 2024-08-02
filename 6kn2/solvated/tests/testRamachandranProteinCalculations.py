import sys
sys.path.append('../')

import openmm.app as app
import numpy as np

import Ramachandran as rm
import Util as util

# Load the 6kn2 position data
includeDir =  '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Backmapping for Proteins/6kn2/6kn2_input_files/charmm36m_ff/'
grofilename = '6kn2_clean_solvated_36m.gro'
grofile = app.GromacsGroFile(includeDir + grofilename)
positions = grofile.getPositions()

# Convert positions to a (1, N_atoms, 3) - tensor
N_atoms = len(positions)
x = np.reshape(util.toNumpyArray(positions), (N_atoms, 3))[np.newaxis, :, :]

# Compute the ramachandran angles of 6kn2
amino_acids = ["GLY", "PHE", "ARG", "SER", "PRO", "CYS", "PRO", "PRO", "PHE", "CYS"]
ram_calculator = rm.Ramachandran(amino_acids)
print(ram_calculator.computeDihedrals(x))
