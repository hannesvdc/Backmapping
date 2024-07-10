'''
Created on 9 Oct 2020

@author: Hannes Vandecasteele
'''

from simtk import openmm, unit
from simtk.openmm import app
import numpy as np

class MARTINI:
    
    def __init__(self, amino_acids, numberOfAtoms):
        self.amino_acids = amino_acids
        self.numberOfAtoms = numberOfAtoms
        
        self.initialize()
        
    def initialize(self):
        H = 1.00784
        C = 12.0107
        N = 14.0067
        O = 15.999
        S = 32.065
        
        self.AminoAcidSizes = { "GLY": 9, "ALA": 0, "SER": 11, "CYS": 10, "THR": 0, "VAL": 0, "LEU": 0, "IIE": 0, "MET": 0, "PRO": 14, 
                                 "ASP": 0, "GLU": 0, "PHE": 20, "TYR": 0, "TRP": 0, "ASN": 0, "GLN": 0, "LYS": 0, "ARG": 24}
        self.AminoAcidCGSizes = { "GLY": 1, "ALA": 0, "SER": 2, "CYS": 2, "THR": 0, "VAL": 0, "LEU": 0, "IIE": 0, "MET": 0, "PRO": 2, 
                                 "ASP": 0, "GLU": 0, "PHE": 4, "TYR": 0, "TRP": 0, "ASN": 0, "GLN": 0, "LYS": 0, "ARG": 3}
        self.n = 3*np.sum(np.array(list(map(lambda y: self.AminoAcidCGSizes[y], self.amino_acids))))
        self.d = 3*self.numberOfAtoms
         
        self.AminoAcidCGIndices = {
            "GLY": [[0,1,2,3,4,5,6]],
            "PHE": [[0,1,2,3,11], [4,5,6,15], [7,9,16,18], [8, 10, 17, 19]],
            "ARG": [[0,1,2,3,11], [4,5,6], [7, 8, 9, 10, 21, 22]],
            "SER": [[0,1,2,3,6], [4,5,10]],
            "PRO": [[0,1,2,3], [4,5,6]],
            "CYS": [[0,1,2,3,6], [4,5]]
            }
        
        self.AminoAcidMasses = {
            "GLY": np.array([N, C, C, O, H, H, H, H, H]),
            "PHE": np.array([N, C, C, O, C, C, C, C, C, C, C, H, H, H, H,H, H, H, H, H]),
            "ARG": np.array([N, C, C, O, C, C, C, N, C, N, N, H, H, H, H, H, H, H, H, H, H, H, H, H]),
            "SER": np.array([N, C, C, O, C, O, H, H, H, H, H]),
            "PRO": np.array([N, C, C, O, C, C, C, H, H, H, H, H, H, H]),
            "CYS": np.array([N, C, C, O, C, S, H, H, H, H])
        }
        
        self.M = np.zeros((self.n, self.d))
        i = 0
        row = 0
        for aa in self.amino_acids:
            row = self.fill(row, i, aa)
            i += 3*self.AminoAcidSizes[aa]
        
        #print('M = ', self.M)
        #print('row sum', np.sum(self.M, axis=1))
        
    def fill(self, row, i, AA):
        cgs = self.AminoAcidCGIndices[AA]
        ms = self.AminoAcidMasses[AA]
        
        for m in range(len(cgs)):
            indices = cgs[m]
            total_mass = np.sum(ms[indices])
            
            for j in range(len(indices)):
                self.M[row+3*m:row+3*(m+1), i+3*indices[j]:i+3*indices[j]+3] = np.identity(3)*ms[indices[j]]/total_mass
                
        return row + 3*len(cgs)
        
    def value(self, x):
        return self.M.dot(self.toNumpyArray(x))

    def value_internal(self, x):
        return self.M.dot(x)
    
    def gradient(self, _):
        return self.M
    
    def toNumpyArray(self, x):
        a = np.zeros(self.d)
        for i in range(self.numberOfAtoms):
            if isinstance(x[i].x, float):
                a[3*i:3*(i+1)] = np.array([x[i].x, x[i].y, x[i].z])
            else:
                a[3*i:3*(i+1)] = np.array([x[i].x._value, x[i].y._value, x[i].z._value])
            
        return a