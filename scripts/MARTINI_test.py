'''
Created on 23 Oct 2020

@author: Hannes Vandecasteele
'''

import sys
sys.path.append('../')

from library.MARTINI import MARTINI

def martini_test():
    amino_acids = ["GLY", "PHE", "ARG", "SER", "PRO", "CYS", "PRO", "PRO", "PHE", "CYS"]
    numberOfAtoms = 150
    
    _ = MARTINI(amino_acids, numberOfAtoms)
    
    
if __name__ == '__main__':
    martini_test()