'''
Created on 9 Oct 2020

@author: Hannes Vandecasteele
'''

from simtk import openmm, unit
from simtk.openmm import app

class MCMCSampler:
    
    def __init__(self, move, record):
        self.move = move
        self.record = record
        
    def initialize(self):
        self.n = 0
        
    def step(self, nSteps):
        pass
    
    def acceptanceRate(self):
        return self.move.acceptanceRate()