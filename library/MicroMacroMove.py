'''
Created on 13 Oct 2020

@author: Hannes Vandecasteele
'''
from library.MCMCMove import MCMCMove

from simtk import openmm, unit
from simtk.openmm import app
from simtk.openmm.Vec3 import Vec3

import numpy as np
import numpy.linalg as lg

import copy as copy

def toVec3(x):
    l = int(x.size/3)
    a = [None] * l
    
    for i in range(l):
        a[i] = Vec3(x[3*i], x[3*i+1], x[3*i+2])
        
    return a

class MicroMacroMove(MCMCMove):
    
    def __init__(self, macroContext, microContext, K, dt, temperature):
        self.macroContext = macroContext
        self.microContext = microContext
        
        self.nSamples = 0
        self.nMacroAccepted = 0
        self.nMicroAccepted = 0
        
        self.K = K
        self.dt = dt
        self.Kb = 8.314459/1000. # units (nm)**2 x Au x (ps)**-2 x K**-1
        self.beta = 1./(self.Kb*temperature)
        
        self.d = len(microContext.getPosition())
        
    def step(self):
        self.nSamples += 1
        
        # Perform a macroscopic step
        prevMState = copy.copy(self.macroContext.getState(getPositions=True, getVelocities=False, getForces=False, getEnergy=True))
        self.macroContext.getIntegrator().step(1)
        nextMState = self.macroContext.getState(getPositions=True, getVelocities=False, getForces=False, getEnergy=True)
        
        u = self.rng.uniform()
        lnMalpha = -self.beta*(nextMState.getPotentialEnergy() - prevMState.getPotentialEnergy()) # We assume a Brownian or HMC-like motion at the macroscopic level
        if np.log(u) > lnMalpha:
            self.macroContext.setState(prevMState)
            return self.microContext.getPositions(), self.macroContext.getPositions()
        self.nMacroAccepted += 1
        
        # Perform Indirect Reconstruction
        zp = copy.copy(nextMState.getPositions())
        prevmState = copy.copy(self.microContext.getState(getPositions=True, getVelocities=False, getForces=False, getEnergy=True))
        self.Nprev = self.Nnew
        self.Nnew = self.indirect_reconstruction(zp)
        
        u = self.rng.uniform()
        lnmalpha = -self.beta*(self.Nnew - self.Nprev) - lnMalpha
        if np.log(u) <= lnmalpha:
            self.nMicroAccepted += 1
        else:
            self.microContext.setState(prevmState)
        return self.microContext.getPositions(), self.macroContext.getPositions()
    
    def indirect_reconstruction(self, z):
        samples = []
        energies = []
        M = np.transpose(self.martini.gradient())
        
        prevState = copy.copy(self.microContext.getState(getPositions=True, getVelocities=False, getForces=True, getEnergy=True))
        prevzp = self.martini.value(prevState.getPositions())
        for _ in range(self.K):
            xp = prevState.getPositions() + self.dt*prevState.getForces() \
                 + self.lam*self.dt*M.dot(prevzp - z) + toVec3(np.sqrt(2.*self.dt/self.beta)*self.rng.normal(0., 1., 3*self.d))
            
            self.microContext.setPositions(xp)
            newState = copy.copy(self.microContext.getState(getPositions=True, getVelocities=False, getForces=True, getEnergy=True))
            newzp = self.martini.value(newState.getPositions())
            lnalpha = -self.beta*(newState.getPotentialEnergy() - prevState.getPotentialEnergy() + self.lam*lg.norm(newzp - z)**2 - self.lam*lg.norm(prevzp - z)**2) \
                      -self.beta/4.*np.dot(-newState.getForces() - prevState.getForces + self.lam*M.dot(newzp - z) + self.lam*M.dot(prevzp - z), \
                                           2*prevState.getPositions()-2*xp + (-newState.getForces() + self.lam*M.dot(newzp - z))*self.dt - (-prevState.getForces() + self.lam*M.dot(prevzp - z))*self.dt)
                      
            u = self.rng.uniform()
            if np.log(u) <= lnalpha:
                prevzp = np.copy(newzp)
                prevState = copy.copy(newState)
            else:
                self.microContext.setState(prevState)
        
            samples.append(prevState.getPositions())
            energies.append(prevState.getPotentialEnergy())
            
        return self.FEC_cartesian(samples, energies, z)
    
    def FEC_cartesian(self, samples, energies, z):
        N_values = np.exp(-self.beta*np.array(energies))
        
        visited = np.zeros(len(samples))
        for k in range(self.K):
            if visited[k] > 0:
                continue
            
            indices = [k]
            num = 1.
            for j in range(k+1, self.K):
                if visited[j] == 0 and self.in_square(samples[j], samples[k]):
                    indices.append(j)
                    num += 1.
                    visited[j] = 1.
            
            N_values[indices] /= (num/self.K*np.prod(self.binsizes))
            
        Nval = np.average(N_values)
        return -self.beta*np.log(Nval)
    
    def in_square(self, x, y):
        return np.all(x >= y) and np.all(x <= y + self.binsizes)
        
    def getMacroscopicAcceptanceRate(self):
        return self.nMacroAccepted/self.nSamples
    
    def getMicroscopicAcceptanceRate(self):
        return self.nMicroAccepted/self.nMacroAccepted
    
    def getAcceptanceRate(self):
        return self.nMicroAccepted/self.nSamples # self.getMacroscopicAcceptanceRate() * self.getMicroscopicAcceptanceRate()
    