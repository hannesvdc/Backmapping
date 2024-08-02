import sys
sys.path.append('../../library/')
sys.path.append('../internal/')
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
import openmm.app as app
import openmm as openmm
import time

from Util import *

# Loading the macroscopic system
def loadMicroscopicSystem():
    includeDir =  '/Users/hannesvdc/Research/Projects/Backmapping for Proteins/6kn2/6kn2_input_files/charmm36m_ff/'
    topfilename = '/Users/hannesvdc/Research/Projects/Backmapping for Proteins/6kn2/6kn2_input_files/charmm36m_ff/6kn2_topol_36m.top'
    grofilename = '/Users/hannesvdc/Research/Projects/Backmapping for Proteins/6kn2/6kn2_input_files/charmm36m_ff/6kn2_clean_solvated_36m.gro'

    grofile = app.GromacsGroFile(grofilename)
    topfile = app.GromacsTopFile(topfilename, periodicBoxVectors=grofile.getPeriodicBoxVectors(), includeDir=includeDir)
    system = topfile.createSystem()
    context = openmm.Context(system, openmm.BrownianIntegrator(1.0, 1.0, 1.0), openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(grofile.getPositions())

    return system, context

# Implement Microscopic HMC Sampler
def microStep(context, masses, dt, beta, rng):
	extended_masses = np.repeat(masses, 3)
	
	# Generate new velocities
	v = sampleVelocities(masses, rng)
	context.setVelocities(toVec3Vector(v, 'velocity'))
	state = context.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True)

	# Get current state variables
	x = toNumpyArray(state.getPositions())
	v = toNumpyArray(state.getVelocities())
	f = toNumpyArray(state.getForces())
	E_pot = state.getPotentialEnergy()._value 
	E_kin = state.getKineticEnergy()._value

	# v1 and x steps
	v12 = v + dt*np.divide(f, extended_masses) / 2.0
	x1 = x + dt*v12

	# Update new positions and velocities
	context.setPositions(toVec3Vector(x1, 'length'))
	context.setVelocities(toVec3Vector(v12, 'velocity'))
	int_state = context.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True)
	int_f = toNumpyArray(int_state.getForces())

	# Final v2 step
	v1 = v12 + dt*np.divide(int_f, extended_masses) / 2.0

	# Compute energy difference for A/R
	context.setVelocities(toVec3Vector(v1,'velocity'))
	end_state = context.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True)
	E_pot_p = end_state.getPotentialEnergy()._value 
	E_kin_p = end_state.getKineticEnergy()._value

	return -beta*(E_pot_p - E_pot + E_kin_p - E_kin)

def runMicroscopicSampler(dt=1.e-3, N=10000):
    temperature = 300.0
    beta = 1000.0/(temperature*8.3145)
    system, context = loadMicroscopicSystem()
    masses = getMasses(system)
    print('Number of Atoms:', system.getNumParticles())
    print("------------- SETUP DONE ---------------")

    rng = np.random.RandomState(seed=round(time.time()))
    n_accepted = 0.0
    n_new_accepted = 0.0

    samples = np.zeros((3*system.getNumParticles(), N+1))
    samples[:,0] = toNumpyArray(context.getState(getPositions=True).getPositions())

    for i in range(1, N+1):
        if i % 10 == 0:
            print('i =', i, n_accepted / i, n_new_accepted / 10)
            n_new_accepted = 0.0
        microState = context.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True)

        lnalpha = microStep(context, masses, dt, beta, rng)
        if np.log(rng.uniform()) <= lnalpha: # If accepted : keep current state and continue
            n_accepted += 1.0
            n_new_accepted += 1.0
        else: # If rejected : reset previous state and continue
            context.setState(microState)

        samples[:,i] = toNumpyArray(context.getState(getPositions=True).getPositions())

    print('Total Acceptance Rate:', n_accepted / N)
    return samples, n_accepted/N

def findOptimalMicroscopicTimestep():
    dt_list = np.logspace(-6.0, -2.0, 15)
    acc_list = []

    for dt in dt_list:
        print('\n\n\ndt =', dt)
        acc = runMicroscopicSampler(dt=dt)
        acc_list.append(acc)

    print('Plotting')
    plt.loglog(dt_list, np.array(acc_list))
    plt.xlabel(r'$\delta t$')
    plt.ylabel('Acceptance Rate')
    plt.show()

if __name__ == '__main__':
    dt = 1.e-3
    N  = 10000
    samples, _ = runMicroscopicSampler(dt, N)

    filename = 'microscopicHMC_N=10000_dt=1e_3_.npy'
    directory = '/Users/hannesvdc/Research_Data/Backmapping for Proteins/6kn2/solvated/'
    np.save(directory + filename, samples)