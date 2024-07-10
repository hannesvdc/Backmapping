import sys
sys.path.append('/Users/hannesvdc/Research/Projects/openmm_protein/library/')

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

import itertools
import time

from MARTINI import *
from martini_openmm.martini_openmm.martini import *
from Reconstruction import *
import numpy.linalg as lg

def sampleVelocities(masses, rng):
    v = []
    for i in range(len(masses)):
        v1 = rng.normal(0.0, 1.0/sqrt(masses[i]))
        v2 = rng.normal(0.0, 1.0/sqrt(masses[i]))
        v3 = rng.normal(0.0, 1.0/sqrt(masses[i]))
 
        v.extend([v1, v2, v3])
    v = np.array(v)
 
    return v

def macroStepHMC(macroContext, macroMasses, Dt, beta, rng):
    # Generate new velocities
    v = sampleVelocities(macroMasses, rng)
    macroContext.setVelocities(toVec3Vector(v, 'velocity'))
    state = macroContext.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True)
 
    # Get current state
    z = toNumpyVector(state.getPositions())
    v = toNumpyVector(state.getVelocities())
    f = np.divide(toNumpyVector(state.getForces()), np.repeat(macroMasses, 3))
    E = state.getPotentialEnergy()._value + state.getKineticEnergy()._value
 
    # Do first HMC step
    v12 = v + Dt*f/2.0
    zp = z + Dt*v12
    macroContext.setPositions(toVec3Vector(zp, 'length'))
    macroContext.setVelocities(toVec3Vector(v12, 'velocity'))
 
    # Do second HMC step
    state2 = macroContext.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True)
    f = np.divide(toNumpyVector(state.getForces()), np.repeat(macroMasses, 3))
    v1 = v12 + Dt*f/2.0
 
    # Compute  energy difference
    macroContext.setVelocities(toVec3Vector(v1, 'velocity'))
    state3 = macroContext.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True)
    Ep = state3.getPotentialEnergy()._value + state3.getKineticEnergy()._value
     
    return zp, -beta*(Ep - E)

def macroStepBrownian(macroContext, macroMasses, Dt, beta, rng):
	state = macroContext.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True)
	z = toNumpyVector(state.getPositions())
	E = state.getPotentialEnergy()._value

	zp = z + np.sqrt(2.0*Dt/beta)*rng.normal(0.0, 1.0, z.size)
	macroContext.setPositions(toVec3Vector(zp, 'length'))
	state2 = macroContext.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True)
	Ep = state2.getPotentialEnergy()._value

	return zp, -beta*(Ep - E)


def optimalMacroscopicStep(method=None, seed=None):
	temperature = 300.0
	beta = 1000.0/(temperature*8.3145)
	N = 10**5

	# Load macroscopic force field
	includeDir =  '/Users/hannesvdc/hannes_phd/Projects/openmm_protein/6kn2/6kn2_input_files/martini_ff/'
	topfilename = '/Users/hannesvdc/hannes_phd/Projects/openmm_protein/6kn2/6kn2_input_files/martini_ff/6kn2_topol_cg.top'
	grofilename = '/Users/hannesvdc/hannes_phd/Projects/openmm_protein/6kn2/6kn2_input_files/martini_ff/6kn2_solvated_cg.gro'
	grofile = GromacsGroFile(grofilename)
	topfile = GromacsMartiniV2TopFile(topfilename, periodicBoxVectors=grofile.getPeriodicBoxVectors(), includeDir=includeDir)

	# Load microscopic postitions
	pdbfilename = '/Users/hannesvdc/hannes_phd/Projects/openmm_protein/6kn2/6kn2_input_files/charmm36m_ff/6kn2.pdb'
	pdbfile = PDBFile(pdbfilename)
	#positions = pdbfile.getPositions()
	positions = toVec3Vector(np.load('data/average_position.npy'), 'length')

	# Load positions
	amino_acids = ["GLY", "PHE", "ARG", "SER", "PRO", "CYS", "PRO", "PRO", "PHE", "CYS"]
	rc = MARTINI(amino_acids, len(positions))
	z0 = rc.value(positions)
	print('z0', z0)

	# Create context and set positions
	macroSystem = topfile.createSystem()
	macroContext = Context(macroSystem, BrownianIntegrator(1.0, 1.0, 1.0), Platform.getPlatformByName("Reference"))
	macroMasses = [macroSystem.getParticleMass(i)._value for i in range(macroSystem.getNumParticles())]
	print("------------- SETUP DONE ---------------")

	if method == "Brownian":
		f = lambda context, mass, t_size, temp, ran: macroStepBrownian(context, mass, t_size, temp, ran)
	elif method == 'HMC':
		f = lambda context, mass, t_size, temp, ran: macroStepHMC(context, mass, t_size, temp, ran)

	dt_list = np.logspace(-6.0, -4.0, 10)
	acc_list = []
	for Dt in dt_list:
		print('\nUsing Dt =', Dt)
		macro_samples = np.zeros((z0.size, N))
		macro_samples[:,0] = np.copy(z0)
		macroContext.setPositions(toVec3Vector(z0, 'length'))
		lnalphas = []

		# Run macroscopic sampler
		if seed is None:
			rng = rd.RandomState(seed=int(time.time()))
		else:
			rng = rd.RandomState(seed=seed)
		n_accepted = 0.0
		for i in range(1, N):
			state = macroContext.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True)
			zp, lnalpha = f(macroContext, macroMasses, Dt, beta, rng)
			lnalphas.append(lnalpha)
			if np.log(rng.uniform()) <= lnalpha: # If accepted : keep current state and continue
				n_accepted += 1.0
			else: # If rejected : reset previous state and continue
				macroContext.setState(state)
			macro_samples[:,i] = toNumpyVector(macroContext.getState(getPositions=True).getPositions())

		acc_list.append(n_accepted/N)
		print("Total acceptance rate with Dt =", Dt, ":", n_accepted/N, 'and beta=',beta)

	if seed is not None:
		return macro_samples, lnalphas
	else:
		print('Plotting')
		plt.title("Macroscopic Acceptance Rate of " + method)
		plt.loglog(dt_list, acc_list)
		plt.xlabel(r'$\Delta t$')
		plt.show()

if __name__ == '__main__':
	optimalMacroscopicStep(method='Brownian')