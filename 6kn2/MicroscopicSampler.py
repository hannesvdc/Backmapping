import sys
sys.path.append('/Users/hannesvdc/hannes_phd/Projects/openmm_protein/library/')
sys.path.append('/Users/hannesvdc/hannes_phd/Projects/openmm_protein/6kn2/internal/')

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

import itertools
import time

from Reconstruction import *
from MARTINI import *
from martini_openmm.martini_openmm.martini import *
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

def microStep(macroContext, microContext, rc, masses, lam, dt, beta, rng):
	# Generate new velocities
	v = sampleVelocities(masses, rng)
	microContext.setVelocities(toVec3Vector(v, 'velocity'))
	microState = microContext.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True)
	macroState = macroContext.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True)

	# Get current microscopic state
	x = toNumpyVector(microState.getPositions())
	v = toNumpyVector(microState.getVelocities())
	fx = toNumpyVector(microState.getForces())
	Ex = microState.getPotentialEnergy()._value 
	Ev = microState.getKineticEnergy()._value

	# Get current macroscopic state
	z = toNumpyVector(macroState.getPositions())
	fz =toNumpyVector(macroState.getPositions())
	Ez = macroState.getPotentialEnergy()._value
	E_pot = lam*Ex + (1.0-lam)*Ez
	E_kin = Ev

	# v1 and x steps
	g = np.transpose(rc.gradient(x))#rc.value_internal(x) # Wrong !!
	f = np.divide(lam*fx + (1.0-lam)*g.dot(fz), np.repeat(masses, 3))
	v12 = v + dt*f/2.0
	x1 = x + dt*v12
	z1 = rc.value_internal(x1)

	# Update new positions and velocities
	microContext.setPositions(toVec3Vector(x1, 'length'))
	microContext.setVelocities(toVec3Vector(v12, 'velocity'))
	macroContext.setPositions(toVec3Vector(z1, 'length'))
	ms = microContext.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True)
	Ms = macroContext.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True)
	fx = toNumpyVector(ms.getForces())
	fz = toNumpyVector(Ms.getForces())

	# Final v2 step
	g = np.transpose(rc.gradient(x1))#g = rc.value_internal(x1)
	f = np.divide(lam*fx + (1.0-lam)*g.dot(fz), np.repeat(masses, 3))
	v1 = v12 + dt*f/2.0

	# Compute energy difference for A/R
	microContext.setVelocities(toVec3Vector(v1,'velocity'))
	s = microContext.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True)
	Exp = s.getPotentialEnergy()._value 
	Evp = s.getKineticEnergy()._value
	Ezp = Ms.getPotentialEnergy()._value
	E_pot_p = lam*Exp + (1.0-lam)*Ezp
	E_kin_p = Evp

	return x1, z1, -beta*(E_pot_p - E_pot + E_kin_p - E_kin)

def optimalMicroscopicStep():
	temperature = 300.0
	beta = 1000.0/(temperature*8.3145)
	N = 10**4

	# Loading the microscopic system
	pdbfilename = '/Users/hannesvdc/hannes_phd/Projects/openmm_protein/6kn2/6kn2_input_files/charmm36m_ff/6kn2.pdb'
	pdbfile = PDBFile(pdbfilename)
	forcefield = ForceField('amber99sb.xml')
	microSystem = forcefield.createSystem(pdbfile.topology)
	masses = [microSystem.getParticleMass(i)._value for i in range(microSystem.getNumParticles())]

	integrator = BrownianIntegrator(temperature, 1.0, 1.0)
	microContext = Context(microSystem, integrator, Platform.getPlatformByName("Reference"))

	# Setting up the reaction coordinate and the randon number generator
	amino_acids = ["GLY", "PHE", "ARG", "SER", "PRO", "CYS", "PRO", "PRO", "PHE", "CYS"]
	rng = rd.RandomState()
	rc = MARTINI(amino_acids, microSystem.getNumParticles())

    # Loading the macroscopic system
	includeDir =  '/Users/hannesvdc/hannes_phd/Projects/openmm_protein/6kn2/6kn2_input_files/martini_ff/'
	topfilename = '/Users/hannesvdc/hannes_phd/Projects/openmm_protein/6kn2/6kn2_input_files/martini_ff/6kn2_topol_cg.top'
	grofilename = '/Users/hannesvdc/hannes_phd/Projects/openmm_protein/6kn2/6kn2_input_files/martini_ff/6kn2_solvated_cg.gro'

	grofile = GromacsGroFile(grofilename)
	topfile = GromacsMartiniV2TopFile(topfilename, periodicBoxVectors=grofile.getPeriodicBoxVectors(), includeDir=includeDir)
	macroSystem = topfile.createSystem()
	macroContext = Context(macroSystem, BrownianIntegrator(1.0, 1.0, 1.0), Platform.getPlatformByName("Reference"))
	
	# Option 1: postions = those in pdb file
	#x0 = pdbfile.getPositions()
	#z0 = toVec3Vector(rc.value(x0), 'length')
	# Option 2: positions = local minimum
	x0 = toVec3Vector(np.load('internal/LocalMinimum.npy'), 'length')
	z0 = toVec3Vector(rc.value(x0), 'length')
	microContext.setPositions(x0)
	macroContext.setPositions(z0)
	print("------------- SETUP DONE ---------------")

	for lam in np.linspace(0.0, 1.0, 11):
	#for lam in [1.0]:
		print('\n\n lambda =', lam)
		dt_list = np.logspace(-7.0, -2.0, 25)
		acc_list = []
		for dt in dt_list:
			print('\nUsing dt =', dt)
			microContext.setPositions(x0)
			macroContext.setPositions(z0)

			# Run microscopic sampler
			rng = rd.RandomState(seed=int(time.time()))
			n_accepted = 0.0
			for i in range(1, N+1):
				macroState = macroContext.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True)
				microState = microContext.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True)

				xp, zp, lnalpha = microStep(macroContext, microContext, rc, masses, lam, dt, beta, rng)
				if np.log(rng.uniform()) <= lnalpha: # If accepted : keep current state and continue
					n_accepted += 1.0
				else: # If rejected : reset previous state and continue
					microContext.setState(microState)
					macroContext.setState(macroState)

			acc_list.append(n_accepted/N)
			print("Total acceptance rate with dt =", dt, ":", n_accepted/N, 'beta=', beta)

		print('Plotting')
		plt.loglog(dt_list, acc_list, label=r'$\lambda = $' + str(lam))
		plt.xlabel(r'$\delta t$')
		plt.legend()
	plt.title("Microscopic Acceptance Rate")	
	plt.show()

if __name__ == '__main__':
	optimalMicroscopicStep()
