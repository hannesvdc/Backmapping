####################################################################
####################################################################
#																   #
#																   #
#         Find a local minimum of the potential energy			   #
#         surface to initialize the Address mM-MCMC                #
#	      method. This starting point improves the sampling        #
#																   #
#																   #
####################################################################
####################################################################

# Author: Hannes Vandecasteele, PhD
# Affiliation: KU Leuven, Johns Hopkins University

import sys
sys.path.append('/Users/hannesvdc/hannes_phd/Source/openmm_protein/library/')
sys.path.append('/Users/hannesvdc/hannes_phd/Source/openmm_protein/6kn2/internal/')
 
from martini_openmm.martini_openmm.martini import *
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

import numpy as np
import numpy.linalg as lg
import scipy as sp

from Reconstruction import *


# Construct the minimization problem and solve it via scipy
def minimize(context, x0):
	x0 = toNumpyVector(x0)

	def f(x):
		y = toVec3Vector(x, 'length')
		context.setPositions(y)

		s = context.getState(getPositions=True, getEnergy=True)
		return s.getPotentialEnergy()._value

	def df(x):
		y = toVec3Vector(x, 'length')
		context.setPositions(y)

		s = context.getState(getPositions=True, getForces=True)
		return -toNumpyVector(s.getForces())

	result = sp.optimize.minimize(f, x0, jac=df, tol=1.e-7)
	return result

# Load the microscopic 6kn2 system in openmm
def loadSystem():
	pdbfilename = '/Users/hannesvdc/hannes_phd/Source/openmm_protein/6kn2/6kn2_input_files/charmm36m_ff/6kn2.pdb'
	pdbfile = PDBFile(pdbfilename)
	forcefield = ForceField('amber99sb.xml')
	system = forcefield.createSystem(pdbfile.topology)

	integrator = BrownianIntegrator(1.0, 1.0, 1.0) # Fake and irrelevant integrator
	context = Context(system, integrator, Platform.getPlatformByName("Reference"))
	positions = pdbfile.getPositions()
	masses = [system.getParticleMass(i)._value for i in range(system.getNumParticles())]
	context.setPositions(positions)

	return context, positions, masses


if __name__ == '__main__':
	context, x0, masses = loadSystem()
	result = minimize(context, x0)
	x_min = result.x

	print('success', result.success)
	print('error', lg.norm(result.jac))
	print('Solution', lg.norm(result.x))

	np.save('internal/LocalMinimum', x_min)



