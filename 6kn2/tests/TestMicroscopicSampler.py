import sys
sys.path.append('/Users/hannesvdc/hannes_phd/Projects/openmm_protein/library/')
sys.path.append('/Users/hannesvdc/hannes_phd/Projects/openmm_protein/6kn2/internal/')
sys.path.append('../')

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

import itertools
import time
import argparse as argparse

from Reconstruction import *
from MARTINI import *
from Ramachandran import *
from martini_openmm.martini_openmm.martini import *
from MicroscopicSampler import *
import numpy.linalg as lg

def runMicroscopicSampler():
	temperature = 3000.0
	beta = 1000.0/(temperature*8.3145)
	N = 10**6
	lam = 1.0

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
	#extension = 'pdb'
	#dt = 2.e-3
	# Option 2: positions = local minimum
	x0 = toVec3Vector(np.load('../internal/LocalMinimum.npy'), 'length')
	extension = 'minimum'
	dt = 2.e-3
	z0 = toVec3Vector(rc.value(x0), 'length')
	microContext.setPositions(x0)
	macroContext.setPositions(z0)
	print("------------- SETUP DONE ---------------")

	rng = rd.RandomState(seed=int(time.time()))
	micro_samples = np.zeros((3*len(x0), N))
	n_accepted = 0.0
	for k in range(1, N+1):
		if k % 1000 == 0:
			print(k, n_accepted / k)
		macroState = macroContext.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True)
		microState = microContext.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True)
		micro_samples[:, k-1] = toNumpyVector(microState.getPositions())

		xp, zp, lnalpha = microStep(macroContext, microContext, rc, masses, lam, dt, beta, rng)
		if np.log(rng.uniform()) <= lnalpha: # If accepted : keep current state and continue
			n_accepted += 1.0
		else: # If rejected : reset to previous state and continue
			microContext.setState(microState)
			macroContext.setState(macroState)

	print("Total acceptance rate with dt =", dt, ":", n_accepted/N)
	print('Storing results')
	np.save('MicroSamples_'+extension+'.npy', micro_samples)

def plotMicroscopicRamachandran():
	file_name = 'MicroSamples_minimum.npy'
	amino_acids = ["GLY", "PHE", "ARG", "SER", "PRO", "CYS", "PRO", "PRO", "PHE", "CYS"]
	rc = MARTINI(amino_acids, 149)
	ram = Ramachandran(amino_acids)

	print("Computing Dihedral Angles.")
	if file_name.endswith('.npy'):
		micro_samples = np.load(file_name, allow_pickle=True)
		ram_values = np.zeros((20, micro_samples.shape[1]))
		for k in range(0, micro_samples.shape[1]):
			if k % 1000 == 0:
				print(k)
			x = micro_samples[:,k]
			ram_values[:,k] = ram.computeDihedrals(toVec3Vector(x, 'length'))
	else:
		file = open(file_name, "r")
		line = file.readline()

		ram_values = []
		while line != "":
			ram_values.append(np.fromstring(line, dtype=float, sep=' '))
			line = file.readline()
		ram_values = np.array(ram_values)

	print("Plotting...")
	for i in range(0, 10):
		plt.figure()
		plt.scatter(ram_values[2*i, :], ram_values[2*i+1,:])
		plt.xlim((-np.pi, np.pi))
		plt.ylim((-np.pi, np.pi))
		plt.xlabel(r"$\phi$")
		plt.ylabel(r"$\psi$")
		plt.title(amino_acids[int(i)]);
	plt.show()

def parseArguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--routine', dest='routine', nargs='?')

	return parser.parse_args()

if __name__ == '__main__':
	args = parseArguments()

	if args.routine == 'TestMicroSampler':
		runMicroscopicSampler()
	elif args.routine == 'PlotMicroRamachandran':
		plotMicroscopicRamachandran()
	else:
		print('Routine is not supported. Abort.')
