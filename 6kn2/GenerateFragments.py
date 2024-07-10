import sys
sys.path.append("/Users/hannesvdc/hannes_phd/Projects/openmm_protein")

import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

from simtk import openmm, unit
from simtk.openmm import app
from library.MARTINI import MARTINI

import itertools
import warnings
warnings.filterwarnings("error")

def toNumpyVector(a):
	n = []
	for i in range(len(a)):
		n.append(a[i][0]._value)
		n.append(a[i][1]._value)
		n.append(a[i][2]._value)

	return np.array(n)

def toVec3Vector(a, u):
	if u == 'length':
		un = unit.nanometer
	elif u == 'time':
		un = unit.picosecond
	elif u  == 'velocity':
		un = unit.nanometer/unit.picosecond
	elif u == 'force':
		un = unit.kilojoules_per_mole/unit.nanometer

	vec3array = []
	for i in range(int(len(a)/3.0)):
		x = openmm.Vec3(unit.Quantity(a[3*i], un), unit.Quantity(a[3*i+1], un), unit.Quantity(a[3*i+2], un))
		vec3array.append(x)

	return vec3array

def sampleNormal(rng, masses, beta):
	x = np.zeros(masses.size)
	for i in range(masses.size):
		x[i] = rng.normal(0.0, 1.0/np.sqrt(beta*masses[i]))

	return toVec3Vector(x, 'velocity')

def toString(x):
	line = ''
	for i in range(len(x)):
		line += str(x[i][0]._value) + ' ' + str(x[i][1]._value) + ' ' + str(x[i][2]._value) + ' '
	line = line[0:len(line)-1] + '\n'

	return line

def microSimulation():
	pdb = app.PDBFile('/Users/hannesvdc/hannes_phd/Source/openmm_protein/6kn2/cpp/forconx/6kn2.pdb')
	positions = pdb.getPositions() #vec3
	forcefield = app.ForceField('amber99sb.xml') #amber99sb.xml
	system = forcefield.createSystem(pdb.topology)
	rng = rd.RandomState()

	temperature = 3000.0
	beta = 1000.0/(temperature*8.3145)

	integrator = openmm.BrownianIntegrator(temperature, 1.0, 1.0)
	context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
	context.setPositions(positions)
	print("Natoms = ", len(positions))

	aminoAcids = ['GLY', 'PHE', 'ARG', 'SER', 'PRO', 'CYS', 'PRO', 'PRO', 'PHE', 'CYS']
	martini = MARTINI(aminoAcids, len(positions))

	masses = []
	for i in range(0, len(positions)):
		masses.append(system.getParticleMass(i)._value)
	masses = np.array(list(itertools.chain.from_iterable(itertools.repeat(x, 3) for x in masses)))
	
	print("------------- SETUP DONE ---------------")

	dt = 1.e-3
	N = 10000
	naccepted = 0.0
	x = toNumpyVector(positions)
	for i in range(1, N+1):
		if i % 1000 == 0:
			print(i)

		v = toNumpyVector(sampleNormal(rng, masses, beta))
		context.setVelocities(toVec3Vector(v, 'velocity'))
		s = context.getState(True, True, True, True)
		f = toNumpyVector(s.getForces())
		E = s.getPotentialEnergy()._value + s.getKineticEnergy()._value

		v12 = v + 0.5*dt*f/masses
		xp = x + dt*v12
		context.setPositions(toVec3Vector(xp, 'length'))
		sp = context.getState(True, True, True, True)
		fp = toNumpyVector(sp.getForces())

		v1 = v12 + 0.5*dt*fp/masses
		context.setVelocities(toVec3Vector(v1, 'velocity'))
		sp = context.getState(True, True, True, True)
		Ep = sp.getPotentialEnergy()._value + sp.getKineticEnergy()._value

		lnalpha = -beta*(Ep - E)
		#print('lnalpha', lnalpha)
		if np.log(rng.uniform()) <= lnalpha:
			x = xp
			naccepted += 1.0
		else:
			context.setState(s)

	print("------------- SIMULATION DONE ---------------")
	print("Total Acceptance Rate: ", naccepted/N)

	x = toVec3Vector(x, 'length')
	print('x',toNumpyVector(x))
	indexMap = martini.AminoAcidCGIndices
	sizes = martini.AminoAcidSizes
	cg_sizes = martini.AminoAcidCGSizes
	cg_coordinates = martini.value(x)

	indices = []
	offsets = []

	filename = 'data/fragments.data'
	file = open(filename, 'w')
	offset = 0
	macro_index = 0
	for i in range(len(aminoAcids)):
		name = aminoAcids[i]
		micro_len = martini.AminoAcidSizes[name]
		n_cg_beads = martini.AminoAcidCGSizes[name]

		for j in range(n_cg_beads):
			indices.append(martini.AminoAcidCGIndices[name][j])
			offsets.append(offset)

			macro_index += 1

		offset +=  micro_len

	lines = []
	micro_line = toString(x); lines.append(micro_line)
	macro_line = toString(toVec3Vector(martini.value(x), 'length')); lines.append(macro_line)
	for i in range(len(offsets)):
		line = str(offsets[i])
		for j in range(len(indices[i])):
			line += ' ' + str(indices[i][j])
		lines.append(line + '\n')

	file.writelines(lines)
	file.close()

if __name__ == '__main__':
	microSimulation()
