import sys
sys.path.append("/Users/hannesvdc/hannes_phd/Projects/openmm_protein/")

from simtk.openmm.app import *
from simtk.openmm import *
import simtk
from simtk.unit import *

import  matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import scipy.optimize as op
import itertools
import math
import warnings
from library.MARTINI import *

warnings.filterwarnings("error")
np.set_printoptions(threshold=np.inf)

def toNumpyVector(a):
	n = []
	for i in range(len(a)):
		n.append(a[i][0]._value)
		n.append(a[i][1]._value)
		n.append(a[i][2]._value)

	return np.array(n)

def toVec3Vector(a, u):
	if u == 'length':
		un = simtk.unit.nanometer
	elif u == 'time':
		un = simtk.unit.picosecond
	elif u  == 'velocity':
		un = simtk.unit.nanometer/simtk.unit.picosecond
	elif u == 'force':
		un = simtk.unit.kilojoules_per_mole/simtk.unit.nanometer

	vec3array = []
	for i in range(int(len(a)/3.0)):
		x = Vec3(Quantity(a[3*i], un), Quantity(a[3*i+1], un), Quantity(a[3*i+2], un))
		vec3array.append(x)

	return vec3array

def fromString(x):
	split = x.split()
	x_list = []

	for i in range(int(len(split)/3.0)):
		x1 = float(split[3*i])
		x2 = float(split[3*i+1])
		x3 = float(split[3*i+2])

		x_list.append(Vec3(Quantity(x1, unit.nanometer), Quantity(x2, unit.nanometer), Quantity(x3, unit.nanometer)))

	return x_list

def toList(string):
	indices = []
	split = string.split()

	for i in range(len(split)):
		indices.append(int(split[i]))

	return indices

def loadFragments():
	filename = '/Users/hannesvdc/hannes_phd/Projects/openmm_protein/6kn2/data/fragments.data'
	file = open(filename, "r")

	amino_acids = ['GLY', 'PHE', 'ARG', 'SER', 'PRO', 'CYS', 'PRO', 'PRO', 'PHE', 'CYS']
	cg = MARTINI(amino_acids, 149)
	offsets = []
	indices = []

	x = fromString(file.readline())
	z = fromString(file.readline())
	line = file.readline()
	while line != '':
		index = toList(line)
		offsets.append(index[0])
		index.pop(0)
		indices.append(index)

		line = file.readline()
	file.close()

	#print('x', toNumpyVector(x))
	#print('z', z)
	#print('offsets', offsets)
	#print('indices', indices)
	return x, z, offsets, indices

def reconstructProtein(x, z, offsets, indices, zp):
	zdiff = []
	for i in range(len(z)):
		zdiff.append(zp[i] - z[i])
		#print(zdiff[i], z[i], zp[i])

	# Make a copy of x
	xp = []
	for i in range(len(x)):
		xp.append(x[i])

	for i in range(len(zdiff)):
		offset = offsets[i]
		index = indices[i]

		for j in range(len(index)):
			xp[offset + index[j]] += zdiff[i]

	return xp

if __name__ == '__main__':
	rng = rd.RandomState()
	x, z, offsets, indices = loadFragments()
	rc = MARTINI(['GLY', 'PHE', 'ARG', 'SER', 'PRO', 'CYS', 'PRO', 'PRO', 'PHE', 'CYS'], len(x))

	# M = rc.gradient(0)
	# Mp = np.transpose(M).dot(lg.inv(M.dot(np.transpose(M))))
	# print(M.shape)
	# print(lg.matrix_rank(M))
	# print(M.dot(Mp))

	max_norm = 0.0
	for i in range(1000000):
		if i % 1000 == 0:
			print(i)
		zp = toVec3Vector(rng.normal(0, 5, 3*len(z)), 'length')

		xp = reconstructProtein(x, z, offsets, indices, zp)
		no = lg.norm(rc.value(xp) - toNumpyVector(zp))
		#print('xp', toNumpyVector(xp))
		#print('rc', lg.norm(rc.value(xp) - toNumpyVector(zp)))

		if no > max_norm:
			max_norm = no

	print("Max norm", max_norm)
	

