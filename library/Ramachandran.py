from simtk.openmm import *
from simtk.unit import *

import numpy as np

def cross(a, b):
	try:
		_x = a[1]._value*b[2]._value - a[2]._value*b[1]._value
		_y = a[2]._value*b[0]._value - a[0]._value*b[2]._value
		_z = a[0]._value*b[1]._value - a[1]._value*b[0]._value
	except:
		_x = a[1]*b[2] - a[2]*b[1]
		_y = a[2]*b[0] - a[0]*b[2]
		_z = a[0]*b[1] - a[1]*b[0]

	v = openmm.Vec3(Quantity(_x, nanometer), Quantity(_y, nanometer), Quantity(_z, nanometer))
	return v

def dot(a, b):
	try:
		return a[0]._value*b[0]._value + a[1]._value*b[1]._value + a[2]._value*b[2]._value
	except:
		return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

class Ramachandran:
	def __init__(self, amino_acids):
		self.amino_acids = amino_acids

		self.lens = {
			"GLY": 9,
			"PHE": 20,
			"ARG": 24,
			"SER": 11,
			"PRO": 14,
			"CYS": 10
		}

	def computeDihedralsNumpy(self, x):
		y = []
		for i in range(x.size // 3):
			v = openmm.Vec3(x[3*i+0], x[3*i+1], x[3*i+2])
			y.append(v)

		return self.computeDihedrals(y)

	def computeDihedrals(self, x):
		rams = []

		total_len = 0
		for i in range(len(self.amino_acids)):
			N = x[total_len + 0]
			Ca= x[total_len + 1]
			C = x[total_len + 2]

			if i == 0:
				Cprev = x[4]
			else:
				Cprev = x[total_len - self.lens[self.amino_acids[i-1]] + 2] # Previous C atom

			if i == len(self.amino_acids)-1:
				Nnext = x[total_len + 4]
			else:
				Nnext = x[total_len + self.lens[self.amino_acids[i]] + 0]

			phi = self.torsion(Cprev, N, Ca, C)
			psi = self.torsion(N, Ca, C, Nnext)
			rams.append(phi._value)
			rams.append(psi._value)

			total_len = total_len + self.lens[self.amino_acids[i]]
		
		return np.array(rams)
	
	def torsion(self, A, B, C, D):
		b1 = A - B
		b2 = C - B
		b3 = D - C

		n1 = cross(b1, b2)
		n1 = n1/sqrt(dot(n1, n1))
		n2 = cross(b2, b3)
		n2 = n2/sqrt(dot(n2, n2))
		m = cross(n1, b2/sqrt(dot(b2, b2)))
        
		xp = dot(n1, n2)
		yp = dot(m, n2)
		phi = atan2((-1)*yp, (-1)*xp)

		return -phi

	def torsion_sq_dist(self, tau1, tau2):
		dist = 0.0
		for i in range(tau1.size):
			d = np.abs(tau1[i] - tau2[i])
			if d > np.pi:
				d = 2.0*np.pi - d

			dist += d*d

		return dist/tau1.size
	
