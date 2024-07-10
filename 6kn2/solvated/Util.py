import simtk.openmm as openmm
import simtk.unit as unit

import numpy as np

# Conversion from [Vec3*] to numpy array
def toNumpyArray(a):
	n = []
	for i in range(len(a)):
		n.append(a[i][0]._value)
		n.append(a[i][1]._value)
		n.append(a[i][2]._value)

	return np.array(n)

# Conversion from numpy array to [Vec3*]
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

def getMasses(system):
    masses = np.zeros(system.getNumParticles())
    for i in range(system.getNumParticles()):
        masses[i] = system.getParticleMass(i)._value
		
    return masses

def sampleVelocities(masses, rng):
    v = []
    for i in range(len(masses)):
        v1 = rng.normal(0.0, 1.0/np.sqrt(masses[i]))
        v2 = rng.normal(0.0, 1.0/np.sqrt(masses[i]))
        v3 = rng.normal(0.0, 1.0/np.sqrt(masses[i]))
 
        v.extend([v1, v2, v3])
    v = np.array(v)
 
    return v