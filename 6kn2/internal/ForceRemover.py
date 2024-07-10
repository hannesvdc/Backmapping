import sys
sys.path.append('/Users/hannesvdc/hannes_phd/Source/openmm_protein/6kn2/')

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

from Reconstruction import *

import itertools
import numpy.random as rd
import numpy.linalg as lg

def removeWaterForces(system, context):
	for index in range(len(system.getForces())):
		f = system.getForce(index)

		if isinstance(f, CustomNonbondedForce):
			print("CustomNonbondedForce")
			if f.getEnergyFunction().startswith("step(rcut-r) * ES"): # Elektrostatic force
				for i in range(24, system.getNumParticles()):
					f.setParticleParameters(i, [0.0])
			else: # LJ force
				#for i in range(24, system.getNumParticles()):
				#	f.setParticleParameters(i, [0.0])
				for i in range(311):
					for j in range(24, 311):
						f.addExclusion(i, j)
				for j in range(311):
					for i in range(24, 311):
						f.addExclusion(i, j)
			f.updateParametersInContext(context)

		elif isinstance(f, CustomBondForce):
			print("CustomBondForce")
			if f.getEnergyFunction().startswith("step(rcut-r) * ES"):
				for i in range(f.getNumBonds()):
					p1, p2, p = f.getBondParameters(i)
					if p1 >= 24 or p2 >= 24:
						f.setBondParameters(i, p1, p2, [0.0])
			else:
				for i in range(f.getNumBonds()):
					p1, p2, p = f.getBondParameters(i)
					if p1 >= 24 or p2 >= 24:
						f.setBondParameters(i, p1, p2, [0.0, 0.0])
			f.updateParametersInContext(context)

		elif isinstance(f, HarmonicAngleForce):
			print("HarmonicAngleForce")
			for i in range(f.getNumAngles()):
				p1, p2, p3, a, k = f.getAngleParameters(i)
				if p1 >= 24 or p2 >= 24 or p3 >= 24:
					f.setAngleParameters(i, p1, p2, p3, a, 0.0)
			f.updateParametersInContext(context)

		elif isinstance(f, HarmonicBondForce):
			print("HarmonicBondForce")
			for i in range(f.getNumBonds()):
				p1, p2, r, k = f.getBondParameters(i)
				if p1 >= 24 or p2 >= 24:
					f.setBondParameters(i, p1, p2, r, 0.0)
			f.updateParametersInContext(context)

		elif isinstance(f, CustomAngleForce):
			print("CustomAngleForce")
			for i in range(f.getNumAngles()):
				p1, p2, p3, p = f.getAngleParameters(i)
				if p1 >= 24 or p2 >= 24 or p3 >= 24:
					f.setAngleParameters(i, p1, p2, p3, [p[0], 0.0])
			f.updateParametersInContext(context)

		elif isinstance(f, CustomTorsionForce):
			print("CustomTorsionForce")
			for i in range(f.getNumTorsions()):
				p1, p2, p3, p4, p = f.getTorsionParameters(i)
				if p1 >= 24 or p2 >= 24 or p3 >= 24 or p4 >= 24:
					f.setPerTorsionParameters(i, p1, p2, p3, p4, [p[0], 0.0])
			f.updateParametersInContext(context)
		else:
			print("CMMotionRemover")

	return system, context

def testWaterForces(system, context, pos):
	zero = [Vec3(Quantity(0.0, nanometer), Quantity(0.0, nanometer), Quantity(0.0, nanometer))]
	one =  [Vec3(Quantity(1.0, nanometer), Quantity(1.0, nanometer), Quantity(1.0, nanometer))]
	w1 = toVec3Vector(rd.randn(3*287), 'length') #list(itertools.chain.from_iterable(itertools.repeat(x, 287) for x in zero))
	w2 = toVec3Vector(rd.randn(3*287), 'length') #list(itertools.chain.from_iterable(itertools.repeat(x, 287) for x in one))

	z1 = pos.copy()
	z1.extend(w1)
	z2 = pos.copy()
	z2.extend(w2)

	context.setPositions(z1)
	s1 = context.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True)
	context.setPositions(z2)
	s2 = context.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True)

	print("Printing forces and potential energy")
	print(s1.getForces())
	print(s2.getForces())
	print(lg.norm(toNumpyVector(s1.getForces()) - toNumpyVector(s2.getForces())))
	print(s1.getPotentialEnergy(), s2.getPotentialEnergy())

def printWrongForces(system, context):
	print("Printing wrong forces")
	for index in range(len(system.getForces())):
		f = system.getForce(index)
		print(type(f))

		if isinstance(f, CustomNonbondedForce) and f.getEnergyFunction().startswith("step(rcut-r) * ES;"):
			name = "CustomNonbondedForce"
			nforces = f.getNumParticles()
			p_name = f.getPerParticleParameterName(0)
			for i in range(nforces):
				param = f.getParticleParameters(i)
				#if i >= 24:
				print("Elektrostatic", name, ", Particle", i, ":", p_name, "=", param[0])
		elif isinstance(f, CustomNonbondedForce) and f.getEnergyFunction().startswith("step(rcut-r)*(energy - corr);"):
			name = "CustomNonbondedForce"
			nforces = f.getNumParticles()
			p_name = f.getPerParticleParameterName(0)
			for i in range(nforces):
				param = f.getParticleParameters(i)
				#if i >= 24:
				print("Lennard-Jones", name, ", Particle", i, ":", p_name, "=", param[0])
		elif isinstance(f, CustomBondForce) and f.getEnergyFunction().startswith("step(rcut-r) * ES;"):
			name = "CustomBondForce"
			nforces = f.getNumBonds()
			p_name = f.getPerBondParameterName(0)
			for i in range(nforces):
				p1, p2, params = f.getBondParameters(i)
				#if p1 >= 24 or p2 >= 24:
				print("Elektrostatic", name, ", p1=", p1, "p2 =", p2, ":",  p_name,"=", params[0])
		elif isinstance(f, CustomBondForce) and f.getEnergyFunction().startswith("step(rcut-r) * (energy - corr);"):
			name = "CustomBondForce"
			nforces = f.getNumBonds()
			p_name1 = f.getPerBondParameterName(0)
			p_name2 = f.getPerBondParameterName(1)
			for i in range(nforces):
				p1, p2, params = f.getBondParameters(i)
				#if p1 >= 24 or p2 >= 24:
				print("Lennard-Jones", name, ", p1=", p1, "p2 =", p2, ":",  p_name1, p_name2, "=", params[0], params[1])
		elif isinstance(f, HarmonicAngleForce):
			#print("Harmonic Abgle force")
			name = "HarmonicAngleForce"
			p_name1 = "theta0"
			p_name2 = "k"
			for i in range(f.getNumAngles()):
				p1, p2, p3, a, k = f.getAngleParameters(i)
				if p1 >= 24 or p2 >= 24 or p3 >= 24:
					print(name, ", p1=", p1, "p2 =", p2, "p3=", p3, ":",  p_name1, p_name2, "=", a, k)
		elif isinstance(f, HarmonicBondForce):
			print("Harmonic Bond forced")
			name = "HarmonicBondForce"
			p_name1 = "r0"
			p_name2 = "k"
			print("n harmoic bonds", f.getNumBonds())
			for i in range(f.getNumBonds()):
				p1, p2, r, k = f.getBondParameters(i)
				if p1 >= 24 or p2 >= 24:
					print(name, ", p1=", p1, "p2 =", p2,":",  p_name1, p_name2, "=", r, k)
		elif isinstance(f, CustomAngleForce):
			print("Custo, Angle Force")
			name = "CustomAngleForce"
			p_name1 = f.getPerAngleParameterName(0)
			p_name2 = f.getPerAngleParameterName(1)
			for i in range(f.getNumAngles()):
				p1, p2, p3, params = f.getAngleParameters(i)
				if p1 >= 24 or p2 >= 24 or p3 >= 24:
					print(name, ", p1=", p1, "p2 =", p2, "p3=", p3, ":",  p_name1, p_name2, "=", params[0], params[1])
		elif isinstance(f, CustomTorsionForce):
			name = "CustomTorsionForce"
			p_name1 = f.getPerTorsionParameterName(0)
			p_name2 = f.getPerTorsionParameterName(1)
			for i in range(f.getNumTorsions()):
				p1, p2, p3, p4, params = f.getTorsionParameters(i)
				if p1 >= 24 or p2 >= 24 or p3 >= 24 or p4 >= 24:
					print(name, ", p1=", p1, "p2 =", p2, "p3=", p3, "p4=", p4, ":",  p_name1, p_name2, "=", params[0], params[1])
