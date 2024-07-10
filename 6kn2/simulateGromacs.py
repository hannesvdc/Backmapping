from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout

from martini_openmm.martini_openmm.martini import *

includeDir = '/Users/hannesvdc/hannes_phd/Source/openmm_protein/6kn2/6kn2_input_files/martini_ff/'
topfilename = '/Users/hannesvdc/hannes_phd/Source/openmm_protein/6kn2/6kn2_input_files/martini_ff/6kn2_topol_cg.top'
grofilename = '/Users/hannesvdc/hannes_phd/Source/openmm_protein/6kn2/6kn2_input_files/martini_ff/6kn2_solvated_cg.gro'

gro = GromacsGroFile(grofilename)
topfile = GromacsMartiniV2TopFile(topfilename, periodicBoxVectors=gro.getPeriodicBoxVectors(), includeDir=includeDir)
system = topfile.createSystem()

integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
simulation = Simulation(topfile.topology, system, integrator)
simulation.context.setPositions(gro.positions)
simulation.minimizeEnergy()
simulation.reporters.append(PDBReporter('output.pdb', 1000))
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True, temperature=True))
simulation.step(10000)

