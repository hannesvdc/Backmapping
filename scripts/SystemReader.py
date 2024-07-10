from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

import argparse as argparse

def parseArguments():
    parser = argparse.ArgumentParser(description='Input arguments for the different order tests.')
    parser.add_argument('--pdbfile', nargs='?', dest='pdbfile', 
                        help='Select the pdb file from which this script loads the system.')
    parser.add_argument('--xml_filename', nargs='?', dest='xml_filename',
    					help="Choose the filename in which to store the system's serialization.")
    parser.add_argument('--txt_filename', nargs='?', dest='txt_filename',
    					help="Choose the filename in which to store the system's initial positions.")
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
	args = parseArguments()

	pdb = PDBFile(args.pdbfile)
	forcefield = ForceField('amber99sb.xml')
	system = forcefield.createSystem(pdb.topology)

	# Storing the system in xml format.1302
	string = XmlSerializer.serializeSystem(system)
	file = open(args.xml_filename, "w")
	file.write(string)
	file.close()

	# Storing the initial positions in plain text format.
	positions = pdb.getPositions()
	file2 = open(args.txt_filename, "w")
	for i in range(len(positions)):
		file2.write(str(positions[i][0]._value)+ " " + str(positions[i][1]._value) + " " + str(positions[i][2]._value)+"\n")
