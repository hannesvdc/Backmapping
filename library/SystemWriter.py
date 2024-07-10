from simtk.openmm.app import *
from simtk.openmm import *
from simtk.openmm.vec3 import *
from simtk.unit import *
import parmed.gromacs as gro
import parmed as pmd
from argparse import *

import numpy as np

def readPositions(filet):
	file = open(filet, "r")
	arr = np.fromstring(file.readline(), sep=' ', dtype=float)
	oarr = []
	for i in range(0, len(arr), 3):
		v = Vec3(10.*arr[i], 10.*arr[i+1], 10.*arr[i+2])
		oarr.append(v)
	return oarr[0:len(oarr)-3]


def parseArguments():
    parser = ArgumentParser(description='Input arguments for the different order tests.')
    parser.add_argument('--type', nargs='?', dest='type', default="",
                        help='Select the pdb file from which this script loads the system.')
    
    args = parser.parse_args()
    return args

def main():
	args = parseArguments()
	reference = PDBFile('/Users/hannesvdc/Nextcloud/molecular_data/6kn2_input_files/charmm36m_ff/6kn2_clean.pdb')

	write_directory = '/Users/hannesvdc/Nextcloud/molecular_data/pdb_files/'
	read_directory = '/Users/hannesvdc/hannes_phd/Source/openmm_protein/6kn2/data/'
	if args.type == 'micro':
		for i in range(20):
			positions = readPositions(read_directory+'micro_'+str(i+1)+'.pdb')
			file = open(write_directory+'micro_t='+str((i+1)*10)+'ps.pdb', 'w')
			reference.writeFile(reference.topology, positions, file)§
	elif args.type == 'mM':
		for i in range(20):
			positions = readPositions(read_directory+'mM_'+str(i+1)+'.pdb')
			file = open(write_directory+'mM_t='+str((i+1)*10)+'ps.pdb', 'w')
			reference.writeFile(reference.topology, positions, file)
	else:
		print("This type of operation is not supported.")
			

if __name__ == '__main__':
	main()
	