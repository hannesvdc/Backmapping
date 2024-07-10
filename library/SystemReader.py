from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import parmed.gromacs as gro
import parmed as pmd
from argparse import *

def parseArguments():
    parser = ArgumentParser(description='Input arguments for the different order tests.')
    parser.add_argument('--pdbfile', nargs='?', dest='pdbfile', default="",
                        help='Select the pdb file from which this script loads the system.')
    parser.add_argument('--filename', nargs='?', dest='filename',
    					help="Choose the filename in which to store the system's serialization.")
    parser.add_argument('--positions', nargs='?', dest='positions_file')
    
    args = parser.parse_args()
    return args

def main(args):
	pdb = PDBFile(args.pdbfile)
	positions = pdb.getPositions();
	forcefield = ForceField('amber99sb.xml')
	system = forcefield.createSystem(pdb.topology)

	string = XmlSerializer.serialize(system)
	file = open(args.filename, "w")
	file.write(string)
	file.close()

	posfile = open(args.positions_file, "w")
	pdbfile = open(args.pdbfile, "r")
	lines = pdbfile.readlines()
	pdbfile.close()

	w_lines = []
	for line in lines[1:len(lines)-1]:
		print('line ', line)
		l = line.split()
		w_lines.append(l[6]+' '+l[7]+' '+l[8]+' ')
	print('to write lines', w_lines)
	posfile.writelines(w_lines)
#	posfile.writelines(['here', 'idiot'])
	posfile.close()
	

if __name__ == '__main__':
	args = parseArguments()
	main(args)
