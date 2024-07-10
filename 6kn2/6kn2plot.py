import sys
from os.path import exists
sys.path.append('../library')


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import scipy.stats as stat
from sklearn.neighbors import KernelDensity
import argparse as argparse


from Ramachandran import *
from Reconstruction import *


def parseArguments():
    parser = argparse.ArgumentParser(description='Input arguments for the different order tests.')
    parser.add_argument('--plot', nargs='?', dest='plot',
                        help='Select the type of plot to make.')
    parser.add_argument('--file', nargs='?', dest='file',
    					help='Select the data file from which to extract the data')
    
    args = parser.parse_args()
    return args

def ramachandranPlot(file_name):
	amino_acids = ["GLY", "PHE", "ARG", "SER", "PRO", "CYS", "PRO", "PRO", "PHE", "CYS"]

	if file_name.endswith('.npy'):
		torsion_values = np.load(file_name, allow_pickle=True)
		print('shape', torsion_values.shape, torsion_values[1, 1])
	else:
		file = open(file_name, "r")
		line = file.readline()

		torsion_values = []

		while line != "":
			torsion_values.append(np.fromstring(line, dtype=float, sep=' '))
			line = file.readline()
		torsion_values = np.array(torsion_values)

	print(torsion_values.shape)
	gamma = 0.3
	for i in range(0, torsion_values.shape[1], 2):
		plt.figure()
		plt.hist2d(torsion_values[:, i], torsion_values[:, i+1], range=[[-np.pi, np.pi], [-np.pi, np.pi]], bins=1000, norm=mcolors.PowerNorm(gamma))
		plt.xlabel(r"$\phi$")
		plt.ylabel(r"$\psi$")
		plt.title(amino_acids[int(i/2.0)]);
	plt.show()

def contourPlot(file_name):
	def flip(two_array):
		for i in range(two_array.shape[1]):
			u = two_array[0,i]
			two_array[0,i] = two_array[1,i]
			two_array[1,i] = u

		return two_array

	amino_acids = ["GLY", "PHE", "ARG", "SER", "PRO", "CYS", "PRO", "PRO", "PHE", "CYS"]
	ram = Ramachandran(amino_acids)

	data = np.load(file_name, allow_pickle=True)
	n_data = data.shape[1]
	n_cols = data.shape[0] // 2
	micro_samples = data[0:n_cols,:]
	micro_forces = data[n_cols:,:]

	# Compute torsion values
	ram_values = np.zeros((20, n_data))
	for i in range(n_data):
		if i % 1000 == 0:
			print(i)
		ram_values[:,i] = ram.computeDihedrals(toVec3Vector(micro_samples[:,i], 'length'))

	# Reflect data for accuracy
	#ram_values[17,:] = np.copy(-ram_values[17,:])

	# Create 2-dimensional density plots
	for i in range(10):
		data_set = np.zeros((2, n_data))
	# 	if i == 8:
	# 		data_set[0,:] = (ram_values[2*i, :]  % (2.0*np.pi)) - np.pi
	# 		data_set[1,:] = ram_values[2*i+1,:]
	# 	elif i == 5:
	# 		data_set[0,:] = ram_values[2*i,:]
	# 		data_set[1,:] = ((-ram_values[2*i+1,:]) % (2.0*np.pi)) - np.pi
	# 	elif i == 4:
	# 		data_set[0,:] = -ram_values[2*i+1,:]
	# 		data_set[1,:] = ram_values[2*i,:]
	# 	elif i == 2:
	# 		data_set[0,:] = ram_values[2*i,:]
	# 		data_set[1,:] = (ram_values[2*i+1,:] % (2.0*np.pi)) - np.pi

	# 		rot_matrix = np.array([[-1.0, 0],[0.0, -1.0]])
	# 		data_set = np.matmul(rot_matrix, data_set)
	# 	elif i == 1:
	# 		data_set[0,:] = -ram_values[2*i,:]
	# 		data_set[1,:] = ram_values[2*i+1,:]

	# 		data_set[0,:] = (data_set[0,:] % (2.0*np.pi)) - np.pi

	# 		rot_matrix = np.array([[0.0, 1.0],[-1.0, 0.0]]) # 90 degrees
	# 		data_set = np.matmul(rot_matrix, data_set)
	# 	else:
		data_set[:,:] = np.copy(ram_values[2*i:2*(i+1),:])
		
		print(data_set.shape)
		X, Y = np.mgrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
		positions = np.vstack([X.ravel(), Y.ravel()])
		kernel = stat.gaussian_kde(data_set)
		Z = np.reshape(kernel(positions).T, X.shape)

		plt.figure()
		plt.contour(X, Y, Z)
		plt.plot([0.0, 0.0], [-np.pi, np.pi],  color='gray')
		plt.plot([-np.pi, np.pi], [0.0, 0.0],  color='gray')
		plt.xlim((-np.pi, np.pi))
		plt.ylim((-np.pi, np.pi))

	plt.show()

def scatterPlot(file_name):
    amino_acids = ["GLY", "PHE", "ARG", "SER", "PRO", "CYS", "PRO", "PRO", "PHE", "CYS"]
    ram = Ramachandran(amino_acids)

    n_data = 0
    n_cols = 0
    ram_file_name = 'data/Ramachandran'+file_name[15:]
    if exists(ram_file_name):
        ram_values = np.load(ram_file_name)
        n_data = max(ram_values.shape)
        n_cols = min(ram_values.shape)
        print(ram_values.shape)
    else:
        data = np.load(file_name, allow_pickle=True)
        n_data = data.shape[1]
        n_cols = data.shape[0] // 2
        micro_samples = data[0:n_cols,:]
        micro_forces = data[n_cols:,:]

        # Compute torsion values
        ram_values = np.zeros((20, n_data))
        for i in range(n_data):
            if i % 1000 == 0:
                print(i)
            ram_values[:,i] = ram.computeDihedrals(toVec3Vector(micro_samples[:,i], 'length'))

        np.save(ram_file_name, ram_values)
    print(n_data,  n_cols, ram_values.shape)

    # Plot torsion values 
    #for i in range(0, ram_values.shape[0], 2):
    for i in range(2, 3):
        # Make Gaussian density
        print('\ni =', i)
        data_set = np.copy(ram_values[i:i+2,:])
        # i = 16
        # for j in range(data_set.shape[1]):
        # 	r = data_set[0,j] - np.pi
        # 	if r < -np.pi:
        # 		r = r + 2.0*np.pi
        # 	data_set[0,j] = r

        # i = 14
        #R = np.array([[-1.0, 0.0], [0.0,  -1.0]])
        #data_set = np.matmul(R, data_set)

        # i = 12
        #R = np.array([[0.0,  1.0],  [-1.0, 0.0]])
        #data_set = np.matmul(R, data_set)

        # i = 8
        #R = np.array([[0.0,  1.0],  [-1.0, 0.0]])
        #data_set = np.matmul(R, data_set)

        # i = 6
        #data_set[0,:] = -data_set[0,:]

        # i = 4
        #for j in range(data_set.shape[1]):
        # 	r = data_set[1,j] + np.pi
        #	if r > np.pi:
        #		r = r - 2.0*np.pi
        #	data_set[1,j] =  r

        # i = 2
        R = np.array([[0.0,  1.0],  [-1.0, 0.0]])
        data_set = np.matmul(R, data_set)
        data_set[1,:] = -data_set[1,:]
        for j in range(data_set.shape[1]):
        	rr = data_set[1,j] + np.pi
        	if rr > np.pi:
        		rr = rr - 2.0*np.pi
        	data_set[1,j] = rr
        R = np.array([[-1.0, 0.0], [0.0,  -1.0]])
        data_set = np.matmul(R, data_set)


        X, Y = np.mgrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        kernel = stat.gaussian_kde(data_set)
        Z = np.reshape(kernel(positions).T, X.shape)

        plt.figure()
        plt.scatter(data_set[0,:], data_set[1,:])
        plt.xlim((-np.pi, np.pi)) 
        plt.ylim((-np.pi, np.pi))
        plt.xlabel(r"$\phi$")
        plt.ylabel(r"$\psi$")
        plt.title(amino_acids[int(i/2.0)])

        print(X.shape, Y.shape, Z.shape)
        plt.contour(X, Y, Z)
        plt.plot([0.0, 0.0], [-np.pi, np.pi],  color='gray')
        plt.plot([-np.pi, np.pi], [0.0, 0.0],  color='gray')
        plt.xlim((-np.pi, np.pi))
        plt.ylim((-np.pi, np.pi))
    plt.show()


if __name__ == "__main__":
	args = parseArguments()

	if args.plot == 'ramachandranPlot':
		ramachandranPlot(args.file)
	elif args.plot == "scatterPlot":
		scatterPlot(args.file)
	elif args.plot == 'contourPlot':
		contourPlot(args.file)
	else:
		print('This type of plot is not supported. Choose another.')
