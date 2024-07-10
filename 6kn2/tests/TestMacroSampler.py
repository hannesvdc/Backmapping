import sys
sys.path.append('/Users/hannesvdc/hannes_phd/Projects/openmm_protein/library/')
sys.path.append('/Users/hannesvdc/hannes_phd/Projects/openmm_protein/6kn2/internal/')
sys.path.append('../')

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

import itertools
import time
import argparse as argparse

from MacroscopicSampler import *
from AddResS import *
import numpy.linalg as lg

def debugMacroSamplers():
	seed = 1233

	samples_address, ln_address = AddResS(seed=seed)
	samples_macro, ln_macro = optimalMacroscopicStep(seed=seed)

	print(samples_address[:,0], ' |', samples_macro[:,0])
	for i in range(1, samples_macro.shape[1]):
		input('')
		
		print('\n', ln_address[i-1], ' |', ln_macro[i-1])
		print(samples_address[:,0], ' |', samples_macro[:,0])


if __name__ == '__main__':
	debugMacroSamplers()
