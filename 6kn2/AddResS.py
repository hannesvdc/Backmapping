import sys
sys.path.append('/Users/hannesvdc/hannes_phd/Projects/openmm_protein/library/')
sys.path.append('/Users/hannesvdc/hannes_phd/Projects/openmm_protein/6kn2/internal/')
 
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
 
from sys import stdout
import itertools
import time
import datetime
 
from Reconstruction import *
from MacroscopicSampler import *
from MicroscopicSampler import *
from MARTINI import *
from Ramachandran import *
from ForceRemover import *
 
from martini_openmm.martini_openmm.martini import *
import numpy.linalg as lg
 
def setMasses(system):
    for i in range(24, system.getNumParticles()):
        system.setParticleMass(i, 0.0)
 
    return system
 
def getMasses(system):
    nParticles = system.getNumParticles()
 
    masses = []
    for i in range(nParticles):
        masses.append(system.getParticleMass(i)._value)
 
    return np.array(masses)
 
def computeCM(positions, masses):
    cm = 0*positions[0]
    total_mass = 0
    for i in range(len(positions)):
        cm = cm + positions[i]*masses[i]
        total_mass += masses[i]
 
    return cm / total_mass
 
def AddResS():
    # Nature constants and method parameters
    temperature = 300.0
    beta = 1000.0/(temperature*8.3145)
    N = 10**6
    K = 20
 
    # Load the reconstruction data
    x_frag, z_frag, offsets, indices = loadFragments()
    aminoAcids = ["GLY", "PHE", "ARG", "SER", "PRO", "CYS", "PRO", "PRO", "PHE", "CYS"]
 
    # Loading the microscopic system
    pdbfilename = '/Users/hannesvdc/hannes_phd/Projects/openmm_protein/6kn2/6kn2_input_files/charmm36m_ff/6kn2.pdb'
    pdbfile = PDBFile(pdbfilename)
    forcefield = ForceField('amber99sb.xml')
    system = forcefield.createSystem(pdbfile.topology)
 
    integrator = BrownianIntegrator(temperature, 1.0, 1.0)
    context = Context(system, integrator, Platform.getPlatformByName("Reference"))
    context.setPositions(pdbfile.getPositions())
    masses = [system.getParticleMass(i)._value for i in range(system.getNumParticles())]
 
    # Setting up the reaction coordinate and the randon number generator
    rng = rd.RandomState()
    rc = MARTINI(aminoAcids, len(pdbfile.getPositions()))
    ram = Ramachandran(aminoAcids)
 
    # Loading the macroscopic system
    includeDir =  '/Users/hannesvdc/hannes_phd/Projects/openmm_protein/6kn2/6kn2_input_files/martini_ff/'
    topfilename = '/Users/hannesvdc/hannes_phd/Projects/openmm_protein/6kn2/6kn2_input_files/martini_ff/6kn2_topol_cg.top'
    grofilename = '/Users/hannesvdc/hannes_phd/Projects/openmm_protein/6kn2/6kn2_input_files/martini_ff/6kn2_solvated_cg.gro'
 
    grofile = GromacsGroFile(grofilename)
    topfile = GromacsMartiniV2TopFile(topfilename, periodicBoxVectors=grofile.getPeriodicBoxVectors(), includeDir=includeDir)
    macroSystem = topfile.createSystem()
    macroContext = Context(macroSystem, BrownianIntegrator(1.0, 1.0, 1.0), Platform.getPlatformByName("Reference"))
    macroMasses = getMasses(macroSystem)
    print("------------- SETUP DONE ---------------")
 
    pos_type= 'avg_brownian'
    if pos_type == 'pdb':
        x = toNumpyVector(pdbfile.getPositions())
        z = rc.value_internal(x)
        dt = 1.0e-5
        Dt = 2.0e-4
    elif pos_type =='min':
        x = np.load('internal/LocalMinimum.npy')
        z = rc.value_internal(x)
        dt = 1.0e-5
        Dt = 1.0e-4
    elif pos_type == 'avg':
        x = np.load('data/average_position.npy')
        z = rc.value_internal(x)
        dt = 6.0e-6
        Dt = 2.0e-2
    elif pos_type == 'avg_brownian':
        x = np.load('data/average_position.npy')
        z = rc.value_internal(x)
        dt = 6.0e-6
        Dt = 1.0e-5
    macroContext.setPositions(toVec3Vector(z, 'length'))
    context.setPositions(toVec3Vector(x, 'length'))
    avg_position = np.zeros(x.size)
     
    macro_samples = np.zeros((z.size, N))
    micro_samples = np.zeros((x.size, N))
    micro_forces = np.zeros((x.size, N))
    n_macro_accepted = 0.0
    n_micro_accepted = 0.0
    t_start = time.time()
    t_end = t_start

    macro_samples[:,0] = np.copy(z)
    micro_samples[:,0] = np.copy(x)
    for n in range(1, N):
        if n % 100 == 0:
            print(n, n_macro_accepted/n)
            t_end = time.time()
 
            time_to_go = (N-n)/100.0 * (t_end - t_start)/(n/100.)
            if n >= 300:
                print("Time to go:", str(datetime.timedelta(seconds=time_to_go))+"\n")
 
        # Macro Step HMC
        #zp, lnalpha = macroStepHMC(macroContext, macroMasses, Dt, beta, rng) # numpy
        zp, lnalpha = macroStepBrownian(macroContext, macroMasses, Dt, beta, rng) # numpy
        if np.log(rng.uniform()) < min(0.0, lnalpha):
            n_macro_accepted += 1.0

        #if True:
            # Direct Reconstruction From zp
            xp = reconstructProtein(x_frag, z_frag, offsets, indices, toVec3Vector(zp, 'length'))
            context.setPositions(xp)
 
            # Backmapping Simulation
            for k in range(K+1):
                lam = (1.0*k)/K
                macroState = macroContext.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True)
                microState = context.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True)
                xp, zp, lnalpha = microStep(macroContext, context,rc, masses, lam**2, dt, beta, rng)
                if np.log(rng.uniform()) <= lnalpha: # Can we uberhaupt define an acceptance rate?
                    n_micro_accepted += 1.0 
 
                macroContext.setPositions(toVec3Vector(zp, 'length'))
                context.setPositions(toVec3Vector(xp, 'length'))
 
            x = np.copy(xp)
            z = np.copy(zp)
        else:
            macroContext.setPositions(toVec3Vector(z, 'length'))
            context.setPositions(toVec3Vector(x, 'length'))
 
        micro_samples[:,n] = np.copy(x)
        macro_samples[:,n] = np.copy(z)
        avg_position += x

    macro_acceptance_rate = n_macro_accepted/N
    micro_acceptance_rate = n_micro_accepted/((K+1)*n_macro_accepted)
    print("Total macroscopic acceptance rate", macro_acceptance_rate)
    print("Total microscopic acceptance rate", micro_acceptance_rate)
    print("Macroscopic Time step", Dt)
    #avg_position = avg_position / N
 
    print("\nStoring Results")
    print('Data samples',  micro_samples.shape)
    datafilename = 'data/mM_AddResS_N='+str(N)+'_K='+str(K)+'_Dt='+str(Dt)+'_T='+str(temperature) + '_pos='+pos_type+ '_macro=brownian'+'.npy'
    np.save(datafilename, np.vstack((micro_samples, micro_forces)))
    #np.save('data/average_position.npy', avg_position)
    print("Done.")
 
 
if __name__ == '__main__':
    AddResS()