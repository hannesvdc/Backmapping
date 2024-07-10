'''
Created on 19 Oct 2020

@author: Hannes Vandecasteele
'''
import numpy as np
import numpy.linalg as lg
import copy as copy

masses = []
atom_positions = []
cg = []


def add_up(positions, index):
#    print('Adding up', positions, index)
    if positions[-1] == len(atom_positions):
        return positions, False
    
    for i in range(index, len(positions)):
        positions[i] += 1
        
    return positions, True

def check(positions, atom_positions, masses, cg):
    mass_mid = np.zeros(3)
    sum_mass = 0.
    
    for i in range(len(positions)):
        mass_mid += masses[positions[i]]*atom_positions[positions[i]]
        sum_mass += masses[positions[i]]
        
    mass_mid /= sum_mass
    
    
    global min_distance
    if lg.norm(mass_mid - cg) < min_distance:
        global exact_pos
        global exact_mid
        min_distance = lg.norm(mass_mid - cg)
        exact_pos = np.copy(positions)
        exact_mid = np.copy(mass_mid)
        
    return False

def backtrack(positions, index, atom_positions, masses, cg):
    original_pos = copy.copy(positions)
    
#    print('Starting method with index=', index)
    if index < len(positions)-1:
        success, poss = backtrack(positions, index+1, atom_positions, masses, cg)
        
        while True:
            if success:
                return True, poss
            
            positions, add_able = add_up(original_pos, index)
            
            if not add_able:
                return False, original_pos
            
            success, poss = backtrack(positions, index+1, atom_positions, masses, cg)
            
    else:        
        i = positions[index]
        pos = copy.copy(positions)
        while i < len(atom_positions):
            pos[index] = i
            print(pos)
            success = check(pos, atom_positions, masses, cg)
            
            if success:
                return True, pos
            
            i += 1
            
        return False, original_pos

def getGlycene():
    atom_positions = []
    masses = []

    atom_positions.append(np.array([-1.582,   3.333,   1.196]))
    atom_positions.append(np.array([-0.312,   3.033,   0.510]))
    atom_positions.append(np.array([0.028,   4.076,  -0.523]))
    atom_positions.append(np.array([-0.866,   4.638,  -1.167]))
    atom_positions.append(np.array([-1.523,   4.257,   1.666]))
    atom_positions.append(np.array([-1.788,   2.606,   1.909]))
    atom_positions.append(np.array([-2.358,   3.357,   0.504]))
    atom_positions.append(np.array([0.482,   2.991,   1.240]))
    atom_positions.append(np.array([-0.398,   2.074,   0.024]))
    masses.append(14.0067)
    masses.append(12.0107)
    masses.append(12.0107)
    masses.append(15.999)
    masses.append(1.00784)
    masses.append(1.00784)
    masses.append(1.00784)
    masses.append(1.00784)
    masses.append(1.00784)
    
    cg = np.array([-0.791,   3.796,  0.035])
    
    return atom_positions, masses, cg

def getPhenylalanine():
    atom_positions = []
    atom_masses = []
    
    atom_positions.append(np.array([ 1.304,   4.356,  -0.680]))
    atom_positions.append(np.array([ 1.756,   5.327,  -1.624]))
    atom_positions.append(np.array([ 3.105,   4.921,  -2.167]))
    atom_positions.append(np.array([ 3.984,   4.522,  -1.410]))
    atom_positions.append(np.array([ 1.873,   6.683,  -0.937]))
    atom_positions.append(np.array([ 2.225,   7.801,  -1.856]))
    atom_positions.append(np.array([ 1.243,   8.479,  -2.547]))
    atom_positions.append(np.array([ 3.540,   8.174,  -2.024]))
    atom_positions.append(np.array([ 1.568,   9.515,  -3.395]))
    atom_positions.append(np.array([ 3.877,   9.209,  -2.871]))
    atom_positions.append(np.array([ 2.888,   9.882,  -3.558]))
    atom_positions.append(np.array([ 1.982,   3.902,  -0.137]))
    atom_positions.append(np.array([ 1.041,   5.396,  -2.426]))
    atom_positions.append(np.array([ 0.939,   6.927,  -0.453]))
    atom_positions.append(np.array([ 2.655,   6.613,  -0.194]))
    atom_positions.append(np.array([ 0.211,   8.190,  -2.414]))
    atom_positions.append(np.array([ 4.305,   7.633,  -1.482]))
    atom_positions.append(np.array([ 0.791,  10.037,  -3.932]))
    atom_positions.append(np.array([ 4.911,   9.494,  -2.995]))
    atom_positions.append(np.array([ 3.146,  10.692,  -4.222]))
    
    atom_masses.append(14.0067)
    atom_masses.append(12.0107)
    atom_masses.append(12.0107)
    atom_masses.append(15.999)
    atom_masses.append(12.0107)
    atom_masses.append(12.0107)
    atom_masses.append(12.0107)
    atom_masses.append(12.0107)
    atom_masses.append(12.0107)
    atom_masses.append(12.0107)
    atom_masses.append(12.0107)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    
    cg = np.array([2.208,   9.750 , -3.523])
    
    return atom_positions, atom_masses, cg

def getArginine():
    atom_positions = []
    atom_masses = []
    
    atom_positions.append(np.array([ 3.267,   5.008,  -3.467]))
    atom_positions.append(np.array([ 4.521,   4.737,  -4.082]))
    atom_positions.append(np.array([ 5.356,   6.007,  -4.101]))
    atom_positions.append(np.array([ 5.161,   6.882,  -4.949]))
    atom_positions.append(np.array([ 4.274,   4.276,  -5.485]))
    atom_positions.append(np.array([ 5.522,   3.996,  -6.279]))
    atom_positions.append(np.array([ 5.166,   3.637,  -7.693]))
    atom_positions.append(np.array([ 6.346,   3.577,  -8.553]))
    atom_positions.append(np.array([ 6.310,   3.381,  -9.872]))
    atom_positions.append(np.array([ 5.148,   3.176, -10.488]))
    atom_positions.append(np.array([ 7.437,   3.377, -10.573]))
    atom_positions.append(np.array([ 2.512,   5.223,  -4.059]))
    atom_positions.append(np.array([ 5.030,   3.963,  -3.533]))
    atom_positions.append(np.array([ 3.674,   3.381,  -5.452]))
    atom_positions.append(np.array([ 3.719,   5.056,  -5.982]))
    atom_positions.append(np.array([ 6.145,   4.878,  -6.282]))
    atom_positions.append(np.array([ 6.055,   3.174,  -5.826]))
    atom_positions.append(np.array([ 4.674,   2.676,  -7.682]))
    atom_positions.append(np.array([ 4.484,   4.387,  -8.063]))
    atom_positions.append(np.array([ 7.217,   3.707,  -8.114]))
    atom_positions.append(np.array([ 4.289,   3.166,  -9.972]))
    atom_positions.append(np.array([ 5.121,   3.029, -11.481]))
    atom_positions.append(np.array([ 8.323,   3.519, -10.120]))
    atom_positions.append(np.array([ 7.421,   3.229, -11.565]))
    
    atom_masses.append(14.0067)
    atom_masses.append(12.0107)
    atom_masses.append(12.0107)
    atom_masses.append(15.999)
    atom_masses.append(12.0107)
    atom_masses.append(12.0107)
    atom_masses.append(12.0107)
    atom_masses.append(14.0067)
    atom_masses.append(12.0107)
    atom_masses.append(14.0067)
    atom_masses.append(14.0067)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    
    cg = np.array([6.324 ,  3.374  ,-9.904])
    
    return atom_positions, atom_masses, cg

def getSerine():
    atom_positions = []
    atom_masses = []

    atom_positions.append(np.array([ 6.260,   6.109,  -3.153]))
    atom_positions.append(np.array([ 7.118,   7.269,  -3.033]))
    atom_positions.append(np.array([ 8.096,   7.339,  -4.210]))
    atom_positions.append(np.array([ 8.825,   6.379,  -4.485]))
    atom_positions.append(np.array([ 7.882,   7.212,  -1.707])) 
    atom_positions.append(np.array([ 8.686,   8.363,  -1.522]))
    atom_positions.append(np.array([ 6.336,   5.386,  -2.495])) 
    atom_positions.append(np.array([ 6.493,   8.147,  -3.039]))
    atom_positions.append(np.array([ 7.176,   7.149,  -0.892]))
    atom_positions.append(np.array([ 8.517,   6.338,  -1.697]))
    atom_positions.append(np.array([ 9.325,   8.189,  -0.822]))
    
    atom_masses.append(14.0067)
    atom_masses.append(12.0107)
    atom_masses.append(12.0107)
    atom_masses.append(15.999)
    atom_masses.append(12.0107)
    atom_masses.append(15.999)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    
    cg = np.array([8.375  , 7.881 , -1.574])
    
    return atom_positions, atom_masses, cg

def getPronine():
    atom_positions = []
    atom_masses = []
    
    atom_positions.append(np.array([ 8.111,   8.483,  -4.926]))
    atom_positions.append(np.array([ 8.981,   8.697,  -6.095]))
    atom_positions.append(np.array([ 10.446,   8.898,  -5.728]))
    atom_positions.append(np.array([11.117,   9.758,  -6.278])) 
    atom_positions.append(np.array([ 8.442,   9.981,  -6.706]))
    atom_positions.append(np.array([ 7.795,  10.707,  -5.581]))
    atom_positions.append(np.array([ 7.257,   9.658,  -4.657])) 
    atom_positions.append(np.array([ 8.895,   7.894,  -6.811]))
    atom_positions.append(np.array([ 9.280,  10.542,  -7.094])) 
    atom_positions.append(np.array([ 7.746,   9.750,  -7.496]))
    atom_positions.append(np.array([ 8.526,  11.315,  -5.068]))
    atom_positions.append(np.array([ 6.992,  11.324,  -5.955]))
    atom_positions.append(np.array([ 7.351,   9.978,  -3.630]))
    atom_positions.append(np.array([ 6.227,   9.446,  -4.898]))
    
    atom_masses.append(14.0067)
    atom_masses.append(12.0107)
    atom_masses.append(12.0107)
    atom_masses.append(15.999)
    atom_masses.append(12.0107)
    atom_masses.append(12.0107)
    atom_masses.append(12.0107)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    
    cg = np.array([12.560,   6.087,  -6.103])
    
    return atom_positions, atom_masses, cg

def getCysine():
    atom_positions = []
    atom_masses = []
    
    atom_positions.append(np.array([ 10.938,   8.098,  -4.836]))
    atom_positions.append(np.array([ 12.311,   8.186,  -4.414]))
    atom_positions.append(np.array([ 12.771,   6.831,  -3.892])) 
    atom_positions.append(np.array([ 12.560,   6.513,  -2.724]))
    atom_positions.append(np.array([12.441 ,  9.242 , -3.326]))
    atom_positions.append(np.array([ 14.134,   9.491,  -2.711])) 
    atom_positions.append(np.array([ 10.356,   7.407,  -4.445])) 
    atom_positions.append(np.array([ 12.922,   8.481,  -5.264]))
    atom_positions.append(np.array([  12.091,  10.184,  -3.725]))
    atom_positions.append(np.array([11.818,   8.960,  -2.490]))
    
    atom_masses.append(14.0067)
    atom_masses.append(12.0107)
    atom_masses.append(12.0107)
    atom_masses.append(15.999)
    atom_masses.append(12.0107)
    atom_masses.append(32.065)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    atom_masses.append(1.00784)
    
    cg = np.array([13.672 ,  9.423 , -2.879])
    
    return atom_positions, atom_masses, cg
      
if __name__ == "__main__":
    # Define the cg pos, atom positions and masses
    atom_positions, masses, cg = getPronine()
    
    global min_distance
    min_distance = 100.
    
    # Do the backtracking
    for l in [1,2,3,4,5,6,7]:
        print('Trying l = ', l)
        
        positions = list(map(lambda i: i, range(l)))  
        success, pos = backtrack(copy.copy(positions), 0, atom_positions, masses, cg)
        
        if success:
            print('Success', pos)
            break
    
    print('Minimal distance', min_distance, exact_pos, exact_mid) 