from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.units import unit, PARTICLEGROUP_UNITS
from pmd_beamphysics.statistics import norm_emit_calc
import numpy as np
import copy

class ParticleGroupExtension(ParticleGroup):
    n_slices = 50
    slice_key = 't'
    
    def __init__(self, input_particle_group=None, data=None):
        if (input_particle_group):
            data={}
            for key in input_particle_group._settable_keys:
                data[key] = copy.copy(input_particle_group[key])  # is deepcopy needed?
        
        super().__init__(data=data)
        
        if ('sqrt_norm_emit_4d' not in PARTICLEGROUP_UNITS.keys()):
            PARTICLEGROUP_UNITS['sqrt_norm_emit_4d'] = unit('m')

        if ('slice_emit_x' not in PARTICLEGROUP_UNITS.keys()):
            PARTICLEGROUP_UNITS['slice_emit_x'] = unit('m')

        if ('slice_emit_y' not in PARTICLEGROUP_UNITS.keys()):
            PARTICLEGROUP_UNITS['slice_emit_y'] = unit('m')

        if ('slice_emit_4d' not in PARTICLEGROUP_UNITS.keys()):
            PARTICLEGROUP_UNITS['slice_emit_4d'] = unit('m')

    @property
    def sqrt_norm_emit_4d(self):
        return np.sqrt(norm_emit_calc(self, planes=['x', 'y']))

    @property
    def slice_emit_x(self):
        (p_list, _, _) = divide_particles(self, nbins = self.n_slices, key=self.slice_key)
        return slice_emit(p_list, 'norm_emit_x')

    @property
    def slice_emit_y(self):
        (p_list, _, _) = divide_particles(self, nbins = self.n_slices, key=self.slice_key)
        return slice_emit(p_list, 'norm_emit_y')

    @property
    def slice_emit_4d(self):
        (p_list, _, _) = divide_particles(self, nbins = self.n_slices, key=self.slice_key)
        return slice_emit(p_list, 'sqrt_norm_emit_4d')


#-----------------------------------------
# helper functions for ParticleGroupExtension class

def slice_emit(p_list, key):
    min_particles = 5
    weights = np.array([0.0 for p in p_list])
    emit = np.array([0.0 for p in p_list])
    for p_i, p in enumerate(p_list):
        if (p.n_particle >= min_particles):
            emit[p_i] = p[key]
            weights[p_i] = p['charge']
    weights = weights/np.sum(weights)
    avg_emit = np.sum(emit*weights)
    
    return avg_emit


def divide_particles(particle_group, nbins = 100, key='t'):
    """
    Splits a particle group into even slices of 'key'. Returns a list of particle groups. 
    """
    x = getattr(particle_group, key) 
    
    if (key == 'r'):
        x = x*x
        xmin = 0  # force r=0 as min, could use min(x) here, optionally
        xmax = max(x)
        dx = (xmax-xmin)/(nbins-1)
        edges = np.linspace(xmin, xmax + 0.01*dx, nbins+1) # extends slightly further than max(r2)
        dx = edges[1]-edges[0]
    else:
        dx = (max(x)-min(x))/(nbins-1)
        edges = np.linspace(min(x) - 0.01*dx, max(x) + 0.01*dx, nbins+1) # extends slightly further than range(r2)
        dx = edges[1]-edges[0]
    
    which_bins = np.digitize(x, edges)-1
    
    if (key == 'r'):
        x = np.sqrt(x)
        edges = np.sqrt(edges)
            
    # Split particles
    plist = []
    for bin_i in range(nbins):
        chunk = which_bins==bin_i
        # Prepare data
        data = {}
        #keys = ['x', 'px', 'y', 'py', 'z', 'pz', 't', 'status', 'weight'] 
        for k in particle_group._settable_array_keys:
            data[k] = getattr(particle_group, k)[chunk]
        # These should be scalars
        data['species'] = particle_group.species
        
        # New object
        p = ParticleGroupExtension(data=data)
        plist.append(p)
    
    # normalization for sums of particle properties, = 1 / histogram bin width
    if (key == 'r'):
        density_norm = 1.0/(np.pi*(edges[1]**2 - edges[0]**2))
    else:
        density_norm = 1.0/(edges[1] - edges[0])
    
    return plist, edges, density_norm