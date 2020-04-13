import numpy as np
from pmd_beamphysics import ParticleGroup



def postprocess_screen(screen, **params):
    
    if ('cylindrical_copies' in params and params['cylindrical_copies']>0):
        screen = add_cylindrical_copies(screen, params['cylindrical_copies'])
    return screen



def add_cylindrical_copies(screen, n_copies):
    species = screen.species
    npart = len(screen.x)
    
    x = np.matlib.repmat(screen.x, 1, n_copies).reshape(npart*n_copies)
    y = np.matlib.repmat(screen.y, 1, n_copies).reshape(npart*n_copies)
    z = np.matlib.repmat(screen.z, 1, n_copies).reshape(npart*n_copies)
    px = np.matlib.repmat(screen.px, 1, n_copies).reshape(npart*n_copies)
    py = np.matlib.repmat(screen.py, 1, n_copies).reshape(npart*n_copies)
    pz = np.matlib.repmat(screen.pz, 1, n_copies).reshape(npart*n_copies)
    t = np.matlib.repmat(screen.t, 1, n_copies).reshape(npart*n_copies)
    status = np.matlib.repmat(screen.status, 1, n_copies).reshape(npart*n_copies)
    weight = np.matlib.repmat(screen.weight, 1, n_copies).reshape(npart*n_copies)
    
    theta = np.linspace(0,2*np.pi,npart+1)
    theta = theta[:-1]
    theta = np.matlib.repmat(theta, n_copies, 1).T.reshape(npart*n_copies)
    
    costh = np.cos(theta);
    sinth = np.sin(theta);
    

    px_new = px*costh - py*sinth
    py_new = px*sinth + py*costh
    px = px_new
    py = py_new

    x_new = x*costh - y*sinth
    y_new = x*sinth + y*costh
    x = x_new
    y = y_new
    
    weight = weight/n_copies
    
    data = dict(
        species=species,
        x=x,
        y=y,
        z=z,
        px=px,
        py=py,
        pz=pz,
        t=t,
        status=status,
        weight=weight
    )
    
    return ParticleGroup(data=data)