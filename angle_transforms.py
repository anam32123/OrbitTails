import numpy as np

def viewing_vector_from_angles(az_deg, el_deg):
    az = np.radians(az_deg)
    el = np.radians(el_deg)
    
    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)
    
    return np.array([x, y, z])

def norm_vector(vector):

    magnitude = np.linalg.norm(vector)

    if magnitude==0:
        return 0
    else:
        normed_vector = vector/magnitude
    
    return normed_vector
