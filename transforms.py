import numpy as np
from astropy.coordinates import CylindricalRepresentation, CylindricalDifferential
import gala.dynamics as gd
import astropy.units as u

def viewing_vector_from_angles(az_deg, el_deg):
    '''
    Converts azimuth and elevation angles (in degrees) that define a viewing direction into a 3-D unit direction vector.

        Parameters
        ----------
        az_deg: float
            Azimuthal angle (measured from x-axis in xy-plane)
        el_deg: float
            Elevation/altitude angle (angle above xy-plane)

        Returns
        -------
        viewing_vector: numpy.ndarray
            3-D unit vector pointing in the direction defined by the input azimuth and elevation angles
        
        Notes
        -----
            Requires numpy
            - Angles are interpreted in a right-handed coordinate system, consistent with spherical coordinates.
    '''

    az = np.radians(az_deg)
    el = np.radians(el_deg)
    
    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)

    viewing_vector = np.array([x, y, z])
    
    return viewing_vector

def norm_vector(vector):

    '''
    Normalizes a vector

        Parameters
        ----------
        vector: numpy.ndarray
            1-D numpy array representing a vector to be normalized

        Returns
        -------
        normed_vector: np.ndarray
            1-D numpy array representing the normalized vector

        Notes
        -----
            Requires numpy
    '''

    magnitude = np.linalg.norm(vector)

    if magnitude==0:
        return np.zeros_like(vector)
    else:
        normed_vector = vector/magnitude
    
    return normed_vector

def create_cylindrical_initial_conditions(r0, phi0, z0, v_r0, v_phi_tan0, v_z0):

    angular_velocity = v_phi_tan0 / r0 * u.rad # → rad/s
    pos = CylindricalRepresentation(rho=r0, phi=phi0, z=z0)
    vel = CylindricalDifferential(d_rho=v_r0, d_phi=angular_velocity.to(u.rad/u.s), d_z=v_z0)
    initial_conditions_cyl = gd.PhaseSpacePosition(pos.with_differentials(vel))

    return initial_conditions_cyl
