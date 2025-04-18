import numpy as np
import matplotlib.pyplot as plt

import plotting_utils as pu
from angle_transforms import *

class Projected_Orbit:

    def __init__(self, original_orbit: Orbit, view_dir: np.ndarray):

        from orbit import Orbit

        '''view_dir is the direction the observer is looking towards!!'''
        
        self.original_orbit = original_orbit
        self.tail_angle_3d_unit_vectors = self.original_orbit.tail_angle_unit_vectors

        # ensure the view_dir vector is normalized
        self.view_dir_normed = norm_vector(view_dir)

        # define another vector in the basis (not parallel to view_dir)
        if np.allclose(self.view_dir_normed, [0, 0, 1]):
            up = np.array([0, 1, 0])
        else:
            up = np.array([0, 0, 1])

        # define a rightwards basis vector perpendicular to the view direction and up vector!
        right = np.cross(up, self.view_dir_normed)
        right /= np.linalg.norm(right)

        # redefine the up vector so it is orthogonal to both
        up = np.cross(self.view_dir_normed, right)

        # transform cartesian tail angle vectors into viewer's basis
        self.rot_matrix = np.vstack([right, up, self.view_dir_normed]).T
        rotated_tail_angles = self.tail_angle_3d_unit_vectors.dot(self.rot_matrix)
        # rotated_tail_angles = np.array(rotated_tail_angles)

        # rotate the orbit using the same matrix
        orbit_points = np.column_stack([self.original_orbit.x, self.original_orbit.y, self.original_orbit.z])
        self.orbit_projected = orbit_points.dot(self.rot_matrix)
        self.orbit_projected = self.orbit_projected[:, :2]
        self.projected_x = self.orbit_projected[:, 0]
        self.projected_y = self.orbit_projected[:, 1]

        # take only the x and y (up and right) components to get the projection in 2D
        self.tail_angles_2d_vectors = rotated_tail_angles[:, :2]
        self.tail_angles_2d_vectors = norm_vector(self.tail_angles_2d_vectors) # normalizing calculated vectors
        self.tail_angles_2d = np.arctan2(self.tail_angles_2d_vectors[:,1], self.tail_angles_2d_vectors[:,0])

        self.t = self.original_orbit.t

    
    def plot_projected_orbit(self, time_index):
        
        fig, ax = plt.subplots(1, 1)

        y_lims = pu.find_axes_limits(self.projected_y.value, 0.1)
        x_lims = pu.find_axes_limits(self.projected_x.value, 0.1)
        max_y_lim = np.max(np.abs(y_lims))
        max_x_lim = np.max(np.abs(x_lims))
        lims = (-np.max([max_x_lim, max_y_lim]), np.max([max_x_lim, max_y_lim]))

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')

        pu.plot_orbit(ax, self.projected_x, self.projected_y, self.t, time_index=time_index)

        tail_angle_vector = self.tail_angles_2d_vectors[time_index, :]
        tail_x, tail_y = pu.calc_tail_line(ax, self.projected_x, self.projected_y, tail_angle_vector, time_index)
        ax.plot(tail_x, tail_y, '-', color='red', zorder=3)

        ax.set_xlabel('$x$ (kpc)')
        ax.set_ylabel('$y$ (kpc)')
        ax.set_title ("Orbit from viewer's perspective")

        return fig

