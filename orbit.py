import numpy as np
from gala import potential as gp
from gala import dynamics as gd
from gala import integrate as gi
from gala.units import galactic

from astropy import units as u

from matplotlib import pyplot as plt

import matplotlib.animation as animation
import matplotlib as mpl

import plotting_utils as pu
# from Projected_Orbit import Projected_Orbit
from angle_transforms import *

mpl.rcParams['animation.embed_limit'] = 50

class Orbit:
    def __init__(self, Gala_Potential, initial_conditions, dt=2., n_steps=2000):
        
        # print(f"Simulating an orbit in the host potential with intial conditions position = {initial_conditions.xyz}, velocity = {initial_conditions.v_xyz}")

        self.orbit = gp.Hamiltonian(Gala_Potential).integrate_orbit(initial_conditions, dt=dt, n_steps=n_steps)
        self.initial_conditions = initial_conditions

        self.x, self.y, self.z = self.orbit.xyz
        self.vx, self.vy, self.vz = self.orbit.v_xyz.to(u.km/u.s)
        self.t = self.orbit.t

        return # don't need this
    
    def plot_orbit(self, *args, **kwargs):
        
        self.orbit.plot(*args, **kwargs)

        return
    
    def calc_3d_tail_angles(self):
    
        vx, vy, vz = self.orbit.v_xyz.to(u.km/u.s)
        
        vx = np.array(vx)
        vy = np.array(vy)
        vz = np.array(vz)
        
        velocity_vectors = np.vstack((vx, vy, vz))
        
        velocity_vectors = velocity_vectors.T
        self.velocity_magnitudes = np.apply_along_axis(np.linalg.norm, axis=1, arr=velocity_vectors)
        velocity_magnitudes = self.velocity_magnitudes[:, np.newaxis]
        
        self.velocity_unit_vectors = velocity_vectors/velocity_magnitudes

        self.tail_angle_unit_vectors = -self.velocity_unit_vectors
        
        return
    
    def Animate_Orbit(self):

        fig, ax = plt.subplots(2, 3, figsize=(30, 20))

        orbit_lines = []
        galaxy_points = []
        data_combos = [[self.x, self.y], [self.x, self.z], [self.y, self.z], [self.vx, self.vy], [self.vx, self.vz], [self.vy, self.vz]]
    
        position_xlim, position_ylim = pu.find_limits_multiple_axes(data_combos[0:3], 0.05)
        velocity_xlim, velocity_ylim = pu.find_limits_multiple_axes(data_combos[3:], 0.05)

        labels = [['$x$ (kpc)', '$y$ (kpc)'], ['$x$ (kpc)', '$z$ (kpc)'], ['$y$ (kpc)', '$z$ (kpc)'], ['$v_x$ (km/s)', '$v_y$ (km/s)'], ['$v_x$ (km/s)', '$v_z$ (km/s)'], ['$v_y$ (km/s)', '$v_z$ (km/s)']]

        for i in range(3):
            ax[0][i].set_xlim(position_xlim)
            ax[0][i].set_ylim(position_ylim)
            ax[1][i].set_xlim(velocity_xlim)
            ax[1][i].set_ylim(velocity_ylim)

        current_data = 0
        time_labels = []
        for row in ax:
            for a in row:
                orbit_line, = a.plot([], [], lw=2)
                galaxy_point, = a.plot([], [], '*')
                time_text = a.text(0.05, 0.95, '', transform=a.transAxes, fontsize=12,
                              verticalalignment='top', horizontalalignment='left', 
                              bbox=dict(facecolor='white', alpha=0.5))
                orbit_lines.append(orbit_line)
                galaxy_points.append(galaxy_point)
                time_labels.append(time_text)
                a.plot(0, 0, color="black", marker="x")
                a.scatter(data_combos[current_data][0][0], data_combos[current_data][1][0], color='orange', marker='*', s=5, alpha=0.75)
                a.set_aspect('equal')
                a.set_xlabel(labels[current_data][0])
                a.set_ylabel(labels[current_data][1])

                current_data += 1

        def init():

            for orbit_line, galaxy_point, time_text in zip(orbit_lines, galaxy_points, time_labels):
                orbit_line.set_data([], [])
                galaxy_point.set_data([], [])
                time_text.set_text('')
            
            return(orbit_lines + galaxy_points)

        def update(frame):

            for i, (orbit_line, galaxy_point, time_text) in enumerate(zip(orbit_lines, galaxy_points, time_labels)):
                    orbit_line.set_data(data_combos[i][0][:frame].value, data_combos[i][1][:frame].value)
                    galaxy_point.set_data([data_combos[i][0][frame].value], [data_combos[i][1][frame].value])
                    time_text.set_text(f"Time = {self.t[frame]}")
            return (orbit_lines + galaxy_points)

        self.anim = animation.FuncAnimation(fig, update, init_func=init, frames=len(self.x), interval=10)
        # plt.show()

        return
    
    def plot_orbits_not_animated(self, time_index, with_tails=False):
        
        '''With_tails requires that you have already calculated 3-D tail angles'''

        fig, ax = plt.subplots(2, 3, figsize=(30, 20))

        orbit_lines = []
        galaxy_points = []
        data_combos = [[self.x, self.y], [self.x, self.z], [self.y, self.z], [self.vx, self.vy], [self.vx, self.vz], [self.vy, self.vz]]

        position_xlim, position_ylim = pu.find_limits_multiple_axes(data_combos[0:3], 0.05)
        velocity_xlim, velocity_ylim = pu.find_limits_multiple_axes(data_combos[3:], 0.05)
        
        labels = [['$x$ (kpc)', '$y$ (kpc)'], ['$x$ (kpc)', '$z$ (kpc)'], ['$y$ (kpc)', '$z$ (kpc)'], ['$v_x$ (km/s)', '$v_y$ (km/s)'], ['$v_x$ (km/s)', '$v_z$ (km/s)'], ['$v_y$ (km/s)', '$v_z$ (km/s)']]

        for i in range(3):
            ax[0][i].set_xlim(position_xlim)
            ax[0][i].set_ylim(position_ylim)
            ax[1][i].set_xlim(velocity_xlim)
            ax[1][i].set_ylim(velocity_ylim)

        current_data = 0
        for row in ax:
            for a in row:
                # a.text(0.05, 0.95, self.t[time_index], transform=a.transAxes, fontsize=12,
                #               verticalalignment='top', horizontalalignment='left', 
                #               bbox=dict(facecolor='white', alpha=0.5))
                # a.plot(0, 0, color="black", marker="x", zorder=1)
                # a.plot(data_combos[current_data][0][0], data_combos[current_data][1][0], 'x', color='red', zorder=1)
                # a.plot(data_combos[current_data][0][:time_index], data_combos[current_data][1][:time_index], alpha=0.75, zorder=2)
                # a.plot(data_combos[current_data][0][time_index], data_combos[current_data][1][time_index], '*', markersize=10, color='orange', zorder=4)
                # a.plot(data_combos[current_data][0], data_combos[current_data][1], '-', color='darkgray', linewidth=0.5, zorder=1, alpha=0.75)
                # a.set_aspect('equal')

                pu.plot_orbit(a, data_combos[current_data][0], data_combos[current_data][1], time=self.t, time_index=time_index)
                a.set_xlabel(labels[current_data][0])
                a.set_ylabel(labels[current_data][1])

                current_data += 1


        if with_tails:
            
            component_indices = {'$x$ (kpc)': 0, '$y$ (kpc)': 1, '$z$ (kpc)': 2}

            for i, a in enumerate(ax[0]):

                # indices for indexing into array of 3-D tail vectors
                x_index = component_indices[labels[i][0]]
                y_index = component_indices[labels[i][1]]

                # points for plotting the tail
                # tail_x_points = data_combos[i][0][time_index-20:time_index:-1]
                # tail_y_points = pu.vector_to_tail_line(self.tail_angle_unit_vectors[time_index,[x_index, y_index]], [data_combos[i][0][time_index], data_combos[i][1][time_index]], tail_x_points)

                # solution: normalize 2-D projected vector
                # galaxy_x = data_combos[i][0][time_index].value
                # galaxy_y = data_combos[i][1][time_index].value
                # tail_length = 100
                # tail_distance = 100
                # steps = np.linspace(0, tail_distance, tail_length)  # negative to go backward

                # normalized_2d_tail_angle_unit_vector = norm_vector(self.tail_angle_unit_vectors[time_index, [x_index,y_index]])

                # tail_x_points = galaxy_x + steps * normalized_2d_tail_angle_unit_vector[0]
                # tail_y_points = galaxy_y + steps * normalized_2d_tail_angle_unit_vector[1]

                # tail_point_selection = np.abs(tail_y_points - data_combos[i][1][time_index].value) < 50
                # tail_x_points_selected = tail_x_points[tail_point_selection]
                # tail_y_points_selected = tail_y_points[tail_point_selection]

                tail_vector = self.tail_angle_unit_vectors[time_index, [x_index, y_index]]

                tail_x_points_selected, tail_y_points_selected = pu.calc_tail_line(a, data_combos[i][0], data_combos[i][1], tail_vector=tail_vector, time_index=time_index)
                
                a.plot(tail_x_points_selected, tail_y_points_selected, '-', color='red', zorder=3)


        return fig
    
    def project_orbit_tail_angles(self, view_dir):

        return Projected_Orbit(self, view_dir)
    
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

