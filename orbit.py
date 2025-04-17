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
        time_labels = []
        for row in ax:
            for a in row:
                a.text(0.05, 0.95, self.t[time_index], transform=a.transAxes, fontsize=12,
                              verticalalignment='top', horizontalalignment='left', 
                              bbox=dict(facecolor='white', alpha=0.5))
                a.plot(0, 0, color="black", marker="x")
                a.plot(data_combos[current_data][0][0], data_combos[current_data][1][0], 'x', color='red')
                a.plot(data_combos[current_data][0][:time_index], data_combos[current_data][1][:time_index], alpha=0.75)
                a.plot(data_combos[current_data][0][time_index], data_combos[current_data][1][time_index], '*', markersize=10, color='orange')
                a.set_aspect('equal')
                a.set_xlabel(labels[current_data][0])
                a.set_ylabel(labels[current_data][1])

                current_data += 1

        if with_tails:
            
            component_indices = {'$x$ (kpc)': 0, '$y$ (kpc)': 1, '$z$ (kpc)': 2}

            for i, a in enumerate(ax[0]):
                x_index = component_indices[labels[i][0]]
                y_index = component_indices[labels[i][1]]
                print(f"{x_index}, {y_index}")
                tail_x_points = data_combos[i][0][time_index-20:time_index:-1]
                # fix this based on what the chatGPT says
                print(f"x_points: {tail_x_points}")
                # tail_y_points = pu.vector_to_tail_line(self.tail_angle_unit_vectors[time_index,[x_index, y_index]], [data_combos[i][0][time_index], data_combos[i][1][time_index]], tail_x_points)


                # solution: normalize 2-D projected vector
                galaxy_x = data_combos[i][0][time_index].value
                galaxy_y = data_combos[i][1][time_index].value
                tail_length = 10000
                tail_distance = 10000
                steps = np.linspace(0, tail_distance, tail_length)  # negative to go backward
                tail_x_points = galaxy_x + steps * self.tail_angle_unit_vectors[time_index, x_index]
                tail_y_points = galaxy_y + steps * self.tail_angle_unit_vectors[time_index, y_index]

                tail_point_selection = np.abs(tail_y_points - data_combos[i][1][time_index].value) < 50
                tail_x_points_selected = tail_x_points[tail_point_selection]
                tail_y_points_selected = tail_y_points[tail_point_selection]
                
                a.plot(tail_x_points_selected, tail_y_points_selected, '-', color='red')

        return fig
        
    
    def calc_2d_tail_angles_and_orbit(self, view_dir):
    
        '''view_dir is the direction the observer is looking towards!!'''

        # ensure the view_dir vector is normalized
        view_dir_normed = view_dir/np.linalg.norm(view_dir)

        # define another vector in the basis (not parallel to view_dir)
        if np.allclose(view_dir_normed, [0, 0, 1]):
            up = np.array([0, 1, 0])
        else:
            up = np.array([0, 0, 1])

        # define a rightwards basis vector perpendicular to the view direction and up vector!
        right = np.cross(up, view_dir_normed)
        right /= np.linalg.norm(right)

        # redefine the up vector so it is orthogonal to both
        up = np.cross(view_dir, right)

        # transform cartesian basis vectors into viewer's basis
        rot_matrix = np.vstack([right, up, view_dir_normed]).T
        rotated_tail_angles = [rot_matrix @ tail_angle_vector for tail_angle_vector in self.tail_angle_unit_vectors]
        rotated_tail_angles = np.array(rotated_tail_angles)

        # take only the x and y (up and right) components to get the projection in 2D
        tail_angles_2d_vectors = rotated_tail_angles[:, :2]
        
        return tail_angles_2d_vectors
