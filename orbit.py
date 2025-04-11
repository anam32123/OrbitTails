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

        return
    
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
    
    # def calc_2d_tail_angles_and_orbit:

    # def plot_2d_orbit_projection:
        