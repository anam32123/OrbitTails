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
from transforms import *

mpl.rcParams['animation.embed_limit'] = 50

class Orbit:
    """
    A class containing information about a galaxy's orbit in a cluster potential and key methods for integrating the orbit, calculating tail angle parameters, and plotting.

        Attributes
        ----------
            orbit: gala.dynamics.Orbit
                An Orbit object from the gala.dynamics library, intialized from initial conditions and the host potential
            initial_conditions: gala.dynamics.PhaseSpacePosition
                A PhaseSpacePosition object containing the galaxy's initial position and velocity in phase space.
            x: astropy.units.Quantity
                A 1D array containing positional information about the galaxy in the x-direction, for all integrated times.
            y: astropy.units.Quantity
                A 1D array containing positional information about the galaxy in the y-direction, for all integrated times.
            z: astropy.units.Quantity
                A 1D array containing positional information about the galaxy in the z-direction, for all integrated times.
            vx: astropy.units.Quantity
                A 1D array of the galaxy's velocity x-component for all times in the integration
            vy: astropy.units.Quantity
                A 1D array of the galaxy's velocity y-component for all times in the integration
            vz: astropy.units.Quantity
                A 1D array of the galaxy's velocity z-component for all times in the integration
            t: astropy.units.Quantity
                A 1D array of the time after the beginning of integration, for each timestep in the integration
            velocity_magnitudes: numpy.ndarray
                A 1D array of the magnitudes of velocity vectors over the course of the orbit.
            tail_angle_unit_vectors: numpy.ndarray
                A 2D NumPy array where each row is a 3D unit vector representing the direction of the galaxy tail at each timestep.
            tail_3d_radial_angle_rad: numpy.ndarray
                An array of 3D tail angles in radians for the galaxy at each point in its orbit, measured as the angle between the
                galaxy tail and the radial direction to cluster center.
            tail_3d_radial_angle_deg: numpy.ndarray
                An array of 3D tail angles in radians for the galaxy at each point in its orbit, measured as the angle between the
                galaxy tail and the radial direction to cluster center.
    """

    def __init__(self, Gala_Potential, initial_conditions, dt=2., n_steps=2000):

        """
        Initializes an Orbit using the gala package's built-in orbit integration function, from initial conditions and a pre-specified host potential.

            Parameters
            ----------
            Gala_Potential: gala.potential.PotentialBase
                A gala potential object (e.g., HernquistPotential, NFWPotential) representing the host cluster potential.
            initial_conditions: gala.dynamics.PhaseSpacePosition
                A PhaseSpacePosition object containing the galaxy's initial position and velocity in phase space.
            dt: float, optional
                Time delta for orbit integration, in the time units of the host potential (default for galactic units is Myr). Default: 2
            n_steps: int, optional
                Number of timesteps of length dt for orbit integration. Default: 2000
        """
        
        # print(f"Simulating an orbit in the host potential with intial conditions position = {initial_conditions.xyz}, velocity = {initial_conditions.v_xyz}")

        self.orbit = gp.Hamiltonian(Gala_Potential).integrate_orbit(initial_conditions, dt=dt, n_steps=n_steps)
        self.initial_conditions = initial_conditions

        self.x, self.y, self.z = self.orbit.xyz
        self.vx, self.vy, self.vz = self.orbit.v_xyz.to(u.km/u.s)
        self.t = self.orbit.t
    
    def plot_orbit(self, *args, **kwargs):

        """
        A wrapper for the `gala.dynamics.Orbit.plot` function, which plots simulated orbits in a variety of potential coordinate systems.

            Parameters
            ----------
            *args: list, optional
                Positional arguments forwarded to `gala.dynamics.Orbit.plot`.
            **kwargs: dict, optional
                Keyword arguments forwarded to gala.dynamics.Orbit.plot

            Returns
            -------
            None
        """
        
        self.orbit.plot(*args, **kwargs)
    
    def calc_3d_tail_angles(self):

        """
        Compute 3D velocity unit vectors, tail direction unit vectors, and tail angles for each timestep.

        This method calculates unit vectors from the galaxy's velocity at each timestep,
        stores the magnitudes of those velocity vectors, and defines the tail direction
        as the opposite of the velocity direction (i.e., trailing behind the motion). Tail angles are calculated
        as the 3D angle between the tail direction vector and the radial direction towards cluster center.

            Sets
            ----
            self.velocity_magnitudes : numpy.ndarray
                A 1D array containing the magnitude of the galaxy's velocity at each timestep.
            self.velocity_unit_vectors : numpy.ndarray
                A 2D array of shape (N, 3), where each row is the unit velocity vector at a given timestep.
            self.tail_angle_unit_vectors : numpy.ndarray
                A 2D array of shape (N, 3), where each row is a unit vector pointing in the tail direction (opposite velocity).
            
            
            Returns
            -------
            None
        """

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

        # forming the angle between the tail and the radial direction
        # 1) Build the radial unit‐vector (galaxy → cluster)
        #    If your galaxy is at (x,y,z) and the cluster is at the origin:
        pos = np.vstack([self.x.value, self.y.value, self.z.value]).T    # shape (N,3)
        radial_vec = -pos                             # points towards the cluster
        # normalize radial vector
        radii = np.linalg.norm(radial_vec, axis=1, keepdims=True)   # shape (N,1)
        # avoid division by zero
        radii[radii == 0] = 1.0                                      
        radial_unit_vec = radial_vec / radii     # shape (N,3)

        # 2) Get your tail‐direction unit‐vectors in 3D
        # tail_unit_vec = self.original_orbit.tail_angle_3d_unit_vectors   # shape (N,3)

        # 3) Dot‐product gives cos(θ)
        cos_theta = np.einsum('ij,ij->i', self.tail_angle_unit_vectors, radial_unit_vec) # performs row-wise dot product on the data
        cos_theta = np.clip(cos_theta, -1.0, 1.0)                  # guard numerical overshoot

        # 4) Angle in radians (0 => perfectly toward; π => perfectly away)
        theta_rad = np.arccos(cos_theta)

        # 5) (Optional) in degrees
        theta_deg = np.degrees(theta_rad)

        # store on your object:
        self.tail_3d_radial_angle_rad = theta_rad
        self.tail_3d_radial_angle_deg = theta_deg

        
        return
    
    def Animate_Orbit(self):

        """
        Generates animations of galaxy orbits over time that include tail angle and position

        THIS NEEDS TO BE FIXED
        """

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
        
        '''
        Plots galaxy orbits and tail angles for all cartesian position and velocity component combinations.

        The user can choose a specific time index of the orbit to view the tail direction and galaxy position/velocity for that
        time, in addition to the past and future orbit.

            Parameters
            ----------
            time_index: int
                Index for the time axis/array, indicating the time at which we want to see a snapshot of the orbit
            with_tails: boolean, optional
                Flag indicating whether the user wants the orbit plotted with tail directions. If True, plots the tail direction 
                for the time specified by time_index together with the galaxy and orbit. Setting with_tails=True requires that 
                calc_3d_tail_angles has already been run for the given orbit. Default: False
            
            Returns
            -------
            fig: matplotlib.pyplot.Figure
                Figure object containing the galaxy orbit plot.

            '''

        fig, ax = plt.subplots(2, 3, figsize=(30, 20))

        orbit_lines = []
        galaxy_points = []
        data_combos = [[self.x, self.y], [self.x, self.z], [self.y, self.z], [self.vx, self.vy], [self.vx, self.vz], [self.vy, self.vz]]

        position_xlim, position_ylim = pu.find_limits_multiple_axes(data_combos[0:3], 0.05)
        velocity_xlim, velocity_ylim = pu.find_limits_multiple_axes(data_combos[3:], 0.05)
        
        max_y_lim = np.max(np.abs(position_ylim))
        max_x_lim = np.max(np.abs(position_xlim))
        pos_lims = (-np.max([max_x_lim, max_y_lim]), np.max([max_x_lim, max_y_lim]))
        
        max_vy_lim = np.max(np.abs(velocity_ylim))
        max_vx_lim = np.max(np.abs(velocity_xlim))
        vel_lims = (-np.max([max_vx_lim, max_vy_lim]), np.max([max_vx_lim, max_vy_lim]))

        labels = [['$x$ (kpc)', '$y$ (kpc)'], ['$x$ (kpc)', '$z$ (kpc)'], ['$y$ (kpc)', '$z$ (kpc)'], ['$v_x$ (km/s)', '$v_y$ (km/s)'], ['$v_x$ (km/s)', '$v_z$ (km/s)'], ['$v_y$ (km/s)', '$v_z$ (km/s)']]

        for i in range(3):
            ax[0][i].set_xlim(pos_lims)
            ax[0][i].set_ylim(pos_lims)
            ax[1][i].set_xlim(vel_lims)
            ax[1][i].set_ylim(vel_lims)
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
        self.radial_tail_angles_3d_deg = self.original_orbit.tail_3d_radial_angle_deg

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

        tail_angles_2d = np.arctan2(self.tail_angles_2d_vectors[:,1], self.tail_angles_2d_vectors[:,0])
        # towards_cluster_center_angle = np.arctan2(-self.projected_y.value, -self.projected_x.value)
        # tail_angles_2d_relative = tail_angles_2d - towards_cluster_center_angle
        # self.tail_angles_2d_relative_wrapped = (tail_angles_2d_relative + np.pi) % (2*np.pi) - np.pi
        # self.tail_angles_2d_relative_deg = np.degrees(self.tail_angles_2d_relative_wrapped)
        cluster_center_angle = np.arctan2(self.projected_y.value, self.projected_x.value)
        self.tail_angles_2d_relative = np.abs(tail_angles_2d - cluster_center_angle) # this gives the angle relative to the direction AWAY from the cluster center
        self.tail_angles_2d_relative_deg = 180 - np.degrees(self.tail_angles_2d_relative) # subtract 180 degrees to get the angle relative to direction towards cluster center

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

    def comparison_plots(self, time_index=None):

        fig, ax = plt.subplots(1, 1)
        ax.set_title('Comparing 2D and 3D Tail Angles for Given Viewing Angle')

        ax.plot(self.t, self.radial_tail_angles_3d_deg, label='3D tail angle\n(relative to radial direction)')
        ax.plot(self.t, self.tail_angles_2d_relative_deg, label='Projected tail angle on the sky\n(relative to direction to cluster center)')
        
        if time_index != None:
            ax.plot(self.t[time_index], self.radial_tail_angles_3d_deg[time_index], '.')
            ax.plot(self.t[time_index], self.tail_angles_2d_relative_deg[time_index], '.')
            ax.axvline(self.t[time_index].value, linestyle='--')

        lims_2d = pu.find_axes_limits(self.radial_tail_angles_3d_deg, 0.75)
        lims_3d = pu.find_axes_limits(self.tail_angles_2d_relative_deg, 0.75)
        lower_lim = np.min([lims_2d[0], lims_3d[0]])
        upper_lim = np.max([lims_2d[1], lims_3d[1]])
        ax.set_ylim(lower_lim, upper_lim)

        ax.set_xlabel('Time (Myr)')
        ax.set_ylabel('Tail angle ($^{\circ}$)')
        ax.grid()
        ax.legend(fontsize=12)

        # plt.plot

        return fig
