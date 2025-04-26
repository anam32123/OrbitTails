import numpy as np
from gala import potential as gp
from gala import dynamics as gd
from gala import integrate as gi
from gala.units import galactic

from astropy import units as u

from matplotlib import pyplot as plt

import matplotlib.animation as animation
import matplotlib as mpl

from . import plotting_utils as pu
# from Projected_Orbit import Projected_Orbit
from .transforms import *

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

        Wrapper for `gala.potential.Hamiltonian.integrate_orbit`
        
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
        
        print(f"Simulating an orbit in the host potential with intial conditions position = {initial_conditions.xyz}, velocity = {initial_conditions.v_xyz}")

        self.orbit = gp.Hamiltonian(Gala_Potential).integrate_orbit(initial_conditions, dt=dt, n_steps=n_steps)
        self.initial_conditions = initial_conditions

        self.x, self.y, self.z = self.orbit.xyz
        self.vx, self.vy, self.vz = self.orbit.v_xyz.to(u.km/u.s)
        self.t = self.orbit.t
        self.host_potential = Gala_Potential
    
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
        as the 3D angle between the tail direction vector and the radial direction towards cluster center, using
        the dot product. An angle of 0 degrees corresponds to a tail pointing directly toward the cluster center, 
        and 180 degrees to a tail pointing directly away.

            Sets
            ----
            self.velocity_magnitudes : numpy.ndarray
                A 1D array containing the magnitude of the galaxy's velocity at each timestep.
            self.velocity_unit_vectors : numpy.ndarray
                A 2D array of shape (N, 3), where each row is the unit velocity vector at a given timestep.
            self.tail_angle_unit_vectors : numpy.ndarray
                A 2D array of shape (N, 3), where each row is a unit vector pointing in the tail direction (opposite velocity).
            self.tail_3d_radial_angle_rad: numpy.ndarray
                An array of 3D tail angles in radians for the galaxy at each point in its orbit, measured as the angle between the
                galaxy tail and the radial direction to cluster center.
            self.tail_3d_radial_angle_deg
                An array of 3D tail angles in degrees for the galaxy at each point in its orbit, measured as the angle between the
                galaxy tail and the radial direction to cluster center.
            
            Returns
            -------
            None
        """

        # create velocity unit vectors
        vx, vy, vz = self.orbit.v_xyz.to(u.km/u.s)
        
        vx = np.array(vx)
        vy = np.array(vy)
        vz = np.array(vz)
        
        velocity_vectors = np.vstack((vx, vy, vz))
        
        # normalize the velocity vectors
        velocity_vectors = velocity_vectors.T
        self.velocity_magnitudes = np.apply_along_axis(np.linalg.norm, axis=1, arr=velocity_vectors)
        velocity_magnitudes = self.velocity_magnitudes[:, np.newaxis]
        self.velocity_unit_vectors = velocity_vectors/velocity_magnitudes

        # define tail angle vectors as the opposite of velocity vectors
        self.tail_angle_unit_vectors = -self.velocity_unit_vectors

        # forming the angle between the tail and the radial direction
        # Build the radial unit‐vector (galaxy → cluster) assuming cluster at origin
        pos = np.vstack([self.x.value, self.y.value, self.z.value]).T
        radial_vec = -pos                             # point toward the cluster, rather than galaxy
        radii = np.linalg.norm(radial_vec, axis=1, keepdims=True)   
        radii[radii == 0] = 1.0             # avoids division by 0                         
        radial_unit_vec = radial_vec / radii

        # Dot‐product gives cos(θ)
        cos_theta = np.einsum('ij,ij->i', self.tail_angle_unit_vectors, radial_unit_vec) # performs row-wise dot product on the data
        cos_theta = np.clip(cos_theta, -1.0, 1.0)                  # guard against numerical overshoot

        # calculate and store the angles
        theta_rad = np.arccos(cos_theta)
        theta_deg = np.degrees(theta_rad)

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
    
    def plot_orbits_not_animated(self, time_index, with_tails=False, include_contours=False):
        
        """
        Plot 2D projections of the galaxy's orbit and velocity in Cartesian coordinates at a specific timestep.

        This function produces six subplots showing all combinations of position and velocity components (x, y, z),
        with an optional overlay of the tail direction vector at a specified time index.

            Parameters
            ----------
            time_index: int
                Index into the time array `self.t` indicating the moment at which to show the orbit snapshot and (optionally) the tail direction.            
            with_tails: boolean, optional
                If True, plots the tail direction for the timestep specified by time_index together with the galaxy and orbit. 
                Requires that calc_3d_tail_angles has already been run for the given orbit. Default: False
            
            Returns
            -------
            fig: matplotlib.pyplot.Figure
                Figure object containing the galaxy orbit position and velocity projection plots.

        """

        fig, ax = plt.subplots(2, 3, figsize=(30, 20))

        # define combinations of datasets to plot on each axes
        data_combos = [[self.x, self.y], [self.x, self.z], [self.y, self.z], [self.vx, self.vy], [self.vx, self.vz], [self.vy, self.vz]]

        # calculate good x/y limits for the plots based on the datasets
        position_xlim, position_ylim = pu.find_limits_multiple_axes(data_combos[0:3], 0.05)
        velocity_xlim, velocity_ylim = pu.find_limits_multiple_axes(data_combos[3:], 0.05)
        
        # choosel the furthest limits of those calculated, and make it symmetrical
        max_y_lim = np.max(np.abs(position_ylim))
        max_x_lim = np.max(np.abs(position_xlim))
        pos_lims = (-np.max([max_x_lim, max_y_lim]), np.max([max_x_lim, max_y_lim]))
        
        max_vy_lim = np.max(np.abs(velocity_ylim))
        max_vx_lim = np.max(np.abs(velocity_xlim))
        vel_lims = (-np.max([max_vx_lim, max_vy_lim]), np.max([max_vx_lim, max_vy_lim]))

        for i in range(3):
            ax[0][i].set_xlim(pos_lims)
            ax[0][i].set_ylim(pos_lims)
            ax[1][i].set_xlim(vel_lims)
            ax[1][i].set_ylim(vel_lims)
            ax[1][i].set_ylim(velocity_ylim)

        labels = [['$x$ (kpc)', '$y$ (kpc)'], ['$x$ (kpc)', '$z$ (kpc)'], ['$y$ (kpc)', '$z$ (kpc)'], ['$v_x$ (km/s)', '$v_y$ (km/s)'], ['$v_x$ (km/s)', '$v_z$ (km/s)'], ['$v_y$ (km/s)', '$v_z$ (km/s)']]

        # plot the orbits on each axes by indexing into data_combos to get the right data
        current_data = 0
        for row in ax:
            for a in row:
                
                pu.plot_orbit(a, data_combos[current_data][0], data_combos[current_data][1], time=self.t, time_index=time_index)
                a.set_xlabel(labels[current_data][0])
                a.set_ylabel(labels[current_data][1])

                current_data += 1

        # add tails to the plots
        if with_tails:
            
            component_indices = {'$x$ (kpc)': 0, '$y$ (kpc)': 1, '$z$ (kpc)': 2}

            for i, a in enumerate(ax[0]):

                # indices for indexing into array of 3-D tail vectors
                x_index = component_indices[labels[i][0]]
                y_index = component_indices[labels[i][1]]

                # retrieve tail vector and transform it into a line anchored at the galaxy position
                tail_vector = self.tail_angle_unit_vectors[time_index, [x_index, y_index]]

                tail_x_points_selected, tail_y_points_selected = pu.calc_tail_line(data_combos[i][0], data_combos[i][1], tail_vector=tail_vector, time_index=time_index)
                
                a.plot(tail_x_points_selected, tail_y_points_selected, '-', color='red', zorder=3)

        if include_contours:
            
            pos_grid = np.linspace(pos_lims[0], pos_lims[1], 100)
            vel_grid = np.linspace(vel_lims[0], vel_lims[1], 100)

            self.host_potential.plot_contours(grid=(pos_grid, pos_grid, 1), ax=ax[0][0], color='green', alpha=0.25)
            self.host_potential.plot_contours(grid=(pos_grid, 1, pos_grid), ax=ax[0][1], color='green', alpha=0.25)
            self.host_potential.plot_contours(grid=(1, pos_grid, pos_grid), ax=ax[0][2], color='green', alpha=0.25)
            self.host_potential.plot_contours(grid=(vel_grid, vel_grid, 1), ax=ax[1][0], color='green', alpha=0.25)
            self.host_potential.plot_contours(grid=(vel_grid, 1, vel_grid), ax=ax[1][1], color='green', alpha=0.25)
            self.host_potential.plot_contours(grid=(1, vel_grid, vel_grid), ax=ax[1][2], color='green', alpha=0.25)

        return fig
    
    def project_orbit_tail_angles(self, view_dir):

        """
        Projects the orbit into a viewing frame defined by a line-of-sight direction, generating a 
        Projected_Orbit object containing the galaxy's orbit and tail angles as seen from a specified viewing direction.

            Parameters
            ----------
            view_dir: numpy.ndarray
                Numpy array of shape (3,) containing a vector pointing in the direction of the viewer's line of sight.

            Returns
            -------
            Projected_Orbit(self, view_dir): Projected_Orbit
                An instance of the Projected_Orbit class containing information about the galaxy orbit and tail angles projected into the given viewing frame.
        """

        return Projected_Orbit(self, view_dir)
    
class Projected_Orbit:

    """
    Represents a galaxy's 3D orbit and tail directions projected into a 2D viewing frame.
    
    Generates a 2D projection of the orbit and tail angles in the observer's frame based on an existing
    `Orbit` instance the the viewer's line of sight direction. Includes methods for plotting projected orbits
    and comparing projected 2D and 3D tail angles.

        Attributes
        ----------
        original_orbit: Orbit
            Instance of the `Orbit` class from which the Projected_Orbit instance is instantiated, containing information about the 3D galaxy orbit and tail angles.
        tail_angle_3d_unit_vectors: numpy.ndarray
            Array of shape (N, 3) where each row is a 3D unit vector representing the galaxy tail direction at each simulation timestep.
            Copied from `original_orbit`.
        radial_tail_angles_3d_deg: numpy.ndarray
            An array of 3D tail angles in degrees for each orbit timestep, measured as the angle between the
            galaxy tail and the radial direction to cluster center. Copied from `original_orbit`.
        view_dir_normed: numpy.ndarray
            A normalized 3D vector along the viewer's line of sight.
        rot_matrix: numpy.ndarray
            3x3 rotation matrix for transforming Cartesian coordinates and vectors into the viewer's coordinate frame.
        orbit_projected: numpy.ndarray
            Numpy array of shape (N, 2) containing projected x and y coordinates of the orbit in each row, for all timesteps.
        projected_x: numpy.ndarray
            1D array of the x-coordinates of the orbit in the 2D projected plane for all timesteps.
        projected_y: numpy.ndarray
            1D array of the y-coordinates of the orbit in the 2D projected plane for all timesteps.
        tail_angles_2d_vectors: numpy.ndarray
            Array of shape (N, 2) in which each row represents a 2D unit vector representing the tail direction projected in the viewing plane, for all timesteps.
        tail_angles_2d_relative_rad: numpy.ndarray
            Array of 2D tail angles in radians measured between the projected tail direction and the projection radial direction toward cluster center.
        tail_angles_2d_relative_deg: numpy.ndarray
            Array of 2D tail angles in degrees measured between the projected tail direction and the projection radial direction toward cluster center.

    """

    def __init__(self, original_orbit: Orbit, view_dir: np.ndarray):

        """
        Instantiates the Projected_Orbit class by performing a rotation and coordinate transformation of an existing `Orbit` instance 
        to the coordinate basis specified by line of sight viewing direction `view_dir`. The orbit and tail vectors are transformed into
        the viewer's 2D coordinate frame, and relevant 2D tail angles are computed.

            Parameters
            ----------
            original_orbit: Orbit
                Existing instance of the `Orbit` class containing information about the 3D galaxy orbit and tail angles.
            view_dir: numpy.ndarray
                A 3D vector along the viewer's line of sight
        """
        
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
        self.rot_matrix = np.vstack([right, up, self.view_dir_normed]).T # construct rotation matrix from basis vectors
        rotated_tail_angles = self.tail_angle_3d_unit_vectors.dot(self.rot_matrix)

        # rotate the orbit using the same matrix
        orbit_points = np.column_stack([self.original_orbit.x, self.original_orbit.y, self.original_orbit.z])
        self.orbit_projected = orbit_points.dot(self.rot_matrix)
        self.orbit_projected = self.orbit_projected[:, :2]
        self.projected_x = self.orbit_projected[:, 0]
        self.projected_y = self.orbit_projected[:, 1]

        # take only the x and y (up and right) components to get the projection in 2D
        self.tail_angles_2d_vectors = rotated_tail_angles[:, :2]
        self.tail_angles_2d_vectors = norm_vector(self.tail_angles_2d_vectors) # normalizing calculated vectors

        # calculate tail angles from direction vectors
        tail_angles_2d = np.arctan2(self.tail_angles_2d_vectors[:,1], self.tail_angles_2d_vectors[:,0])
        cluster_center_angle = np.arctan2(self.projected_y.value, self.projected_x.value)
        self.tail_angles_2d_relative_rad = np.abs(tail_angles_2d - cluster_center_angle) # this gives the angle relative to the direction AWAY from the cluster center
        self.tail_angles_2d_relative_deg = 180 - np.degrees(self.tail_angles_2d_relative_rad) # subtract 180 degrees to get the angle relative to direction towards cluster center

        self.t = self.original_orbit.t

    
    def plot_projected_orbit(self, time_index, with_tail=False):

        """
        Plot 2D projection of the galaxy's orbit and velocity in Cartesian coordinates, with an optional overlay of the tail direction vector at a specified time index.

            Parameters
            ----------
            time_index: int
                Index into the time array `self.t` indicating the moment at which to show the orbit snapshot and (optionally) the tail direction.            
            with_tails: boolean, optional
                If True, plots the tail direction for the timestep specified by time_index together with the galaxy and orbit. 
                Requires that calc_3d_tail_angles has already been run for the given orbit. Default: False
            
            Returns
            -------
            fig: matplotlib.pyplot.Figure
                Figure object containing the galaxy orbit position plot.

        """
        
        fig, ax = plt.subplots(1, 1)

        # set reasonable x/y axis limits for the plots
        y_lims = pu.find_axes_limits(self.projected_y.value, 0.1)
        x_lims = pu.find_axes_limits(self.projected_x.value, 0.1)
        max_y_lim = np.max(np.abs(y_lims))
        max_x_lim = np.max(np.abs(x_lims))
        lims = (-np.max([max_x_lim, max_y_lim]), np.max([max_x_lim, max_y_lim]))

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')

        # plot the projected orbit on the axis
        pu.plot_orbit(ax, self.projected_x, self.projected_y, self.t, time_index=time_index)

        # add tail to the plot based on calculated tail angle vectors
        if with_tail:
            tail_angle_vector = self.tail_angles_2d_vectors[time_index, :]
            tail_x, tail_y = pu.calc_tail_line(self.projected_x, self.projected_y, tail_angle_vector, time_index)
            ax.plot(tail_x, tail_y, '-', color='red', zorder=3)

        ax.set_xlabel('$x$ (kpc)')
        ax.set_ylabel('$y$ (kpc)')
        ax.set_title ("Orbit from viewer's perspective")

        return fig

    def comparison_plots(self, time_index=None):

        """
        Produces a plot for comparing true 3D tail angles and with their projected 2D counterparts.

            Parameters
            ----------
            time_index: int, optional
                Index into the time array `self.t` to create a vertical line indicating tail angles at a given point in the orbit.            
            
        """

        fig, ax = plt.subplots(1, 1)
        ax.set_title('Comparing 2D and 3D Tail Angles for Given Viewing Angle')

        # plot 2D and 3D tail angles
        ax.plot(self.t, self.radial_tail_angles_3d_deg, label='3D tail angle\n(relative to radial direction)', color='deepskyblue')
        ax.plot(self.t, self.tail_angles_2d_relative_deg, label='Projected tail angle on the sky\n(relative to direction to cluster center)', color='darkorchid')
        
        # add a vertical line indicating the selected time
        if time_index != None:
            ax.plot(self.t[time_index], self.radial_tail_angles_3d_deg[time_index], '.', color='deepskyblue')
            ax.plot(self.t[time_index], self.tail_angles_2d_relative_deg[time_index], '.', color='darkorchid')
            ax.axvline(self.t[time_index].value, linestyle='--', color='lightcoral')

        # calculate limits on the plot so there is room for the large legend
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
