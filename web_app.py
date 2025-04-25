import astropy.units as u
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from gala import potential as gp
from gala import dynamics as gd
from gala import integrate as gi
from gala.units import galactic
import streamlit.components.v1 as components

from transforms import *

import matplotlib as mpl

from orbit import Orbit
from orbit import Projected_Orbit
# from Projected_Orbit import Projected_Orbit

mpl.rcParams['animation.embed_limit'] = 100
plt.rcParams['font.family']='serif'
plt.rcParams['font.size']=14

st.title('OrbitTails Web App')

st.write("Use this software to to simulate the orbits of galaxies in large galaxy clusters and calculate the " \
        "angle of the galaxy's ram pressure stripped tail in three dimensions and on the plane of the sky.")

st.sidebar.header('Navarro-Frenk-White Cluster Potential Parameters')
st.sidebar.write('The default input parameters approximate the gravitational potential of the Coma galaxy cluster')
M200 = st.sidebar.number_input('Virial Mass ($M_{\odot}$)', value=1.26e15, format='%e', step=0.01e15, min_value=0.)
c = st.sidebar.number_input('Concentration', value=4., min_value=0.)

st.sidebar.header('Galaxy Parameters')
timestep = st.sidebar.number_input('Integration timestep (Myr)', value=2, min_value=0)
n_timesteps = st.sidebar.number_input('Number of timesteps', value=2000, min_value=0)
initial_conditions_dict = {}
coord_sys = st.sidebar.selectbox(label='Coordinate system', options=['Cartesian', 'Cylindrical'], index=0)
st.sidebar.subheader('Initial position')
if coord_sys=='Cartesian':
    initial_conditions_dict['x0'] = st.sidebar.number_input('$x$ (kpc)', value=400)
    initial_conditions_dict['y0'] = st.sidebar.number_input('$y$ (kpc)', value=0)
    initial_conditions_dict['z0'] = st.sidebar.number_input('$z$ (kpc)', value=0)
if coord_sys=='Cylindrical':
    initial_conditions_dict['r0'] = st.sidebar.number_input('Radial Position ($r$) [kpc]', value=400)
    initial_conditions_dict['phi0'] = st.sidebar.number_input('Azimuthal angle ($\phi$) [$^\circ$]', value=0)
    initial_conditions_dict['z0'] = st.sidebar.number_input('Height ($z$) [kpc]', value=0)
st.sidebar.subheader('Initial Velocity')
if coord_sys=='Cartesian':
    initial_conditions_dict['v_x0'] = st.sidebar.number_input('$v_x$ (km/s)', value=50)
    initial_conditions_dict['v_y0'] = st.sidebar.number_input('$v_y$ (km/s)', value=1200)
    initial_conditions_dict['v_z0'] = st.sidebar.number_input('$v_z$ (km/s)', value=100)
if coord_sys=='Cylindrical':
    initial_conditions_dict['v_r0'] = st.sidebar.number_input('Radial Velocity ($v_r$) [km/s]', value=400)
    initial_conditions_dict['v_phi_tan0'] = st.sidebar.number_input('Tangential (azimuthal) velocity ($v_{\phi}$) [km/s]', value=1200)
    initial_conditions_dict['v_z0'] = st.sidebar.number_input('Vertical velocity ($v_z$) [km/s]')

if (
    "host_potential" not in st.session_state or
    st.session_state.get("M200") != M200 or
    st.session_state.get("c") != c
):
    st.session_state.host_potential = gp.NFWPotential.from_M200_c(M200 * u.Msun, c, units=galactic)
    st.session_state.M200 = M200
    st.session_state.c = c


# host_potential = gp.NFWPotential.from_M200_c(M200 * u.Msun, c, units=galactic)
st.write(f"Initialized cluster potential with Virial Mass {M200:.4e}" + " $M_{\odot}$ " + f"and concentration {c}")
# improve this grid
x = np.linspace(-1000, 1000, 100)
z = np.linspace(-1000, 1000, 100)
fig = st.session_state.host_potential.plot_contours(grid=(x, 1., z))
st.write(fig)

recompute = (
    st.session_state.get("coord_sys") != coord_sys or
    st.session_state.get("n_timesteps") != n_timesteps or
    st.session_state.get("time_step") != timestep or
    st.session_state.get("initial_conditions_dict") != initial_conditions_dict
)

if recompute:
    if coord_sys == "Cartesian":
        pos = [initial_conditions_dict['x0'], initial_conditions_dict['y0'], initial_conditions_dict['z0']] * u.kpc
        vel = [initial_conditions_dict['v_x0'], initial_conditions_dict['v_y0'], initial_conditions_dict['v_z0']] * u.km/u.s
        initial_conditions = gd.PhaseSpacePosition(pos=pos, vel=vel)

    elif coord_sys == "Cylindrical":
        rho = initial_conditions_dict['r0'] * u.kpc
        phi = initial_conditions_dict['phi0'] * u.deg
        z = initial_conditions_dict['z0'] * u.kpc

        v_rho = initial_conditions_dict['v_r0'] * u.km/u.s
        v_phi_tangential = initial_conditions_dict['v_phi_tan0']*u.km/u.s # angular velocity!
        v_z = initial_conditions_dict['v_z0'] * u.km/u.s

        initial_conditions = create_cylindrical_initial_conditions(rho, phi, z, v_rho, v_phi_tangential, v_z)
    
    st.session_state.initial_conditions = initial_conditions
    st.session_state.galaxy_orbit = Orbit(st.session_state.host_potential, initial_conditions, 
                                          n_steps=n_timesteps, dt=timestep)
    st.session_state.coord_sys = coord_sys
    st.session_state.initial_conditions_dict = initial_conditions_dict
    st.session_state.n_timesteps = n_timesteps
    st.session_state.time_step = timestep
    st.session_state.galaxy_orbit.calc_3d_tail_angles()



# if (
#     "initial_conditions" not in st.session_state or
#     st.session_state.get("x0") != x0 or
#     st.session_state.get("y0") != y0 or
#     st.session_state.get("z0") != z0 or
#     st.session_state.get("v_x0") != v_x0 or
#     st.session_state.get("v_y0") != v_y0 or
#     st.session_state.get("v_z0") != v_z0 or
#     st.session_state.get("n_timesteps") != n_timesteps or
#     st.session_state.get("time_step") != timestep
# ):
#     st.session_state.initial_conditions = gd.PhaseSpacePosition(pos=[x0,y0,z0] * u.kpc,
#                                            vel=[v_x0,v_y0,v_z0] * u.km/u.s)
#     st.session_state.galaxy_orbit = Orbit(st.session_state.host_potential, st.session_state.initial_conditions, 
#                                           n_steps=n_timesteps, dt=timestep)
#     st.session_state.x0 = x0
#     st.session_state.y0 = y0
#     st.session_state.z0 = z0
#     st.session_state.v_x0 = v_x0
#     st.session_state.v_y0 = v_y0
#     st.session_state.v_z0 = v_z0
#     st.session_state.n_timesteps = n_timesteps
#     st.session_state.time_step = timestep

# # initial_conditions = gd.PhaseSpacePosition(pos=[x0,y0,z0] * u.kpc,
# #                                            vel=[v_x0,v_y0,v_z0] * u.km/u.s)
# initial_conditions = st.session_state.initial_conditions
# galaxy_orbit = st.session_state.galaxy_orbit
# galaxy_orbit = Orbit(st.session_state.host_potential, initial_conditions)

# fig = galaxy_orbit.plot_orbit(['x', 'y', 'z'])
# st.pyplot(fig)
# fig = galaxy_orbit.plot_orbit(['v_x', 'v_y', 'v_z'])
# st.pyplot(fig)

# trying the animation
# galaxy_orbit.Animate_Orbit()
# components.html(galaxy_orbit.anim.to_jshtml(), height=1000)

# calculate tail angles
galaxy_orbit = st.session_state.galaxy_orbit
st.subheader('View 3D Tail Angles')
angle_time = st.select_slider('Time since initial conditions (Myr)', options=list(galaxy_orbit.t), value=0., )
angle_time_index = list(galaxy_orbit.t).index(angle_time)
fig = galaxy_orbit.plot_orbits_not_animated(angle_time_index, with_tails=True)
st.pyplot(fig)
vec_3d = galaxy_orbit.tail_angle_unit_vectors[angle_time_index, :]
st.markdown(f"**3D Tail Direction Unit Vector**: ({vec_3d[0]:.2f}, {vec_3d[1]:.2f}, {vec_3d[2]:.2f})")
st.write(f'We report the tail angle in degrees as the angle between the tail direction vector and the radial direction to the cluster center: {galaxy_orbit.tail_3d_radial_angle_deg[angle_time_index]:.2f}' + '$^{\circ}$')

st.subheader('2D Tail Angle Projections')
st.write("Here, choose a line-of-sight viewing angle. The viewing angle, whether in vector or altitude/azimuth form, corresponds to the" \
"direction the viewer is looking, in a coordinate system whose origin is at cluster center. Input as a 3-D vector in " \
"Cartesian coordinates or azimuth and elevation angles.")
viewing_angle_sys = st.selectbox(label='Choose how to input viewing angle:', options=['Altitude/Azimuth', '3D Vector'])
if viewing_angle_sys=='Altitude/Azimuth':
    alt_angle = st.number_input('Altitude', value=0, min_value=-90, max_value=90)
    az_angle = st.number_input('Azimuth', value=0, min_value=0, max_value=360)

    viewing_angle_vector = viewing_vector_from_angles(az_angle, alt_angle)

if viewing_angle_sys=='3D Vector':
    view_x = st.number_input('$x$ component', value=0)
    view_y = st.number_input('$y$ component', value=0)
    view_z = st.number_input('$z$ component', value=1)

    viewing_angle_vector = [view_x, view_y, view_z]

angle_time = st.select_slider('Time since initial conditions (Myr)', options=list(galaxy_orbit.t), value=0., key='2dtime')
angle_time_index = list(galaxy_orbit.t).index(angle_time)

orbit_projected = galaxy_orbit.project_orbit_tail_angles(view_dir = viewing_angle_vector)
projected_figure = orbit_projected.plot_projected_orbit(angle_time_index)

st.write("2D tail angle is measured relative to the diection towards cluster center: An angle of $0^{\circ}$ means that the tail is pointing directly " \
"toward the cluster center, while an angle of $180^{\circ}$ means it is pointing directly away from the cluster center.")
vec_2d = orbit_projected.tail_angles_2d_vectors[angle_time_index, :]
st.markdown(f"**2D projected direction of 3D tail in vector form**: ({vec_2d[0]:.2f}, {vec_2d[1]:.2f})")
# st.write(f"2D projected direction of 3D tail in vector form: {orbit_projected.tail_angles_2d_vectors[angle_time_index, :]}")
st.markdown(f"**2D tail angle:** {orbit_projected.tail_angles_2d_relative_deg[angle_time_index]:.2f}$^\circ$")
st.pyplot(projected_figure)

st.subheader('Comparing 2D and 3D Tail Angles')
fig = orbit_projected.comparison_plots(angle_time_index)
st.pyplot(fig)