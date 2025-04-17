import astropy.units as u
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from gala import potential as gp
from gala import dynamics as gd
from gala import integrate as gi
from gala.units import galactic
import streamlit.components.v1 as components

import matplotlib as mpl

from orbit import Orbit

mpl.rcParams['animation.embed_limit'] = 100

st.title('OrbitTails Web App')

st.sidebar.header('Navarro-Frenk-White Cluster Potential Parameters')
M200 = st.sidebar.number_input('Virial Mass ($M_{\odot}$)', value=1.26e15, format='%e', step=0.01e15)
c = st.sidebar.number_input('Concentration', value=4)

st.sidebar.header('Galaxy Parameters')
coord_sys = st.sidebar.selectbox(label='Coordinate system', options=['Cartesian', 'Spherical', 'Cylindrical'], index=0)
st.sidebar.subheader('Initial position')
if coord_sys=='Cartesian':
    x0 = st.sidebar.number_input('$x$ (kpc)', value=400)
    y0 = st.sidebar.number_input('$y$ (kpc)', value=0)
    z0 = st.sidebar.number_input('$z$ (kpc)', value=0)
st.sidebar.subheader('Initial Velocity')
if coord_sys=='Cartesian':
    v_x0 = st.sidebar.number_input('$v_x$ (km/s)', value=50)
    v_y0 = st.sidebar.number_input('$v_y$ (km/s)', value=1200)
    v_z0 = st.sidebar.number_input('$v_z$ (km/s)', value=100)

if (
    "host_potential" not in st.session_state or
    st.session_state.M200 != M200 or
    st.session_state.c != c
):
    st.session_state.host_potential = gp.NFWPotential.from_M200_c(M200 * u.Msun, c, units=galactic)
    st.session_state.M200 = M200
    st.session_state.c = c


# host_potential = gp.NFWPotential.from_M200_c(M200 * u.Msun, c, units=galactic)
st.write(f"Initialized cluster potential with Virial Radius {M200:.4e} and concentration {c}")
# improve this grid
x = np.linspace(-1000, 1000, 100)
z = np.linspace(-1000, 1000, 100)
fig = st.session_state.host_potential.plot_contours(grid=(x, 1., z))
st.write(fig)

st.write('Calculating orbit...')

if (
    "intial_conditions" not in st.session_state or
    st.session_state.x0 != x0 or
    st.session_state.y0 != y0 or
    st.session_state.z0 != z0 or
    st.session_state.v_x0 != v_x0 or
    st.session_state.v_y0 != v_y0 or
    st.session_state.v_z0 != v_z0
):
    st.session_state.initial_conditions = gd.PhaseSpacePosition(pos=[x0,y0,z0] * u.kpc,
                                           vel=[v_x0,v_y0,v_z0] * u.km/u.s)
    st.session_state.galaxy_orbit = Orbit(st.session_state.host_potential, st.session_state.initial_conditions)
    st.session_state.x0 = x0
    st.session_state.y0 = y0
    st.session_state.z0 = z0
    st.session_state.v_x0 = v_x0
    st.session_state.v_y0 = v_y0
    st.session_state.v_z0 = v_z0

# initial_conditions = gd.PhaseSpacePosition(pos=[x0,y0,z0] * u.kpc,
#                                            vel=[v_x0,v_y0,v_z0] * u.km/u.s)
initial_conditions = st.session_state.initial_conditions
galaxy_orbit = st.session_state.galaxy_orbit
# galaxy_orbit = Orbit(st.session_state.host_potential, initial_conditions)

# fig = galaxy_orbit.plot_orbit(['x', 'y', 'z'])
# st.pyplot(fig)
# fig = galaxy_orbit.plot_orbit(['v_x', 'v_y', 'v_z'])
# st.pyplot(fig)

# trying the animation
# galaxy_orbit.Animate_Orbit()
# components.html(galaxy_orbit.anim.to_jshtml(), height=1000)

# calculate tail angles
st.subheader('View 3D Tail Angles')
galaxy_orbit.calc_3d_tail_angles()
angle_time = st.select_slider('Time since initial conditions (Myr)', options=list(galaxy_orbit.t), value=0.)
angle_time_index = list(galaxy_orbit.t).index(angle_time)
fig = galaxy_orbit.plot_orbits_not_animated(angle_time_index, with_tails=True)
st.pyplot(fig)
st.write(f'3D tail angle unit vector: {galaxy_orbit.tail_angle_unit_vectors[angle_time_index, :]}')

st.subheader('2D Tail Angle Projections')
viewing_angle_sys = st.selectbox(label='Choose how to input viewing angle:', options=['Altitude/Azimuth', 'RA/Dec', '3D Vector'])
if viewing_angle_sys=='Altitude/Azimuth':
    st.number_input('Altitude', value=0, min_value=0, max_value=180)
if viewing_angle_sys=='3D Vector':
    view_x = st.number_input('$x$ component', value=0)
    view_y = st.number_input('$y$ component', value=0)
    view_z = st.number_input('$z$ component', value=0)
tail_angle_vectors2d = galaxy_orbit.calc_2d_tail_angles_and_orbit(view_dir = [view_x, view_y, view_z])