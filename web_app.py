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

host_potential = gp.NFWPotential.from_M200_c(M200 * u.Msun, c, units=galactic)
st.write(f"Initialized cluster potential with Virial Radius {M200:.4e} and concentration {c}")
x = np.linspace(-1000, 1000, 100)
z = np.linspace(-1000, 1000, 100)
fig = host_potential.plot_contours(grid=(x, 1., z))
st.write(fig)

st.write('Calculating orbit...')
initial_conditions = gd.PhaseSpacePosition(pos=[x0,y0,z0] * u.kpc,
                                           vel=[v_x0,v_y0,v_z0] * u.km/u.s)
galaxy_orbit = Orbit(host_potential, initial_conditions)

fig = galaxy_orbit.plot_orbit(['x', 'y', 'z'])
st.pyplot(fig)
fig = galaxy_orbit.plot_orbit(['v_x', 'v_y', 'v_z'])
st.pyplot(fig)

# trying the animation
# galaxy_orbit.Animate_Orbit()
# components.html(galaxy_orbit.anim.to_jshtml(), height=1000)

# calculate tail angles
st.subheader('View 3D Tail Angles')
galaxy_orbit.calc_3d_tail_angles()
angle_time = st.slider('Time since initial conditions (Myr)', min_value=0., max_value=np.nanmax(galaxy_orbit.t).value, value=0.)
st.write(f'3D tail angle unit vector: {galaxy_orbit.tail_angle_unit_vectors[:, angle_time]}')

st.subheader('2D Tail Angle Projections')
st.number_input('Viewing angle inclination ($^\circ$)', value=0, min_value=0, max_value=180)