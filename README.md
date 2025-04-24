# OrbitTails

Welcome to OrbitTails, a user-friendly Python package for simulating the orbits of ram pressure stripped galaxies in clusters and calculating tail angles. OrbitTails allows for easy comparison of 2D and 3D tail angles, facilitating our understanding of observational studies of ram pressure stripped galaxies.

## Background

Ram pressure stripping is the removal of gas from galaxies orbitting in clusters due to the pressure differential between a galaxy's interstellar medium and surrounding galactic medium. This process is key to the evolution of galaxies in large clusters, as complete gas removal quenches star formation. Ram pressure stripped galaxies form tails of stripped gas that extend behind the galaxy along its orbital path. Measurements of the angle of this tail, particularly relative to the cluster center, are key to understanding the evolution of ram pressure stripping during a galaxy's orbit, classifying ram pressure stripped galaxies, and elucidating how and why ram pressure stripping affects galaxies.

However, such measurements are complicated because a galaxy's apparent tail angle on the sky may not match up with the reality of its three-dimensional directional or give us a clear understanding of the galaxy's three-dimensional motion in the cluster. OrbitTails aims to help observational astronomers understand the relationship between three-dimensional orbital motion and tail angle and observed two-dimensional angles in the sky. Users can simulate galaxy orbits in a cluster potential and compare the galaxy's three-dimensional tail direction with the observed two-dimensional angles at a variety of viewing angles throughout the entire orbit.

## Installation and Usage

### Streamlit Web App

An interactive web app developed with and hosted by **Streamlit** is the most user-friendly way to interact with OrbitTails. This deployment can be accessed on the internet without having to download or clone this git repository:

[orbittails.streamlit.app](https://orbittails.streamlit.app)

## Local Streamlit App

Using the Streamlit app locally offers the same interface with slightly higher performance. To run it, ensure that git is set up with SSH keys, then clone the repository:

```
git clone
```

## Basic Functionality

 - Generate a gravitational potential that approximates that of the galaxy cluster of interest.
     - For the Coma cluster, we adopt a Navarro-Frenk-White potential with Virial Mass $1.26\times10^{15} M_{sun}$ and concentration 4 (Lokas & Mamon 2003).
 - Simulate the orbit of a galaxy with user-specified initial conditions in the cluster potential.
 - Compute 3D tail directions throughout the orbit.
 - Specify a line-of-sight viewing direction, and project the orbit and tail directions into the viewer's 2D perspective. 
     - Construct a coordinate frame, with origin at cluster center, that defines the viewer's perspective/line-of-sight.
     - Rotate/transform orbits from the Cartesian coordinate basis into the viewer's coordinate system.
     - Convert to two dimensions by considering only components in the viewing plane, not along the line of sight.
 - Plot orbits in Cartesian coordinate frames and the coordinate system of the viewer, and compare 2D tail angles with 3D directions.

 ## Documentation and Tutorials

 Navigate to the `/tutorial` directory and access `OrbitTails_tutorial.py` for example usage of the OrbitTails package in Python. 

 All methods, classes, and scripts are well-commented and contain descriptive docstrings.