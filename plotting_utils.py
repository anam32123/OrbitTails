import numpy as np
import matplotlib.pyplot as plt
import transforms as angles

def find_axes_limits(data, padding=0.05):

    """
    Finds ideal axes limits for a given dataset and certain padding percentage.
    These limits should be applied in a plot to the axis corresponding to the dimension in which the dataset is plotted.

        Parameters
        ----------
        data: numpy.ndarray
            NumPy array containing data to be plotted.
        padding: float
            Fraction of the total data range to use as padding beyond the min and max values. Default: 0.05

        Returns
        -------
        lower_limit : float
            The padded lower bound for the axis.
        upper_limit : float
            The padded upper bound for the axis.
    """

    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    range_padding = np.abs(data_max - data_min) * padding

    return data_min - range_padding, data_max + range_padding

def find_limits_multiple_axes(data, padding=0.05):
    """
    Compute global axis limits for multiple 2D data plots with consistent padding.

    This function takes a list of [x, y] data pairs (e.g., for subplots showing different
    2D projections) and computes the minimum and maximum x and y limits across all pairs,
    with a consistent fractional padding applied to each axis.

    Parameters
    ----------
    data : list of list of astropy.units.Quantity or numpy.ndarray
        A list where each element is a [x_data, y_data] pair. Each `x_data` and `y_data` should be
        1D arrays with units (e.g., astropy Quantities) or plain NumPy arrays of the same shape.
    padding : float, optional
        Fraction of the data range to use as padding beyond the min and max values for both axes. Default: 0.05

    Returns
    -------
    x_limits : list of float
        A list [x_min, x_max] representing the padded global x-axis limits across all input pairs.
    y_limits : list of float
        A list [y_min, y_max] representing the padded global y-axis limits across all input pairs.
    """
    position_y_lowerlim = np.zeros(3)
    position_y_upperlim = np.zeros(3)
    position_x_lowerlim = np.zeros(3)
    position_x_upperlim = np.zeros(3)

    for i in range(len(data)):
        position_x_lowerlim[i], position_x_upperlim[i] = find_axes_limits(data[i][0].value, padding)
        position_y_lowerlim[i], position_y_upperlim[i] = find_axes_limits(data[i][1].value, padding)

    position_x_lowerlim = np.nanmin(position_x_lowerlim)
    position_x_upperlim = np.nanmax(position_x_upperlim)
    position_y_lowerlim = np.nanmin(position_y_lowerlim)
    position_y_upperlim = np.nanmax(position_y_upperlim)

    return [position_x_lowerlim, position_x_upperlim], [position_y_lowerlim, position_y_upperlim]



def plot_orbit(ax, x_data, y_data, time, time_index):

    """
    Plot a galaxy's orbit on a 2D subplot with annotations for current position and time.

    This function plots the full orbit trajector and the current position of the
    galaxy at a specified time index, and marks the cluster center. It also displays the
    simulation time as an annotation within the plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Matplotlib Axes object on which to plot the orbit.
    x_data : array-like
        The x-coordinates of the galaxy's position over time (can be a AstroPy Quantity or NumPy array).
    y_data : array-like
        The y-coordinates of the galaxy's position over time (AstroPy Quantity or NumPy array).
    time : array-like
        Array of time values corresponding to each point in the orbit.
    time_index : int
        Index of the current timestep to highlight in the plot.

    Returns
    -------
    None
    """
    
    ax.text(0.05, 0.95, time[time_index], transform=ax.transAxes, fontsize=12,
                              verticalalignment='top', horizontalalignment='left', 
                              bbox=dict(facecolor='white', alpha=0.5))
    ax.plot(0, 0, color="black", marker="x", zorder=1)
    ax.plot(x_data[0], y_data[0], 'x', color='red', zorder=1)
    ax.plot(x_data[:time_index], y_data[:time_index], alpha=0.75, zorder=2)
    ax.plot(x_data[time_index], y_data[time_index], '*', markersize=10, color='orange', zorder=4)
    ax.plot(x_data, y_data, '-', color='darkgray', linewidth=0.5, zorder=1, alpha=0.75)
    ax.set_aspect('equal')

    return

def calc_tail_line(x_data, y_data, tail_vector, time_index, tail_length=100, tail_distance=100):

    """
    This function generates a line segment representing the galaxy's tail vector
    in a 2D projected frame. The tail is constructed as a series of points extending
    from the galaxy's current position in the direction of the given tail vector.

    Parameters
    ----------
    x_data : array-like
        1D array of x-coordinates of the galaxy over time.
    y_data : array-like
        1D array of y-coordinates of the galaxy over time.
    tail_vector : array-like
        A 2D vector representing the projected tail direction (does not need to be normalized).
    time_index : int
        The index of the timestep at which to place the tail's origin (i.e., the galaxy's current position).
    tail_length : int, optional
        Number of points to generate along the tail line. Default is 100.
    tail_distance : float, optional
        Maximum physical distance from the galaxy to extend the tail line. Default is 100.

    Returns
    -------
    tail_x_points_selected : numpy.ndarray
        1D array of x-coordinates for the tail segment to be plotted.
    tail_y_points_selected : numpy.ndarray
        1D array of y-coordinates for the tail segment to be plotted.
    """

    # get current x-y position of the galaxy
    galaxy_x = x_data[time_index].value
    galaxy_y = y_data[time_index].value

    steps = np.linspace(0, tail_distance, tail_length)

    normalized_2d_tail_angle_unit_vector = angles.norm_vector(tail_vector)

    # walk along in x and y to generate the vector
    tail_x_points = galaxy_x + steps * normalized_2d_tail_angle_unit_vector[0]
    tail_y_points = galaxy_y + steps * normalized_2d_tail_angle_unit_vector[1]

    # only choose points on the tail that are within 50 kpc of the galaxy position
    tail_point_selection = np.abs(tail_y_points - y_data[time_index].value) < 50
    tail_x_points_selected = tail_x_points[tail_point_selection]
    tail_y_points_selected = tail_y_points[tail_point_selection]

    return tail_x_points_selected, tail_y_points_selected