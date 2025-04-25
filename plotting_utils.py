import numpy as np
import matplotlib.pyplot as plt
import transforms as angles

def find_axes_limits(data, padding=0.05):

    '''
    Finds ideal axes limits for a given dataset and certain padding percentage.
    These limits should be applied in a plot to the axis corresponding to the dimension in which the dataset is plotted.

        Parameters
        ----------
        data: numpy.ndarray
        Dataset for which we calculate outer limits
        padding: float
        Fraction of the total data range by which we wish to pad (to make m)
    '''
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    range_padding = np.abs(data_max - data_min) * padding

    return data_min - range_padding, data_max + range_padding

def find_limits_multiple_axes(data, padding):
    '''
    data should include [[xdata, ydata], [xdata, ydata], ...] in the number of pairs of data
    '''
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

# def vector_to_tail_line(tail_vector, gal_position, x_vals, length=50):
    
#     '''
#     A function to calculate a line to draw a tail on a galaxy plot, given a galaxy position (a 2-d point, doesn't matter whether its xy, xz, etc.).
    
#     '''
    
#     line_slope = tail_vector[1]/tail_vector[0]
    
#     # if np.abs(line_slope)<0.001:
#     #     print('Oops')
    
#     y_points = gal_position[1] + line_slope*(x_vals - gal_position[0])
    
#     return y_points

def plot_orbit(ax, x_data, y_data, time, time_index):

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

def calc_tail_line(ax, x_data, y_data, tail_vector, time_index, tail_length=100, tail_distance=100):

    # points for plotting the tail
    tail_x_points = x_data[time_index-20:time_index:-1]

    # solution: normalize 2-D projected vector
    galaxy_x = x_data[time_index].value
    galaxy_y = y_data[time_index].value

    steps = np.linspace(0, tail_distance, tail_length)  # negative to go backward

    normalized_2d_tail_angle_unit_vector = angles.norm_vector(tail_vector)

    tail_x_points = galaxy_x + steps * normalized_2d_tail_angle_unit_vector[0]
    tail_y_points = galaxy_y + steps * normalized_2d_tail_angle_unit_vector[1]

    tail_point_selection = np.abs(tail_y_points - y_data[time_index].value) < 50
    tail_x_points_selected = tail_x_points[tail_point_selection]
    tail_y_points_selected = tail_y_points[tail_point_selection]

    return tail_x_points_selected, tail_y_points_selected