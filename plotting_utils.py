import numpy as np

def find_axes_limits(data, padding):
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

def vector_to_tail_line(tail_vector, gal_position, x_vals, length=50):
    
    '''
    A function to calculate a line to draw a tail on a galaxy plot, given a galaxy position (a 2-d point, doesn't matter whether its xy, xz, etc.).
    
    '''
    
    line_slope = tail_vector[1]/tail_vector[0]
    
    # if np.abs(line_slope)<0.001:
    #     print('Oops')
    
    y_points = gal_position[1] + line_slope*(x_vals - gal_position[0])
    
    return y_points