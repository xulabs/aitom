"""
the main function to the packing process of a target macromolecule and a few neighbors
"""

import numpy as np
import math


def get_box_size(radius_list, show_log=0):
    """
    :param
        radius_list: the radius list of all the protein in this simulation field
        show_log: print log or not

    :return:
        the size of a box, a int number
    """
    if show_log != 0:
        print('Start to set box size')
    # number of protein put in one column
    tmp1 = math.ceil(math.sqrt(len(radius_list)))
    # the max radius
    tmp2 = max(radius_list[0])
    # get a tmp boxsize
    tmp_boxsize = int(tmp1 * tmp2 * 2)

    # format it to a *0000 number
    if int(str(tmp_boxsize)[0]) < 5:
        first = 3
    else:
        first = 15
    box_size = (10 ** (len(str(tmp_boxsize)))) * first

    if show_log != 0:
        print('number in column,', tmp1)
        print('max radius', tmp2)
        print('tmp box size', tmp_boxsize)
        print('box_size:', box_size)
        print('Finish set box size, box size is', box_size, '\n')
    return box_size


def overlap_detection(radius_list, x, y, z, show_info=0):
    """
    :param
        radius_list: the radius list of all the protein in this simulation field
        x: the coordinate (x) list of all the protein in this simulation field
        y: the coordinate (y) list of all the protein in this simulation field
        z: the coordinate (z) list of all the protein in this simulation field
        show_info: print log or not

    :return:
        Boolean value
    """
    overlapOrNot = 0

    # detecte if these ball are overlap
    for ii in range(0, len(radius_list)):
        for jj in range(ii + 1, len(radius_list)):
            tempR1 = radius_list[ii][0]
            tempR2 = radius_list[jj][0]
            tempDis = (x[ii] - x[jj]) ** 2 + (y[ii] - y[jj]) ** 2 + (z[ii] - z[jj]) ** 2
            Distance = math.sqrt(tempDis)
            # print log or not
            if show_info != 0:
                print('radius of NO.', ii, ': ', tempR1, '\tradius of NO.', jj, ': ', tempR2, '\tDistance: ', Distance)

            # detection
            if (tempR1 + tempR2) > Distance:
                # print('overlap!radius and distance are: ', tempR1, tempR2, Distance)
                overlapOrNot = 1
                break

    return overlapOrNot


def initialization(radius_list, box_size=5000, show_log=0):
    """
    :param
        radius_list: the radius list of all the protein in this simulation field
        box_size: an int number of the simulation field size
        show_log: print log or not

    :return:
        the inital location of all the macromolecules in the simulation field
        location = {
            'x' : [P1, P2, ... ,Pn]
            'y' : [P1, P2, ... ,Pn]
            'z' : [P1, P2, ... ,Pn]}
    """
    if show_log != 0:
        print('Start initialization!')
    if show_log != 0:
        print('radius:', radius_list)

    # initialization all proteins with different localization without overlap
    while 1:
        location = np.random.rand(3, len(radius_list)) * box_size  # set a box
        x, y, z = location[0], location[1], location[2]
        mark = overlap_detection(radius_list, x, y, z, 0)  # detection if they are overlap
        if mark == 0:
            break

    # show initial Coordinates
    if show_log != 0:
        print('Initialization Coordinates:')
        print('x: ', x, '\n y:', y, '\n z:', z)
        print('Finish initialization\n')
    return location


def do_packing(radius_list, location, iteration=5001, step=1, show_log=0):
    """
    :param
        radius_list: the radius list of all the protein in this simulation field
        location: the initialization location of all the macromolecules in the simulation field
        iteration: the number of iteration time
        step: the moving step(parameter to chotrol how long a macromolecules will move) in each iteration
        show_log: print log or not

    :return:
        the packing result, including sum, grad, x, y, z, sum_list
        dict = {
            'sum': a number, final value of the loss
            'grad': a number, final valus of the grad of the loss
            'x': list, final location of all macromolecules in the simulation field
            'y': list, final location of all macromolecules in the simulation field
            'z': list, final location of all macromolecules in the simulation field
            'sum_list': list, the sumlist of all the macromolecules, is used to draw the img of loss in each iteration}
    """
    # packing process
    if show_log != 0:
        print('Start packing!')

    # initialization
    x = []
    y = []
    z = []
    x, y, z = location[0], location[1], location[2]

    if show_log != 0:
        print('processing...')
    learningRate = step
    sum_list = []
    for ii in range(1, iteration):
        for jj in range(0, len(radius_list)):
            sumx = sum(x)
            sumy = sum(y)
            sumz = sum(z)
            GradX = len(radius_list) * x[jj] - sumx
            GradY = len(radius_list) * y[jj] - sumy
            GradZ = len(radius_list) * z[jj] - sumz
            tempsum = GradX * GradX + GradY * GradY + GradZ * GradZ
            tempsum = math.sqrt(tempsum)

            if show_log != 0:
                if (ii == iteration - 1 or ii == math.ceil(iteration * 4 / 5) or ii == math.ceil(
                        iteration * 3 / 5) or ii == math.ceil(iteration * 2 / 5) or ii == math.ceil(iteration * 1 / 5)) and jj == 0:
                    print('setp', ii, 'tempsum:', tempsum, 'GradX:', GradX)

            GradX2 = GradX / tempsum
            GradY2 = GradY / tempsum
            GradZ2 = GradZ / tempsum
            x[jj] = x[jj] - GradX2 * learningRate
            y[jj] = y[jj] - GradY2 * learningRate
            z[jj] = z[jj] - GradZ2 * learningRate

            # overlap detection
            mark = overlap_detection(radius_list, x, y, z, 0)
            if mark == 1:
                x[jj] = x[jj] + GradX2 * learningRate
                y[jj] = y[jj] + GradY2 * learningRate
                z[jj] = z[jj] + GradZ2 * learningRate

            sum_list.append(tempsum)

    if show_log != 0:
        print('The radius and distance:')
        # print center and distance between all these balls
        overlap_detection(radius_list, x, y, z, 1)

    dict = {'sum': tempsum,
            'grad': GradX,
            'x': x,
            'y': y,
            'z': z,
            'sum_list': sum_list}
    if show_log != 0:
        print('Finish packing!\n')
    return dict
