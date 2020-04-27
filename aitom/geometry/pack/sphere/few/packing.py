import numpy as np
import matplotlib.pyplot as plt
import math



def overlap_detection(radius_list, x, y, z , show_info = 0):
    overlapOrNot = 0

    # detecte if these ball are overlap
    for ii in range(0, len(radius_list)):
        for jj in range(ii + 1, len(radius_list)):
            tempR1 = radius_list[ii][0]
            tempR2 = radius_list[jj][0]
            tempDis = (x[ii] - x[jj]) ** 2 + (y[ii] - y[jj]) ** 2 + (z[ii] - z[jj]) ** 2
            Distance = math.sqrt(tempDis)
            # print log or not
            if show_info == 0:
                pass
            else:
                print('radius of NO.', ii, ': ', tempR1, '\tradius of NO.', jj, ': ',tempR2, '\tDistance: ', Distance)

            #detection
            if (tempR1 + tempR2) > Distance:
                # print('overlap!radius and distance are: ', tempR1, tempR2, Distance)
                overlapOrNot = 1
                break

    return overlapOrNot

def show_image(x,y,z):
    print("Show distribution img of protein's center")
    ax = plt.subplot(111, projection='3d')  # create project
    ax.scatter(x, y, z, c='r')  # draw

    ax.set_zlabel('Z')  # aix
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()

def show_sum_img(sumlist):
    print('Show sum img')
    ax = plt.subplot()
    ax.scatter(range(len(sumlist)), sumlist, alpha=0.5)
    plt.show()


def initialization(radius_list, box_size = 5000, show_img = 0, show_log = 0):
    print('Start initialization!')
    if show_log == 0:
        pass
    else:
        print('radius:', radius_list)

    # initialization all proteins with different localization without overlap
    while 1:
        location = np.random.rand(3, len(radius_list)) * box_size #set a box
        x, y, z = location[0], location[1], location[2]
        mark = overlap_detection(radius_list, x, y, z, 0) # detection if they are overlap
        if mark == 0:
            break

    #show initial Coordinates
    if show_log != 0:
        print('Initialization Coordinates:')
        print('x: ', x,'\n y:', y, '\n z:', z)
    else:
        pass

    #show initial image
    if show_img == 0:
        pass
    else:
        show_image(x, y, z)

    print('Finish initialization\n')
    return location

def do_packing(radius_list, location, iteration = 10001, step = 1,  show_img = 0, show_log = 0):
        # packing process
        print('Start packing!')

        # initialization
        x, y, z = location[0], location[1], location[2]

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

                # for i in range(6):
                #     if ii == i * (math.floor(iteration / 5)) and jj == 0:
                #         print('index',ii, 'tempsum:', tempsum, 'GradX:', GradX)
                if show_log != 0:
                    if (ii == iteration-1 or ii == math.ceil(iteration*4/5) or ii == math.ceil(iteration*3/5) or ii == math.ceil(iteration*2/5) or ii == math.ceil(iteration*1/5) ) and jj == 0:
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

            if show_img == 0:
                pass
            else:
                if ii == (iteration-1):
                    show_image(x, y, z)
                    show_sum_img(sum_list)

        if show_log != 0:
            print('The coordinate of all the proteins center:')
            print(x)
            print(y)
            print(z)

        print('The radius and distance:')
        # print center and distance between all these balls
        overlap_detection(radius_list, x, y, z, 1)

        dict = {}
        dict['sum'] = tempsum
        dict['grad'] = GradX
        dict['x'] = x
        dict['y'] = y
        dict['z'] = z
        print('Finish packing!\n')
        return dict

def get_box_size(radius_list, show_log = 0):
    print('Start to set box size')
    tmp1 = math.ceil(math.sqrt(len(radius_list))) # number of protein put in one column
    tmp2 = max(radius_list[0]) # the max radius
    tmp_boxsize = int(tmp1 * tmp2 *2 ) # get a tmp boxsize

    # format it to a *0000 number
    if int(str(tmp_boxsize)[0]) < 5:
        first = 1
    else:
        first = 5
    box_size =  (10 ** (len(str(tmp_boxsize))) ) * first

    if show_log != 0:
        print('number in column,', tmp1)
        print('max radius', tmp2)
        print('tmp box size', tmp_boxsize)
        print('box_size:', box_size)
    print('Finish set box size, box size is', box_size, '\n')
    return box_size





