"""
functions to partition a large volume so that it can be processed part by part inside memory
"""

import numpy as N


def gen_bases(size, nonoverlap_width, overlap_width):
    """
    for a volume of given size, define the partitions, and give starting and ending point of each partition.
    The partition's size is defined by nonoverlap with plus overlap width
    """
    se0 = []
    i0 = 0
    while True:
        j0 = min(i0 + nonoverlap_width + overlap_width, size[0])

        se1 = []
        i1 = 0
        while True:
            j1 = min(i1 + nonoverlap_width + overlap_width, size[1])
            se2 = []
            i2 = 0
            while True:
                j2 = min(i2 + nonoverlap_width + overlap_width, size[2])
                se2.append([[i0, j0], [i1, j1], [i2, j2]])
                if j2 >= size[2]:
                    break
                i2 += nonoverlap_width
            se1.append(se2)
            if j1 >= size[1]:
                break
            i1 += nonoverlap_width
        se0.append(se1)
        if j0 >= size[0]:
            break
        i0 += nonoverlap_width

    return N.array(se0)
