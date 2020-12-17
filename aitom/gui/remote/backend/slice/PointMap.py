import numpy as np
from math import floor, ceil


class PointMap:
    def __init__(self):
        self.xs = []
        self.ys = []
        self.zs = []
        self.grays = []

    def add_point(self, x, y, z, gray):
        self.xs.append(x)
        self.ys.append(y)
        self.zs.append(z)
        self.grays.append(gray)

    def generate_gray_map(self):
        self.xs = np.array(self.xs)
        self.ys = np.array(self.ys)
        self.zs = np.array(self.zs)
        self.grays = np.array(self.grays)
        self.xs -= np.min(self.xs)
        self.ys -= np.min(self.ys)

        N, M = floor(np.max(self.xs)) + 2, floor(np.max(self.ys)) + 2
        gray_map = [[], ] * N
        for i in range(N):
            gray_map[i] = [None, ] * M

        _max = float('-inf')
        _min = float('+inf')
        for x, y, z, gray in zip(self.xs, self.ys, self.zs, self.grays):
            x0, y0, x1, y1 = floor(x), floor(y), ceil(x), ceil(y)
            iter = [(x0, y0), (x0, y1), (x1, y0), (x1, y1)]
            for int_x, int_y in iter:
                dis = abs(x - int_x) + abs(y - int_y) + abs(z)
                if gray_map[int_x][int_y] is None or dis < gray_map[int_x][int_y][1]:
                    gray_map[int_x][int_y] = [gray, dis]
                    _max = max(_max, gray)
                    _min = min(_min, gray)

        # for x in range(N):
        #    print(x)
        #    for y in range(M):
        #        gray_map[x][y] = interpolate(gray_map[x][y])
        bg = _max * 0.9 + _min * 0.1
        ret = np.full((N, M), bg)
        for x in range(N):
            for y in range(M):
                if gray_map[x][y] is not None:
                    ret[x, y] = gray_map[x][y][0]
        return ret
