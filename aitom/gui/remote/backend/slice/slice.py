from math import tan, pi, atan, sin, cos
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
import mrcfile
from tqdm import tqdm
from .PointMap import PointMap
from io import BytesIO
import base64

# avoid UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
matplotlib.use('agg')


def get_rotation_matrix(center, x_rot: float, y_rot: float, z_rot: float, reverse=False):
    """
    Get the rotation matrix with given center point and rotation angle(in radian)
    @center: len >= 3 tuple/list/numpy array
    @reverse: If False, convert world coordinates to user coordinates, and vice versa
    @return: 4 * 4 numpy array
    """
    Rx = np.array(
        [[1, 0, 0, 0],
         [0, cos(x_rot), -sin(x_rot), 0],
         [0, sin(x_rot), cos(x_rot), 0],
         [0, 0, 0, 1]]
    )
    Ry = np.array(
        [[cos(y_rot), 0, -sin(y_rot), 0],
         [0, 1, 0, 0],
         [sin(y_rot), 0, cos(y_rot), 0],
         [0, 0, 0, 1]]
    )
    Rz = np.array(
        [[cos(z_rot), -sin(z_rot), 0, 0],
         [sin(z_rot), cos(z_rot), 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]
    )
    T = np.array(
        [[1, 0, 0, center[0]],
         [0, 1, 0, center[1]],
         [0, 0, 1, center[2]],
         [0, 0, 0, 1]]
    )
    ret = T @ Rx @ Ry @ Rz
    return np.linalg.inv(ret) if reverse else ret


def convert_angle_to_surface(center, x_rot, y_rot, z_rot, default_plane='xoy'):
    """
    Get the plane parameter equation with given center point and rotation angle(in radian)
    equation: Ax + By + Cz + D = 0
    @center: len >= 3 tuple/list/numpy array (integer)
    @default_plane: 'xoy' (0, 0, 1), 'yoz' (1, 0, 0) or 'xoz' (0, 1, 0)
    @return: A, B, C, D
    """
    A, B, C = 0, 0, 0
    if default_plane == 'xoy':
        C = 1
    elif default_plane == 'yoz':
        A = 1
    elif default_plane == 'xoz':
        B = 1
    else:
        raise KeyError
    r = get_rotation_matrix((0, 0, 0), x_rot, y_rot, z_rot)
    A, B, C, _ = r @ np.array([A, B, C, 1]).T
    # (A, B, C) = map(tan, (x_rot, y_rot, z_rot))
    D = A * center[0] + B * center[1] + C * center[2]
    return A, B, C, -D


def slice3d(model, center, x_rot: float, y_rot: float, z_rot: float, default_plane='xoy'):
    """
    Get a slice surface with given 3Dmodel, center and rotation angle(in degree)
    @model: 3D numpy array
    @center: len >= 3 tuple/list/numpy array
    @default_plane: 'xoy', 'yoz' or 'xoz'
    @return: pass
    """
    d2r = lambda x: x * pi / 180
    x_rot, y_rot, z_rot = d2r(x_rot), d2r(y_rot), d2r(z_rot)
    A, B, C, D = convert_angle_to_surface(center, x_rot, y_rot, z_rot, default_plane=default_plane)
    x, y, z = model.shape
    base = (A ** 2 + B ** 2 + C ** 2) ** 0.5
    rot_matrix = get_rotation_matrix(center, x_rot, y_rot, z_rot, reverse=True)
    ret = []
    fig = plt.figure(frameon=False)
    ax = fig.add_subplot(221, projection='3d')
    xs, ys, zs = [], [], []
    mp = PointMap()

    def bn_search():
        # Notice: this implementation method is both low-speed and less-effective
        for k in tqdm(range(z)):
            for j in range(y):
                l, r = 0, x - 1
                disl = (A * l + B * j + C * k + D) / base
                disr = (A * r + B * j + C * k + D) / base
                i = None
                while l <= r:
                    mid = (l + r) // 2
                    dis = (A * mid + B * j + C * k + D) / base
                    if -1e-6 <= dis < 0.9:
                        i = mid
                        break
                    elif dis > 0.9:
                        if disr > disl:
                            r = mid - 1
                        else:
                            l = mid + 1
                    else:
                        if disr < disl:
                            r = mid - 1
                        else:
                            l = mid + 1
                if i is None:
                    continue
                new_pos = (rot_matrix @ np.array([i, j, k, 1]).T).T
                xs.append(i)
                ys.append(j)
                zs.append(k)
                mp.add_point(new_pos[0], new_pos[1], new_pos[2], model[i, j, k])

    def neighborhood():
        use = set()
        st = [tuple(center)]
        sx = [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        sy = [-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1]
        sz = [-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1]
        while st:
            p = st.pop()
            if p in use:
                continue
            i, j, k = p
            if not (0 <= i < x and 0 <= j < y and 0 <= k < z):
                continue
            use.add(p)
            dis = (A * i + B * j + C * k + D) / base
            if -1e-6 <= dis < 0.9:
                new_pos = (rot_matrix @ np.array([i, j, k, 1]).T).T
                xs.append(i)
                ys.append(j)
                zs.append(k)
                mp.add_point(new_pos[0], new_pos[1], new_pos[2], model[i, j, k])
                for s in range(len(sx)):
                    st.append((i + sx[s], j + sy[s], k + sz[s]))

    neighborhood()
    ax.scatter(xs, ys, zs)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    gray_pic = mp.generate_gray_map()
    ax2 = fig.add_subplot(222)
    ax2.imshow(gray_pic, cmap='gray')
    # plt.show()
    buf = BytesIO()

    plt.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.read())
    # have to return base64 encoded string, which is 37% larger in size,
    # because otherwise the image data gets corrupted during AJAX request
    # https://stackoverflow.com/a/42929211/4634893


if __name__ == '__main__':
    mrc = mrcfile.open('tomotarget0.mrc')
    model = mrc.data
    # 73 32 -47
    # x_rot, y_rot, z_rot = 12, -23, 83
    x_rot, y_rot, z_rot = 73, 32, -47
    center = (10, 10, 10)
    slice3d(model, center, x_rot, y_rot, z_rot)
