

def rotation_matrix_zyz(ang):
    phi = ang[0];       theta = ang[1];     psi_t = ang[2];
    
    a1 = rotation_matrix_axis(2, psi_t)       # first rotate about z axis for angle psi_t
    a2 = rotation_matrix_axis(1, theta)
    a3 = rotation_matrix_axis(2, phi)
    
    rm = a3.dot(a2).dot(a1)      # for matrix left multiplication
    
    rm = rm.transpose()       # note: transform because tformarray use right matrix multiplication

    return rm



def rotation_matrix_axis(dim, theta):
    # following are left handed system (clockwise rotation)
    # IMPORTANT: different to MATLAB version, this dim starts from 0, instead of 1
    if dim == 0:        # x-axis
        rm = N.array(  [[1.0, 0.0, 0.0], [0.0, math.cos(theta), -math.sin(theta)], [0.0, math.sin(theta), math.cos(theta)]]  )
    elif dim == 1:    # y-axis
        rm = N.array(  [[math.cos(theta), 0.0, math.sin(theta)], [0.0, 1.0, 0.0], [-math.sin(theta), 0.0, math.cos(theta)]]  )
    elif dim == 2:        # z-axis
        rm = N.array(  [[math.cos(theta), -math.sin(theta), 0.0], [math.sin(theta), math.cos(theta), 0.0], [0.0, 0.0, 1.0]]  )
    else:
        raise    
    
    return rm


