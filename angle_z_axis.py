import numpy as np


def get_z_axis_angle_derv(p0, p1, const=1.):
    """
    Calculates the derivatives of an angle with respect to z axis.

    Inputs:
    -------
    p0 : list
        Positions of atom 0

    p1 : list
        Positions of atom 1

    Returns:
    --------

    M = [
        [dx_a0, dy_a0, dz_a0],
        [dx_a1, dy_a1, dz_a1]
    ]
    """

    x0 = p0[0]
    y0 = p0[1]
    z0 = p0[2]

    x1 = p1[0]
    y1 = p1[1]
    z1 = p1[2]

    dx_a0 = const*(-(-x0 + 1.0*x1)*(-z0 + z1)*((-x0 + x1)**2  + (-y0 + y1)**2  + (-z0 + z1)**2)**(-1.5)) / np.sqrt(- (-z0 + z1)**2 /((-x0 + x1)**2  + (-y0 + y1)**2  + (-z0 + z1)**2 ) + 1 )
    dy_a0 = const*(-(-y0 + 1.0*y1)*(-z0 + z1)*((-x0 + x1)**2  + (-y0 + y1)**2  + (-z0 + z1)**2)**(-1.5)) / np.sqrt(- (-z0 + z1)**2 /((-x0 + x1)**2  + (-y0 + y1)**2  + (-z0 + z1)**2 ) + 1 )
    dz_a0 = -1*const*((-z0 + z1)*(-z0 + z1)*((-x0 + x1)**2  + (-y0 + y1)**2  + (-z0 + z1)**2)**(-1.5) -  ((-x0 + x1)**2  + (-y0 + y1)**2  + (-z0 + z1)**2)**(-0.5)) / np.sqrt(- (-z0 + z1)**2 /((-x0 + x1)**2  + (-y0 + y1)**2  + (-z0 + z1)**2 ) + 1 )

    dx_a1 = const*(-(1.0*x0 - x1)*(-z0 + z1)*((-x0 + x1)**2  + (-y0 + y1)**2  + (-z0 + z1)**2)**(-1.5)) / np.sqrt(- (-z0 + z1)**2 /((-x0 + x1)**2  + (-y0 + y1)**2  + (-z0 + z1)**2 ) + 1 )
    dy_a1 = const*(-(1.0*y0 - y1)*(-z0 + z1)*((-x0 + x1)**2  + (-y0 + y1)**2  + (-z0 + z1)**2)**(-1.5)) / np.sqrt(- (-z0 + z1)**2 /((-x0 + x1)**2  + (-y0 + y1)**2  + (-z0 + z1)**2 ) + 1 )
    dz_a1 = -1*const*((-z0 + z1)*(z0 - z1)*((-x0 + x1)**2  + (-y0 + y1)**2  + (-z0 + z1)**2)**(-1.5) +  ((-x0 + x1)**2  + (-y0 + y1)**2  + (-z0 + z1)**2)**(-0.5)) / np.sqrt(- (-z0 + z1)**2 /((-x0 + x1)**2  + (-y0 + y1)**2  + (-z0 + z1)**2 ) + 1 )

    return np.array([[dx_a0, dy_a0, dz_a0], [dx_a1, dy_a1, dz_a1]])

