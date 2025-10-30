from numpy import shape, sqrt


def weighted_arc_lengths_flux_surface(r: "float[:]", z: "float[:]", grad_psi: "float[:]", dwls: "float[:]", kind: int):
    """
    Computes the weighted arc lengths

    dwl_(j+1/2) = dl_(j+1/2) * 1/2 * [ (R/(h*|grad(psi)|))_j + (R/(h*|grad(psi)|))_(j+1) ]

    between two angles (theta_j, theta_(j+1)) on a flux surface. The function h determines the meaning of the angular coordinate:

        1. h = R/|grad(psi)| | equal arc length
        2. h = R^2           | straight field lines
        3. h = R             | constant area
        4. h = 1             | constant volume

    Implementation according to p. 131-133 in [1].

    Parameters
    ----------
    r : xp.ndarray
        R coordinates of the flux surface.

    z : xp.ndarray
        Z coordinates of the flux surface.

    grad_psi : xp.ndarray
        Absolute values of the flux function gradient on the flux surface: |grad(psi)| = sqrt[ (d_R psi)**2 + (d_Z psi)**2 ].

    dwls : xp.ndarray
        The weighted arc lengths will be written into this array. Length must be one smaller than lengths of r, z and grad_psi.

    kind : int
        Which weight to use (see above table: 1: equal arc length, 2: straight field line, etc.)

    References
    ----------
    [1] Jardin Stephen, Computational Methods in Plasma Physics, Taylor and Francis Group 2010.
    """

    # number of angle boundaries
    n_th = r.size

    for j in range(n_th - 1):
        # local orthonormal coordinate system at line segment (j --> j+1)
        er_1 = r[j + 1] - r[j]
        er_2 = z[j + 1] - z[j]

        # normalization
        ds = sqrt(er_1**2 + er_2**2)
        er_1 /= ds
        er_2 /= ds

        # counter-clockwise rotation by 90° to ensure er.dot(ez) = 0
        ez_1 = -1 * er_2
        ez_2 = 1 * er_1

        # vectors (j-1 --> j+1) and (j --> j+2) in global coordinate system
        vec1_r = r[j + 1] - r[(j - 1) % n_th]
        vec1_z = z[j + 1] - z[(j - 1) % n_th]

        vec2_r = r[(j + 2) % n_th] - r[(j - 0) % n_th]
        vec2_z = z[(j + 2) % n_th] - z[(j - 0) % n_th]

        # base transformation to convert to local coord system A = [[er_1, ez_1], [er_2, ez_2]]
        detA = er_1 * ez_2 - ez_1 * er_2

        # vectors in local coordinate system
        vec1_1 = (ez_2 * vec1_r - ez_1 * vec1_z) / detA
        vec1_2 = (er_1 * vec1_z - er_2 * vec1_r) / detA

        vec2_1 = (ez_2 * vec2_r - ez_1 * vec2_z) / detA
        vec2_2 = (er_1 * vec2_z - er_2 * vec2_r) / detA

        # slopes z/r in local coordinates
        m1 = vec1_2 / vec1_1
        m2 = vec2_2 / vec2_1

        # arc length dl_(j+1/2)
        dls = ds * (1 + (2 * m1**2 + 2 * m2**2 - m1 * m2) / 30)

        # weighted arc length dl_(j+1/2) * 1/2 * [(R/(h*|grad(psi)|))_j + (R/(h*|grad(psi)|))_(j+1)]

        # h = R/|grad(psi)| (equal arc)
        if kind == 1:
            dwls[j] = dls

        # h = R^2 (straight field line)
        elif kind == 2:
            dwls[j] = dls * 1 / 2 * (1 / (r[j] * grad_psi[j]) + 1 / (r[j + 1] * grad_psi[j + 1]))

        # h = R (constant area)
        elif kind == 3:
            dwls[j] = dls * 1 / 2 * (1 / grad_psi[j] + 1 / grad_psi[j + 1])

        # h = 1 (constant volume)
        elif kind == 4:
            dwls[j] = dls * 1 / 2 * (r[j] / grad_psi[j] + r[j + 1] / grad_psi[j + 1])
