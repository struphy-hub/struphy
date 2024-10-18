geometries = {
    "Cuboid": {
        "type": "Cuboid",
        "Cuboid": {"l1": 0.0, "r1": 20.0, "l2": 0.0, "r2": 20.0, "l3": 0.0, "r3": 1.0},
    },
    "Colella": {
        "type": "Colella",
        "Colella": {
            "Lx": 20.0,  # length in x-direction
            "Ly": 20.0,  # length in y-direction
            "alpha": 0.1,  # distortion factor
            "Lz": 1.0,  # length in third direction
        },
    },
    "HollowTorus": {
        "type": "HollowTorus",
        "HollowTorus": {
            "a1": 0.1,  # inner radius
            "a2": 1.0,  # minor radius
            "R0": 10.0,  # major radius
            "sfl": True,  # straight field line coordinates?
            "tor_period": 1,  # toroidal periodicity built into the mapping: phi = 2*pi * eta3 / tor_period
        },
    },
    "HollowCylinder": {
        "type": "HollowCylinder",
        "HollowCylinder": {
            "a1": 0.2,  # inner radius
            "a2": 1.0,  # outer radius
            "Lz": 4.0,  # length of cylinder
        },
    },
    "Tokamak": {
        "type": "Tokamak",
        "Tokamak": {
            "Nel": [8, 32],  # number of poloidal grid cells for spline mapping, >p
            "p": [3, 3],  # poloidal spline degrees for spline mapping, >1
            "psi_power": 0.7,  # parametrization of radial flux coordinate eta1=psi_norm^psi_power, where psi_norm is normalized flux
            "psi_shifts": [
                2.0,
                2.0,
            ],  # start and end shifts of polidal flux in % --> cuts away regions at the axis and edge
            "xi_param": "equal_angle",  # parametrization of angular coordinate (equal_angle, equal_arc_length or sfl (straight field line))
            "r0": 0.3,  # initial guess for radial distance from axis used in Newton root-finding method for flux surfaces
            "Nel_pre": [
                64,
                256,
            ],  # number of poloidal grid cells of pre-mapping needed for equal_arc_length and sfl
            "p_pre": [
                3,
                3,
            ],  # poloidal spline degrees of pre-mapping needed for equal_arc_length and sfl
            "tor_period": 1,  # toroidal periodicity built into the mapping: phi = 2*pi * eta3 / tor_period
        },
    },
    "GVECunit": {"type": "GVECunit"},
    "DESCunit": {"type": "DESCunit"},
}
