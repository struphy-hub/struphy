from sympde.topology import Cube, Derham
from psydac.api.discretization import discretize

import psydac.core.bsplines as bsp
import psydac.core.bsplines_pyccel as bsp_p

import numpy as np

# Psydac symbolic domain
DOMAIN_symb = Cube('C', bounds1=(0, 1), bounds2=(0, 1), bounds3=(0, 1))

# Psydac symbolic Derham
DERHAM_symb = Derham(DOMAIN_symb)

# grid parameters
Nel      = [8, 9, 8]
p        = [2, 3, 3]
spl_kind = [True, True, True] 
n_quad   = [4, 4, 4]

print(f'Nel: {Nel}, p: {p}, spl_kind: {spl_kind}')

# Psydac discrete De Rham
DOMAIN_PSY  = discretize(DOMAIN_symb, ncells=Nel)
DERHAM_PSY  = discretize(DERHAM_symb, DOMAIN_PSY, degree=p, periodic=spl_kind)

# Psydac spline spaces
print(f'derham spaces: {DERHAM_PSY.spaces}')
V0 = DERHAM_PSY.V0
V1 = DERHAM_PSY.V1
V2 = DERHAM_PSY.V2
V3 = DERHAM_PSY.V3

# Psydac projectors
P0, P1, P2, P3  = DERHAM_PSY.projectors(nquads=n_quad)

# Inter- and histopolation point sets
pts_0 = P0._grid_x[0]
pts_3 = P3._grid_x[0]

# 1d space info


#for n, space, pts in zip(range(3), V0.spaces, pts_0):
for n, (space, pts) in enumerate(zip(V3.spaces, pts_3)):

    print(f'\nDirection {n + 1}, Nel={Nel[n]}, p={p[n]}, periodic={spl_kind[n]}, space attributes:')
    print('---------------------------------------------------------')
    print(f'breaks       : {space.breaks}')
    print(f'degree       : {space.degree}')
    print(f'knots        : {space.knots}')
    print(f'kind         : {space.basis}')
    print(f'greville     : {space.greville}')
    print(f'ext_greville : {space.ext_greville}')
    print(f'histopol_grid: {space.histopolation_grid}')
    print(f'interp ready : {space._interpolation_ready}')
    print(f'histop ready : {space._histopolation_ready}')
    print('pts:')
    print(pts)

    T = space.knots
    pi = space.degree

    #pts = pts_li[0]
    # FIX:
    pts = pts%1.

    print('\nP0 point sets (from _grid_x attribute), spans and basis values:')
    print('Points repaired:')
    print(pts)

    # Knot span indices from "find_span", and basis values 
    span_1  = np.zeros(pts.shape)
    basis_1 = np.zeros((*pts.shape, pi + 1))
    for i in range(pts.shape[0]):
        for iq in range(pts.shape[1]):
            x   = pts[i, iq]
            span = bsp.find_span(T, pi, x)
            span_1[i, iq] = span
            basis_1[i, iq, :] = bsp.basis_funs(T, pi, x, span)

    # Knot span indices as in basis_ders_on_quad_grid, and basis values 
    span_2  = np.zeros(pts.shape)
    basis_2 = np.zeros((*pts.shape, pi + 1))

    temp_spans = np.zeros(len(T), dtype=int)
    span_all_i = bsp.elements_spans(T, pi, temp_spans)

    for i in range(pts.shape[0]):
        span = span_all_i[i]
        for iq in range(pts.shape[1]):
            x = pts[i, iq]
            span_2[i, iq] = span
            basis_2[i, iq, :] = bsp.basis_funs(T, pi, x, span)

    # Basis values from basis_ders_on_quad_grid
    basis_temp = bsp.basis_ders_on_quad_grid(T, pi, pts, 0, normalization=False)
    basis_3 = basis_temp[:, :, 0, :]

    print('span 1 (from find_span):')
    print(span_1)

    print('span 2 (internal span in basis_ders_on_quad_grid):')
    print(span_2)

    print('\nBasis values in first element (for each quad point):')
    for iq in range(pts.shape[1]):
        print(f'\npoint: {pts[0, iq]}')
        print(f'span_1      : {basis_1[0, iq, :].transpose()}')
        print(f'span_2      : {basis_2[0, iq, :].transpose()}')
        print(f'on_quad_grid: {basis_3[0, :, iq]}')

    print('\nBasis values in second element (for each quad point):')
    for iq in range(pts.shape[1]):
        print(f'\npoint: {pts[1, iq]}')
        print(f'span_1    : {basis_1[1, iq, :].transpose()}')
        print(f'span_2    : {basis_2[1, iq, :].transpose()}')
        print(f'basis_ders: {basis_3[1, :, iq]}')

    print('\nBasis values in third element (for each quad point):')
    for iq in range(pts.shape[1]):
        print(f'\npoint: {pts[2, iq]}')
        print(f'span_1    : {basis_1[2, iq, :].transpose()}')
        print(f'span_2    : {basis_2[2, iq, :].transpose()}')
        print(f'basis_ders: {basis_3[2, :, iq]}')






