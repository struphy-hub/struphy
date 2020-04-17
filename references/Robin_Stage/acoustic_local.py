# needed imports
from numpy import zeros, ones, linspace, zeros_like, asarray
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# nedded imports
from bsplines    import elements_spans  # computes the span for each element
from bsplines    import make_knots      # create a knot sequence from a grid
from bsplines    import quadrature_grid # create a quadrature rule over the whole 1d grid
from bsplines    import basis_ders_on_quad_grid # evaluates all bsplines and their derivatives on the quad grid
from bsplines    import basis_from_1D_basis #creates a 2D spline basis on 2D quad grid form 1D spline basis
from utilities   import point_on_bspline_curve_2D

from quadratures import gauss_legendre

import scipy
import scipy.sparse as sparse
from scipy.sparse.linalg import gmres

import His_lin_2D as pcurl
import His_lin_2D_acoustic as pcurl_bspline
import QI_quad_2D as p1


from numpy import transpose
from numpy import amax
from numpy import amin
from numpy import newaxis




def I(i, j, n):
	return i+j*n

def invI(k, n):
	return[k%n, int(k/n)]

#@types('function', 'int', 'int', 'int', 'int', 'numpy.ndarray', 'numpy.ndarray', 'numpy.ndarray', 'numpy.ndarray', 'numpy.ndarray', 'numpy.ndarray', 'numpy.ndarray', 'numpy.ndarray')
def assemble_F(f, nex, ney, p1, p2, spans_2x, spans_2y, basis_1, weightsx, weightsy, pointsx, pointsy, F1):    
	k1x = weightsx.shape[1]
	k1y = weightsy.shape[1]
	# ...

	# ... build rhs
	for ie1 in range(0, nex):
		i_span_1 = spans_2x[ie1]
		for je1 in range(0, ney):
			j_span_1 = spans_2y[je1]
			for il_1 in range(0, p1+1):
				for jl_1 in range(0, p2+1):
					i1 = i_span_1 - p1 + il_1
					j1 = j_span_1 - p2 + jl_1


					v = 0.0
					for g1 in range(0, k1x):
						for g2 in range(0, k1y):
							bi_0 = basis_1[I(ie1, je1, nex), I(il_1, jl_1, p1+1), 0, I(g1, g2, k1x)]
							bi_x = basis_1[I(ie1, je1, nex), I(il_1, jl_1, p1+1), 1, I(g1, g2, k1x)]  
							bi_y = basis_1[I(ie1, je1, nex), I(il_1, jl_1, p1+1), 2, I(g1, g2, k1x)]
							x1    = pointsx[ie1, g1]
							y1    = pointsy[je1, g2]
							wvol  = weightsx[ie1, g1]*weightsy[je1, g2]

							v += bi_0 * f(x1, y1) * wvol

					F1[I(i1, j1, nex+p1)] += v
	# ...

	# ...
	return F1
	# ...
# ...



def assemble_M(nex, ney, px, py, spans_x, spans_y, basis, weightsx, weightsy, matrix):

	k1x = weightsx.shape[1]
	k1y = weightsy.shape[1]
	# ...

	# ... build matrices
	for ie1 in range(0, nex):
		for je1 in range(0, ney):
			i_span_1 = spans_x[ie1]
			j_span_1 = spans_y[je1]
			for il_1 in range(0, px+1):
				for jl_1 in range(0, py+1):
					for il_2 in range(0, px+1):
						for jl_2 in range(0, py+1):
							i1 = i_span_1 - px + il_1
							j1 = j_span_1 - py + jl_1
							i2 = i_span_1 - px + il_2
							j2 = j_span_1 - py + jl_2

							v = 0.0
							for g1 in range(0, k1x):
								for g2 in range(0, k1y):
									bi_0 = basis[I(ie1, je1, nex), I(il_1, jl_1, px+1), 0, I(g1, g2, k1x)]
									bi_x = basis[I(ie1, je1, nex), I(il_1, jl_1, px+1), 1, I(g1, g2, k1x)]
									bi_y = basis[I(ie1, je1, nex), I(il_1, jl_1, px+1), 2, I(g1, g2, k1x)]                    

									bj_0 = basis[I(ie1, je1, nex), I(il_2, jl_2, px+1), 0, I(g1, g2, k1x)]
									bj_x = basis[I(ie1, je1, nex), I(il_2, jl_2, px+1), 1, I(g1, g2, k1x)]
									bj_y = basis[I(ie1, je1, nex), I(il_2, jl_2, px+1), 2, I(g1, g2, k1x)]                 
									wvol = weightsx[ie1, g1]*weightsy[je1, g2]

									v += (bi_0*bj_0) * wvol
							matrix[I(i1, j1, nex+px), I(i2, j2, nex+px)]  += v
	# ...

	return matrix

def assemble_K(nex, ney, px, py, spans_x1, spans_y1, spans_x2, spans_y2, nbasis_x, nbasis_y, basis, weightsx, wieghtsy, pointsx, pointsy, knots_x, knots_y, alpha, K, direction):
			  #nex, ney, px, py, spans_1x, spans_1y, spans_2x, spans_1y, nbasis_1x, nbasis_1y, basis_Hcurl_1, weightsx, weightsy, pointsx, pointsy, knots_2x, knots_1y, alpha, K1, direction='x'
	k1x = weightsx.shape[1]
	k1y = weightsy.shape[1]
	# ...
	if direction=='x':
		px2=px-1
		py2=py
	elif direction=='y':
		px2=px
		py2=py-1
	# ... build matrices
	for ie1 in range(0, nex):
		for je1 in range(0, ney):
			i_span_1 = spans_x2[ie1]
			j_span_1 = spans_y2[je1]
			i_span_2 = spans_x1[ie1]
			j_span_2 = spans_y1[je1]
			for il_1 in range(0, px2+1):
				for jl_1 in range(0, py2+1):
					for il_2 in range(0, px+1):
						for jl_2 in range(0, py+1):
							i1 = i_span_1 - px2 + il_1
							j1 = j_span_1 - py2 + jl_1
							i2 = i_span_2 - px + il_2
							j2 = j_span_2 - py + jl_2
							m1, m2 = pcurl_bspline.lambda_ij_der(knots_x, knots_y, i2, j2, alpha, direction)
							v=0.0
							for g1 in range(0, k1x):
								for g2 in range(0, k1y):
									x1    = pointsx[ie1, g1]
									y1    = pointsy[je1, g2]

									bi_0 = basis[I(ie1, je1, nex), I(il_1, jl_1, px2+1), 0, I(g1, g2, k1x)]

									val2=pcurl_bspline.evaluate_projection_der(knots_x, knots_y, i2, j2, m1, m2, x1, y1, direction)

									wvol = weightsx[ie1, g1]*weightsy[je1, g2]
									v += bi_0 * val2 * wvol
							K[I(i1, j1, nex+px2), I(i2, j2, nex+px)]  += v

	# ...
	
	return K


def dirichlet_F(rhs, nbasis_1x, nbasis_1y):
	n=nbasis_1x*nbasis_1y-2*nbasis_1x-2*nbasis_1y+4
	temp=zeros((nbasis_1x, nbasis_1y))
	rhs_d=zeros((n, 1))
	for i in range(0, nbasis_1x):
		for j in range(0, nbasis_1y):
			temp[i, j]=rhs[I(i, j, nbasis_1x)]
	temp=temp[1 : -1, 1 : -1]
	for i in range(0, n):
		rhs_d[i]=temp[invI(i, nbasis_1x-2)[0], invI(i, nbasis_1x-2)[1]]
	return rhs_d

def dirichlet_M(M, nbasis_1x, nbasis_1y):
	n=nbasis_1x*nbasis_1y-2*nbasis_1x-2*nbasis_1y+4
	M_d=zeros((n, n))
	temp = zeros((nbasis_1x, nbasis_1y, nbasis_1x, nbasis_1y))
	for i in range(0, nbasis_1x):
		for j in range(0, nbasis_1y):
			for k in range(0, nbasis_1x):
				for l in range(0, nbasis_1y):
					temp[i, j, k, l]=M[I(i, j, nbasis_1x), I(k, l, nbasis_1x)]
	temp=temp[1:-1, 1:-1, 1:-1, 1:-1]
	for i in range(0, n):
		for j in range(0, n):
			M_d[i, j]=temp[invI(i, nbasis_1x-2)[0], invI(i, nbasis_1x-2)[1], invI(j, nbasis_1x-2)[0], invI(j, nbasis_1x-2)[1]]
	return M_d

def dirichlet_K(A, nbasis_1x, nbasis_1y, nbasis1, nbasis2):
	n=nbasis_1x*nbasis_1y-2*nbasis_1x-2*nbasis_1y+4
	A_d=zeros((nbasis1*nbasis2, n))
	temp = zeros((nbasis1, nbasis2, nbasis_1x, nbasis_1y))
	for i in range(0, nbasis1):
		for j in range(0, nbasis2):
			for k in range(0, nbasis_1x):
				for l in range(0, nbasis_1y):
					temp[i, j, k, l]=A[I(i, j, nbasis1), I(k, l, nbasis_1x)]
	temp=temp[:, :, 1:-1, 1:-1]
	for i in range(0, nbasis1*nbasis2):
		for j in range(0, n):
			A_d[i, j]=temp[invI(i, nbasis1)[0], invI(i, nbasis1)[1], invI(j, nbasis_1x-2)[0], invI(j, nbasis_1x-2)[1]]
	return A_d

def compute_energy(M1, Mcurl1, Mcurl2, p, ux, uy):
	pt = np.transpose(p)
	uxt = np.transpose(ux)
	uyt = np.transpose(uy)
	val1=M1.dot(p)
	val1=pt.dot(val1)
	val2=Mcurl1.dot(ux)
	val2=uxt.dot(val2)
	val3=Mcurl2.dot(uy)
	val3=uyt.dot(val3)
	return val1 + val2 + val3

def array_to_matrix_dirichlet(nbasis_x, nbasis_y, p):
	P = zeros((nbasis_x, nbasis_y))
	for i in range(1, nbasis_x-1):
		for j in range(1, nbasis_y-1):
			P[i, j]=p[I(i-1, j-1, nbasis_x-2)]
	return P

def array_to_matrix(nbasis_x, nbasis_y, p):
	P = zeros((nbasis_x, nbasis_y))
	for i in range(0, nbasis_x):
		for j in range(0, nbasis_y):
			P[i, j]=p[I(i, j, nbasis_x)]
	return P

def matrix_to_array(nbasis_x, nbasis_y, P):
	p = zeros((nbasis_x*nbasis_y, 1))
	for i in range(0, nbasis_x):
		for j in range(0, nbasis_y):
			p[I(i, j, nbasis_x)]=P[i, j]
	return p

def plot_field_2d_dirichlet(knots1, knots2, p1, p2, nbasis1, nbasis2, u, nx=101, ny=101):
	xmin = knots1[p1]
	xmax = knots1[-p1-1]
	ymin = knots2[p2]
	ymax = knots2[-p2-1]
	xs = np.linspace(xmin, xmax, nx)
	ys = np.linspace(ymin, ymax, ny)
	U=zeros((nbasis1, nbasis2))
	for i in range(1, nbasis1-1):
		for j in range(1, nbasis2-1):
				U[i, j]=u[I(i-1, j-1, nbasis1-2)]
	Q = np.zeros((nx, ny))
	for i in range(0, nx):
		for j in range(0, ny):
			Q[i,j] = point_on_bspline_curve_2D(knots1, knots2, U, p1, p2, xs[i], ys[j])
	im = plt.imshow(Q,origin='lower',interpolation='bilinear')
	return im


def plot_field_2d(knots1, knots2, p1, p2, nbasis1, nbasis2, u, nx=101, ny=101):
	xmin = knots1[p1]
	xmax = knots1[-p1-1]
	ymin = knots2[p2]
	ymax = knots2[-p2-1]
	xs = np.linspace(xmin, xmax, nx)
	ys = np.linspace(ymin, ymax, ny)
	U=zeros((nbasis1, nbasis2))
	for i in range(0, nbasis1):
		for j in range(0, nbasis2):
				U[i, j]=u[I(i, j, nbasis1)]
	Q = np.zeros((nx, ny))
	for i in range(0, nx):
		for j in range(0, ny):
			Q[i,j] = point_on_bspline_curve_2D(knots1, knots2, U, p1, p2, xs[i], ys[j])
	
	im = plt.imshow(Q,origin='lower',interpolation='bilinear')
	return im


def solve(dt, N, M1, Mcurl1, Mcurl2, K1, K2, p_0, ux_0, uy_0, E, P, Ux, Uy):
	p=p_0
	ux=ux_0
	uy=uy_0
	for i in range(1, N+1):
		rhs=M1.dot(p)-dt*(transpose(K1).dot(ux)+transpose(K2).dot(uy))
		
		p_np1, info = gmres(M1, rhs, tol=1e-6, maxiter=5000)
		p_np1 = p_np1[np.newaxis]
		p_np1=transpose(p_np1)

		p_mat=array_to_matrix_dirichlet(nbasis_1x, nbasis_1y, p_np1)

		pcurl_bspline.Dx_L(knots_1x, knots_1y, p_mat, dxp)
		pcurl_bspline.Dy_L(knots_1x, knots_1y, p_mat, dyp)

		ux_mat=array_to_matrix(nbasis_2x, nbasis_1y, ux)
		uy_mat=array_to_matrix(nbasis_1x, nbasis_2y, uy)


		ux_np1 = ux_mat + dt*pcurl_bspline.coeff_spline(knots_2x, knots_1y, dxp, alpha, direction='x')
		uy_np1 = uy_mat + dt*pcurl_bspline.coeff_spline(knots_1x, knots_2y, dyp, alpha, direction='y')


		ux_np1 = matrix_to_array(nbasis_2x, nbasis_1y, ux_np1)
		uy_np1 = matrix_to_array(nbasis_1x, nbasis_2y, uy_np1)
		
		E[i]=compute_energy(M1, Mcurl1, Mcurl2, p_np1, ux_np1, uy_np1)
		P[:, int(i)][newaxis]=transpose(p_np1)
		Ux[:, int(i)][newaxis]=transpose(ux_np1)
		Uy[:, int(i)][newaxis]=transpose(uy_np1)
		p=p_np1
		ux=ux_np1
		uy=uy_np1
		print(float(i)/N*100)


dt = .2 #time step
T=500 #final time
N=int(T/dt) #Number of time steps

alpha = lambda x,y : 0.1*(1 + ((x-.5)**2 + (y-.5)**2))

#initial conditions
ux_0_re = lambda x,y : -(x-.5)/(1+30*((x-.5)**2+(y-.5)**2))**2
uy_0_re = lambda x,y : -(y-.5)/(1+30*((x-.5)**2+(y-.5)**2))**2
p_0_re = lambda x,y : 0

px  = 2    # spline degree
py  = 2
nex = 10  # number of elements
ney = 10

gridx  = linspace(0., 1., nex+1)
gridy  = linspace(0., 1., ney+1)
knots_1x = make_knots(gridx, px, periodic=False)
knots_2x = make_knots(gridx, px-1, periodic=False)
knots_1y = make_knots(gridy, py, periodic=False)
knots_2y = make_knots(gridy, py-1, periodic=False)
spans_1x = elements_spans(knots_1x, px)
spans_2x = elements_spans(knots_2x, px-1)
spans_1y = elements_spans(knots_1y, py)
spans_2y = elements_spans(knots_2y, py-1)

nbasis_1x     = (len(knots_1x) - px - 1)
nbasis_2x     = (len(knots_2x) - px)
nbasis_1y     = (len(knots_1y) - py - 1)
nbasis_2y     = (len(knots_2y) - py)

nderiv = 1

# create the gauss-legendre rule, on [-1, 1]
ux, wx = gauss_legendre( px )
uy, wy = gauss_legendre( py )

# for each element on the grid, we create a local quadrature grid
pointsx, weightsx = quadrature_grid( gridx, ux, wx )
pointsy, weightsy = quadrature_grid( gridy, uy, wy )

# for each element and a quadrature points, 
# we compute the non-vanishing B-Splines
basis_1x = basis_ders_on_quad_grid( knots_1x, px, pointsx, nderiv )
basis_2x = basis_ders_on_quad_grid( knots_2x, px-1, pointsx, nderiv )
basis_1y = basis_ders_on_quad_grid( knots_1y, py, pointsy, nderiv )
basis_2y = basis_ders_on_quad_grid( knots_2y, py-1, pointsy, nderiv )

basis_H1 = basis_from_1D_basis(basis_1x, basis_1y)
basis_Hcurl_1=basis_from_1D_basis(basis_2x, basis_1y)
basis_Hcurl_2=basis_from_1D_basis(basis_1x, basis_2y)


#projection of initial conditions on respective FEM spaces
ux_0=pcurl.coeff(knots_2x, knots_1y, ux_0_re, direction='x')
ux_0=matrix_to_array(nbasis_2x, nbasis_1y, ux_0)
uy_0=pcurl.coeff(knots_1x, knots_2y, uy_0_re, direction='y')
uy_0=matrix_to_array(nbasis_1x, nbasis_2y, uy_0)
p_0=p1.coeff(knots_1x, knots_1y, p_0_re)
p_0=matrix_to_array(nbasis_1x, nbasis_1y, p_0)
p_0=dirichlet_F(p_0, nbasis_1x, nbasis_1y)


#assembling mass and rigidity matrices
print('assembling matrices')
M1=zeros((nbasis_1x*nbasis_1y, nbasis_1x*nbasis_1y))
M1=assemble_M(nex, ney, px, py, spans_1x, spans_1y, basis_H1, weightsx, weightsy, M1)
print('M1 assembled')
Mcurl1=zeros((nbasis_2x*nbasis_1y, nbasis_2x*nbasis_1y))
Mcurl1=assemble_M(nex, ney, px-1, py, spans_2x, spans_1y, basis_Hcurl_1, weightsx, weightsy, Mcurl1)
print('Mcurl1 assembled')
Mcurl2=zeros((nbasis_1x*nbasis_2y, nbasis_1x*nbasis_2y))
Mcurl2=assemble_M(nex, ney, px, py-1, spans_1x, spans_2y, basis_Hcurl_2, weightsx, weightsy, Mcurl2)
print('Mcurl2 assembled')
M1=dirichlet_M(M1, nbasis_1x, nbasis_1y)

K1=zeros((nbasis_2x*nbasis_1y, nbasis_1x*nbasis_1y))
K1=assemble_K(nex, ney, px, py, spans_1x, spans_1y, spans_2x, spans_1y, nbasis_1x, nbasis_1y, basis_Hcurl_1, weightsx, weightsy, pointsx, pointsy, knots_2x, knots_1y, alpha, K1, direction='x')
print('K1 assembled')
K2=zeros((nbasis_1x*nbasis_2y, nbasis_1x*nbasis_1y))
K2=assemble_K(nex, ney, px, py, spans_1x, spans_1y, spans_1x, spans_2y, nbasis_1x, nbasis_1y, basis_Hcurl_2, weightsx, weightsy, pointsx, pointsy, knots_1x, knots_2y, alpha, K2, direction='y')
print('K2 assembled')
K1=dirichlet_K(K1, nbasis_1x, nbasis_1y, nbasis_2x, nbasis_1y)
K2=dirichlet_K(K2, nbasis_1x, nbasis_1y, nbasis_1x, nbasis_2y)

K1=sparse.csr_matrix(K1)
K2=sparse.csr_matrix(K2)
M1=sparse.csr_matrix(M1)


p=p_0
ux=ux_0
uy=uy_0

E=zeros((N+1, 1))
P=zeros(((nbasis_1x-2)*(nbasis_1y-2), int(N)+1))
Ux=zeros((nbasis_2x*nbasis_1y, int(N)+1))
Uy=zeros((nbasis_1x*nbasis_2y, int(N)+1))

E[0]=compute_energy(M1, Mcurl1, Mcurl2, p, ux, uy)
P[:, 0][np.newaxis]=np.transpose(p_0)
Ux[:, 0][np.newaxis]=np.transpose(ux_0)
Uy[:, 0][np.newaxis]=np.transpose(uy_0)


dxp=np.zeros((nbasis_2x, nbasis_1y))
dyp=np.zeros((nbasis_1x, nbasis_2y))

solve(dt, N, M1, Mcurl1, Mcurl2, K1, K2, p_0, ux_0, uy_0, E, P, Ux, Uy)


energy_file = open('energy.txt', 'w+')

energy_file.write("%d\n" % N)
energy_file.write("%f\n" % dt)

for i in range(0, N+1):
	energy_file.write("%f\n" % E[i])

energy_file.close()