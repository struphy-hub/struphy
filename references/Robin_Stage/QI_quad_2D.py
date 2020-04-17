import utilities as ut 
import bsplines as bs
import numpy as np
import matplotlib.pyplot as plt

f = lambda x,y: x

def lambda_x_i(Tx, y, i, f):
	#Computes the i_th component of the interpolation of f in the x direction

	nx=len(Tx)-3
	y0=f(Tx[i+1], y)
	y2=f(Tx[i+2], y)
	y1=f((Tx[i+1]+Tx[i+2])/2, y)
	if 1<=i and i<=nx-1:
		val=-0.5*y0+2*y1-0.5*y2
	elif i == 0:
		val=y0
	elif i==nx:
		val=y2
	return val


def lambda_ij(Tx, Ty, i, j, f):
	#Computes the ij_th component of the interpolation of f

	ny=len(Ty)-3
	y0=lambda_x_i(Tx, Ty[j+1], i, f)
	y2=lambda_x_i(Tx, Ty[j+2], i, f)
	y1=lambda_x_i(Tx, 0.5*Ty[j+1]+0.5*Ty[j+2], i, f)
	if 1<=j and j<=ny-1:
		val=-0.5*y0+2*y1-0.5*y2
	elif j == 0:
		val=y0
	elif j==ny:
		val=y2
	return val


def coeff(Tx, Ty, f):
	#Builds a matrix for the components of the interpolation of f

	nx=len(Tx)-3
	ny=len(Ty)-3
	L=np.zeros((nx, ny), dtype=float)
	for i in range(0, nx):
		for j in range(0, ny):
			L[i, j]=lambda_ij(Tx, Ty, i, j, f)
	return L

def QI(Tx, Ty, L, x, y):
	"""
	Computes the value of the interpolated function at a point x,y

	Parameters
	----------
	T : array
	knot sequence

	L : array
	coefficients for the interpolation

	x,y : floats
	point at witch the interpolated function is evaluated

	Returns
	-------
	val : float
	value of the interpolated function at the point x,y
	"""
	span_x = ut.find_span(Tx, 2, x)
	N_x = ut.all_bsplines(Tx, 2, x, span_x)
	span_y = ut.find_span(Ty, 2, y)
	N_y = ut.all_bsplines(Ty, 2, y, span_y)
	val = 0
	for i in range(0, 3):
		for j in range(0, 3):
			I=i+span_x-2
			J=j+span_y-2
			val+=L[I, J]*N_x[i]*N_y[j]
	return val

def plot_QI  (x_0, x_n, nbx, nby, nx):
	"""
	Plots the interpolated function and the real function in the interval [x_0, x_n]

	Parameters
	----------
	x_0 : float
	first point of the interval

	x_n : float
	last point of the interval

	nb : int
	number of grid points

	nx : int
	number of points at witch the functions are evaluated for the plot

	"""

	#Knots vector construction

	xs = np.linspace(x_0, x_n, nbx)
	ys = np.linspace(x_0, x_n, nbx)
	Tx = [x_0]*2 + list(xs) + [x_n]*2
	Ty = [x_0]*2 + list(xs) + [x_n]*2
	Tx = np.asarray(Tx)
	Ty = np.asarray(Ty)
	L=coeff(Tx, Ty, f)

	#Interpolated function and real function vectors on the points of X

	X     = np.linspace(x_0, x_n, nx+1)
	Y_int = np.zeros((nx+1,nx+1), dtype=float)
	Y     = np.zeros((nx+1,nx+1), dtype=float)
	for i in range(0, nx+1):
		for j in range(0, nx+1):
			Y_int[i, j] = QI(Tx, Ty, L, X[i], X[j])
			Y[i, j] = f(X[i], X[j])

	#Plot of the interpolation and the real function

	plt.figure(figsize=(30, 10))

	plt.subplot(131)
	plt.imshow(Y_int,origin='lower',interpolation='bilinear')
	plt.colorbar()
	plt.subplot(132)
	plt.imshow(Y,origin='lower',interpolation='bilinear')
	plt.colorbar()
	plt.subplot(133)
	plt.imshow(Y-Y_int,origin='lower',interpolation='bilinear')
	plt.colorbar()
	plt.show()

"""
plot_QI(0, 1, 8, 8, 100)
"""