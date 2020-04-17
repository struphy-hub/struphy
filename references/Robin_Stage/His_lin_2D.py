import bsplines as bs
import utilities as ut
import numpy as np
import Quad_Gauss as qg
import matplotlib.pyplot as plt


def lambda_x_i(Tx, y, i, f):
	n=len(Tx)-3
	nb=100

	xm2= Tx[i]
	x0 = Tx[i+1]
	x2 = Tx[i+2]
	xm1= (xm2+x0)/2
	x1 = (x0+x2)/2

	ym2=qg.Quad_Gauss_2D_x(xm2, xm1, nb, f, y, 3)
	ym1=qg.Quad_Gauss_2D_x(xm1, x0,  nb, f, y, 3)
	y0 =qg.Quad_Gauss_2D_x(x0,  x1,  nb, f, y, 3)
	y1 =qg.Quad_Gauss_2D_x(x1,  x2,  nb, f, y, 3)

	if i == 0:
		val=(2*(ym2+ym1)+3./2*(y0)-1./2*y1)
	if i>=1 and i<=n-1:
		val=(-1./2*ym2+3./2*(ym1+y0)-1./2*y1)
	if i==n:
		val=-1./2*ym2+3./2*ym1
	val=val*2/(Tx[i+2]-Tx[i])
	return val

def lambda_y_i(Ty, x, i, f):
	n=len(Ty)-3
	nb=100

	xm2= Ty[i]
	x0 = Ty[i+1]
	x2 = Ty[i+2]
	xm1= (xm2+x0)/2
	x1 = (x0+x2)/2

	ym2=qg.Quad_Gauss_2D_y(xm2, xm1, nb, f, x, 3)
	ym1=qg.Quad_Gauss_2D_y(xm1, x0,  nb, f, x, 3)
	y0 =qg.Quad_Gauss_2D_y(x0,  x1,  nb, f, x, 3)
	y1 =qg.Quad_Gauss_2D_y(x1,  x2,  nb, f, x, 3)

	if i == 0:
		val=2*(ym2+ym1)+3./2*(y0)-1./2*y1
	if i>=1 and i<=n-1:
		val=(-1./2*ym2+3./2*(ym1+y0)-1./2*y1)
	if i==n:
		val=-1./2*ym2+3./2*ym1
	val=val*2/(Ty[i+2]-Ty[i])
	return val


def lambda_ij(Tx, Ty, i, j, f, direction):
	if direction=='x':
		n=len(Ty)-3
		y0=lambda_x_i(Tx, Ty[j+1], i, f)
		y2=lambda_x_i(Tx, Ty[j+2], i, f)
		y1=lambda_x_i(Tx, 0.5*Ty[j+1]+0.5*Ty[j+2], i, f)
		if 1<=j and j<=n-1:
			val=-0.5*y0+2*y1-0.5*y2
		elif j == 0:
			val=y0
		elif j==n:
			val=y2
		return val
	elif direction=='y':
		n=len(Tx)-3
		y0=lambda_y_i(Ty, Tx[i+1], j, f)
		y2=lambda_y_i(Ty, Tx[i+2], j, f)
		y1=lambda_y_i(Ty, 0.5*Tx[i+1]+0.5*Tx[i+2], j, f)
		if 1<=i and i<=n-1:
			val=-0.5*y0+2*y1-0.5*y2
		elif i == 0:
			val=y0
		elif i==n:
			val=y2
		return val

def coeff(Tx, Ty, f, direction):
	if direction=='x':
		nx=len(Tx)-2
		ny=len(Ty)-3
	elif direction=='y':
		nx=len(Tx)-3
		ny=len(Ty)-2
	L=np.zeros((nx, ny), dtype=float)
	for i in range(0, nx):
		for j in range(0, ny):
			L[i, j]=lambda_ij(Tx, Ty, i, j, f, direction)
	return L

def QI(Tx, Ty, L, x, y, direction):

	if direction=='x':
		span_x = ut.find_span(Tx, 1, x)
		N_x = ut.all_bsplines(Tx, 1, x, span_x)
		span_y = ut.find_span(Ty, 2, y)
		N_y = ut.all_bsplines(Ty, 2, y, span_y)
		val = 0
		for i in range(0, 2):
			for j in range(0, 3):
				I=i+span_x-1
				J=j+span_y-2
				val+=L[I, J]*N_x[i]*N_y[j]
	elif direction=='y':
		span_y = ut.find_span(Ty, 1, y)
		N_y = ut.all_bsplines(Ty, 1, y, span_y)
		span_x = ut.find_span(Tx, 2, x)
		N_x = ut.all_bsplines(Tx, 2, x, span_x)
		val = 0
		for i in range(0, 3):
			for j in range(0, 2):
				I=i+span_x-2
				J=j+span_y-1
				val+=L[I, J]*N_x[i]*N_y[j]
	return val

def plot_QI  (x_0, x_n, nbx, nby, nx, direction):

	xs = np.linspace(x_0, x_n, nbx)
	ys = np.linspace(x_0, x_n, nby)
	if direction=='x':
		Tx = [x_0]*1 + list(xs) + [x_n]*1
		Ty = [x_0]*2 + list(ys) + [x_n]*2
		Tx = np.asarray(Tx)
		Ty = np.asarray(Ty)
	elif direction=='y':
		Tx = [x_0]*2 + list(xs) + [x_n]*2
		Ty = [x_0]*1 + list(ys) + [x_n]*1
		Tx = np.asarray(Tx)
		Ty = np.asarray(Ty)
	L=coeff(Tx, Ty, f, direction)

	#Interpolated function and real function vectors on the points of X

	X     = np.linspace(x_0, x_n, nx+1)
	Y_int = np.zeros((nx+1,nx+1), dtype=float)
	Y     = np.zeros((nx+1,nx+1), dtype=float)
	for i in range(0, nx+1):
		for j in range(0, nx+1):
			Y_int[i, j] = QI(Tx, Ty, L, X[i], X[j], direction)
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

