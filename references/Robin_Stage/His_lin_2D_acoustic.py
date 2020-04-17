import bsplines as bs
import utilities as ut
import numpy as np
import Quad_Gauss_acoustic as qg
import matplotlib.pyplot as plt

from numpy import shape



p = lambda x,y: 1
alpha = lambda x, y: np.sin(np.pi*x)*np.sin(np.pi*y)


def Dx_L(Tx, Ty, L, DxL):
	from numpy import shape
	n = shape(L)
	nx= n[0]
	ny= n[1]
	for i in range(0, nx-1):
		for j in range(0, ny):
			DxL[i, j]=2*(L[i+1, j]-L[i, j])/(Tx[i+3]-Tx[i+1])



def Dy_L(Tx, Ty, L, DyL):
	from numpy import zeros
	from numpy import shape
	n = shape(L)
	nx= n[0]
	ny= n[1]
	for i in range(0, nx):
		for j in range(0, ny-1):
			DyL[i, j]=2*(L[i, j+1]-L[i, j])/(Ty[j+3]-Ty[j+1])


def lambda_x_i(Tx, Ty, y, i, I, J, f):
	n=len(Tx)-3
	px=1
	py=2
	xm2= Tx[i]
	x0 = Tx[i+1]
	x2 = Tx[i+2]
	xm1= (xm2+x0)/2
	x1 = (x0+x2)/2

	ym2=qg.Quad_Gauss_2D_x(xm2, xm1, f, y, Tx, Ty, I, J, px, py, 1)
	ym1=qg.Quad_Gauss_2D_x(xm1, x0, f, y, Tx, Ty, I, J, px, py, 1)
	y0 =qg.Quad_Gauss_2D_x(x0, x1, f, y, Tx, Ty, I, J, px, py, 1)
	y1 =qg.Quad_Gauss_2D_x(x1, x2, f, y, Tx, Ty, I, J, px, py, 1)
	if i == 0:
		val=(2*(ym2+ym1)+3./2*(y0)-1./2*y1)
	if i>=1 and i<=n-1:
		val=(-1./2*ym2+3./2*(ym1+y0)-1./2*y1)
	if i==n:
		val=-1./2*ym2+3./2*ym1
	val=val*2/(Tx[i+2]-Tx[i])
	return val


def lambda_y_i(Tx, Ty, x, i, I, J, f):
	n=len(Ty)-3
	px=2
	py=1

	xm2= Ty[i]
	x0 = Ty[i+1]
	x2 = Ty[i+2]
	xm1= (xm2+x0)/2
	x1 = (x0+x2)/2

	ym2=qg.Quad_Gauss_2D_y(xm2, xm1, f, x, Tx, Ty, I, J, px, py, 1)
	ym1=qg.Quad_Gauss_2D_y(xm1, x0, f, x, Tx, Ty, I, J, px, py, 1)
	y0 =qg.Quad_Gauss_2D_y(x0, x1, f, x, Tx, Ty, I, J, px, py, 1)
	y1 =qg.Quad_Gauss_2D_y(x1, x2, f, x, Tx, Ty, I, J, px, py, 1)
	if i == 0:
		val=2*(ym2+ym1)+3./2*(y0)-1./2*y1
	if i>=1 and i<=n-1:
		val=(-1./2*ym2+3./2*(ym1+y0)-1./2*y1)
	if i==n:
		val=-1./2*ym2+3./2*ym1
	val=val*2/(Ty[i+2]-Ty[i])
	return val


def lambda_ij(Tx, Ty, i, j, I, J, f, direction):
	if direction=='x':
		n=len(Ty)-3
		y0=lambda_x_i(Tx, Ty, Ty[j+1], i, I, J, f)
		y2=lambda_x_i(Tx, Ty, Ty[j+2], i, I, J, f)
		y1=lambda_x_i(Tx, Ty, .5*Ty[j+1]+.5*Ty[j+2], i, I, J, f)
		if 1<=j and j<=n-1:
			val=-0.5*y0+2*y1-0.5*y2
		elif j == 0:
			val=y0
		elif j==n:
			val=y2
	elif direction=='y':
		n=len(Tx)-3
		y0=lambda_y_i(Tx, Ty, Tx[i+1], j, I, J, f)
		y2=lambda_y_i(Tx, Ty, Tx[i+2], j, I, J, f)
		y1=lambda_y_i(Tx, Ty, .5*Tx[i+1]+.5*Tx[i+2], j, I, J, f)		
		if 1<=i and i<=n-1:
			val=-0.5*y0+2*y1-0.5*y2
		elif i == 0:
			val=y0
		elif i==n:
			val=y2
	return val



def lambda_ij_der(Tx, Ty, i, j, f, direction):
	nx=len(Tx)-3
	ny=len(Ty)-3
	if direction=='x':
		if i>0 and i<nx+2:
			l1=2/(Tx[i+1]-Tx[i-1])*lambda_ij(Tx, Ty, i-1, j, i-1, j, f, direction)
		else:
			l1=0
		if i+1>0 and i+1<nx+2:
			l2=2/(Tx[i+2]-Tx[i])*lambda_ij(Tx, Ty, i, j, i, j, f, direction)
		else:
			l2=0
	elif direction=='y':
		if j>0 and j<ny+2:
			l1=2/(Ty[j+1]-Ty[j-1])*lambda_ij(Tx, Ty, i, j-1, i, j-1, f, direction)
		else:
			l1=0
		if j+1>0 and j+1<ny+2:
			l2=2/(Ty[j+2]-Ty[j])*lambda_ij(Tx, Ty, i, j, i, j, f, direction)
		else:
			l2=0
	return l1, l2


def evaluate_projection_der(Tx, Ty, i, j, l1, l2, x, y, direction):
	nx=len(Tx)-3
	ny=len(Ty)-3
	if direction=='x':
		if i>0 and i<nx+2:
			val1=QI_loc(Tx, Ty, l1, i-1, j, x, y, direction)
		else:
			val1=0
		if i+1>0 and i+1<nx+2:
			val2=QI_loc(Tx, Ty, l2, i, j, x, y, direction)
		else:
			val2=0
	elif direction=='y':
		if j>0 and j<ny+2:
			val1=QI_loc(Tx, Ty, l1, i, j-1, x, y, direction)
		else:
			val1=0
		if j+1>0 and j+1<ny+2:
			val2=QI_loc(Tx, Ty, l2, i, j, x, y, direction)
		else:
			val2=0
	return val1-val2



def coeff(Tx, Ty, I, J, f, direction):
	if direction=='x':
		nx=len(Tx)-2
		ny=len(Ty)-3
	elif direction=='y':
		nx=len(Tx)-3
		ny=len(Ty)-2
	L=np.zeros((nx, ny), dtype=float)
	for i in range(0, nx):
		for j in range(0, ny):
			L[i, j]=lambda_ij(Tx, Ty, i, j, I, J, f, direction)
	return L


def coeff_spline(Tx, Ty, P, f, direction):
	if direction=='x':
		nx=len(Tx)-2
		ny=len(Ty)-3
	elif direction=='y':
		nx=len(Tx)-3
		ny=len(Ty)-2
	L=np.zeros((nx, ny), dtype=float)
	for i in range(0, nx):
		for j in range(0, ny):
			L[i, j]=P[i, j]*lambda_ij(Tx, Ty, i, j, i, j, f, direction)
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


def QI_loc(Tx, Ty, l, i, j, x, y, direction):

	if direction=='x':
		span_sx=i+1
		span_sy=j+2
		span_x = ut.find_span(Tx, 1, x)
		N_x = ut.all_bsplines(Tx, 1, x, span_x)
		span_y = ut.find_span(Ty, 2, y)
		N_y = ut.all_bsplines(Ty, 2, y, span_y)
		val=0
		jx=-span_x+span_sx
		jy=-span_y+span_sy
		if jx>=0 and jx<=1 and jy>=0 and jy<=2:
			val = l*N_x[jx]*N_y[jy]
	elif direction=='y':
		span_sx=i+2
		span_sy=j+1
		span_x = ut.find_span(Tx, 2, x)
		N_x = ut.all_bsplines(Tx, 2, x, span_x)
		span_y = ut.find_span(Ty, 1, y)
		N_y = ut.all_bsplines(Ty, 1, y, span_y)
		val=0
		jx=-span_x+span_sx
		jy=-span_y+span_sy
		if jx>=0 and jx<=2 and jy>=0 and jy<=1:
			val = l*N_x[jx]*N_y[jy]
	return val


def plot_QI  (Tx, Ty, I, J, nx, f, direction):

	L=coeff(Tx, Ty, I, J, f, direction)

	#Interpolated function and real function vectors on the points of X

	X     = np.linspace(Tx[0], Tx[-1], nx+1)
	Y_int = np.zeros((nx+1,nx+1), dtype=float)
	Y     = np.zeros((nx+1,nx+1), dtype=float)
	for i in range(0, nx+1):
		for j in range(0, nx+1):
			Y_int[i, j] = QI(Tx, Ty, L, X[i], X[j], direction)
			Y[i, j] = f(X[i], X[j])

	#Plot of the interpolation and the real function

	plt.imshow(Y_int,origin='lower',interpolation='bilinear')
	plt.colorbar()

def plot_QI_spline  (Tx, Ty, P, nx, f, direction):

	L=coeff_spline(Tx, Ty, P, f, direction)

	#Interpolated function and real function vectors on the points of X
	X     = np.linspace(Tx[0], Tx[-1], nx+1)
	Y_int = np.zeros((nx+1,nx+1), dtype=float)
	Y     = np.zeros((nx+1,nx+1), dtype=float)
	for i in range(0, nx+1):
		for j in range(0, nx+1):
			Y_int[i, j] = QI(Tx, Ty, L, X[i], X[j], direction)
			Y[i, j] = f(X[i], X[j])


	#Plot of the interpolation and the real function

	plt.imshow(Y_int,origin='lower',interpolation='bilinear')
	plt.colorbar()
