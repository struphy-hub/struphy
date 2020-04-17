import numpy as np
import utilities as ut

def Quad_Gauss_loc(a, b, phi, i):
	if i==1:
		return (b-a)*phi((b+a)/2)
	elif i==2:
		y1 = phi((b+a)/2-(b-a)/(2*np.sqrt(3)))
		y2 = phi((b+a)/2+(b-a)/(2*np.sqrt(3)))
		return (b-a)/2*(y1+y2)
	elif i==3:
		y1 = phi((b+a)/2-np.sqrt(3./5)*(b-a)/2)
		y2 = phi((b+a)/2)
		y3 = phi((b+a)/2+np.sqrt(3./5)*(b-a)/2)
		return(b-a)/2*(5./9*y1+8./9*y2+5./9*y3)

def Quad_Gauss(a, b, n, phi, i):
	X=np.linspace(a, b, n)
	val=0
	for j in range(0, n-1):
		val+=Quad_Gauss_loc(X[j], X[j+1], phi, i)
	return val

def Quad_Gauss_loc_2D_x(a, b, phi, y, i):
	if i==1:
		return (b-a)*phi((b+a)/2, y)
	elif i==2:
		y1 = phi((b+a)/2-(b-a)/(2*np.sqrt(3)), y)
		y2 = phi((b+a)/2+(b-a)/(2*np.sqrt(3)), y)
		return (b-a)/2*(y1+y2)
	elif i==3:
		y1 = phi((b+a)/2-np.sqrt(3./5)*(b-a)/2, y)
		y2 = phi((b+a)/2, y)
		y3 = phi((b+a)/2+np.sqrt(3./5)*(b-a)/2, y)
		return(b-a)/2*(5./9*y1+8./9*y2+5./9*y3)


def Quad_Gauss_2D_x(a, b, n, phi, y, i):
	X=np.linspace(a, b, n)
	val=0
	for j in range(0, n-1):
		val+=Quad_Gauss_loc_2D_x(X[j], X[j+1], phi, y, i)
	return val

def Quad_Gauss_loc_2D_y(a, b, phi, x, i):
	if i==1:
		return (b-a)*phi(x, (b+a)/2)
	elif i==2:
		y1 = phi(x, (b+a)/2-(b-a)/(2*np.sqrt(3)))
		y2 = phi(x, (b+a)/2+(b-a)/(2*np.sqrt(3)))
		return (b-a)/2*(y1+y2)
	elif i==3:
		y1 = phi(x, (b+a)/2-np.sqrt(3./5)*(b-a)/2)
		y2 = phi(x, (b+a)/2)
		y3 = phi(x, (b+a)/2+np.sqrt(3./5)*(b-a)/2)
		return(b-a)/2*(5./9*y1+8./9*y2+5./9*y3)


def Quad_Gauss_2D_y(a, b, n, phi, x, i):
	X=np.linspace(a, b, n)
	val=0
	for j in range(0, n-1):
		val+=Quad_Gauss_loc_2D_y(X[j], X[j+1], phi, x, i)
	return val