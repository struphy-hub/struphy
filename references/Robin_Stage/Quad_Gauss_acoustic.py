import numpy as np
import utilities as ut
import matplotlib.pyplot as plt


def Quad_Gauss_2D_x(a, b, phi, y, Tx, Ty, I, J, px, py, points):
	span_sx=I+px
	span_sy=J+py
	span_y=ut.find_span(Ty, py, y)
	if points==1:
		x1 = (b+a)/2
		span_x=ut.find_span(Tx, px, x1)
		jx=-span_x+span_sx
		jy=-span_y+span_sy
		if jx>=0 and jx<=px and jy>=0 and jy<=py:
			z0=phi(x1, y)*ut.all_bsplines(Tx, px, x1, span_x)[jx]*ut.all_bsplines(Ty, py, y, span_y)[jy]
		else:
			z0=0
		return (b-a)*z0
	


def Quad_Gauss_2D_y(a, b, phi, x, Tx, Ty, I, J, px, py, points):
	span_sx=I+px
	span_sy=J+py
	span_x=ut.find_span(Tx, px, x)
	if points==1:
		y1 = (b+a)/2
		span_y=ut.find_span(Ty, py, y1)
		jx=-span_x+span_sx
		jy=-span_y+span_sy
		if jx>=0 and jx<=px and jy>=0 and jy<=py:
			z0=phi(x, y1)*ut.all_bsplines(Tx, px, x, span_x)[jx]*ut.all_bsplines(Ty, py, y1, span_y)[jy]
		else:
			z0=0
		return (b-a)*z0
	elif points==2:
		y1 = (b+a)/2-(b-a)/(2*np.sqrt(3))
		y2 = (b+a)/2+(b-a)/(2*np.sqrt(3))
		z1 = phi(x, y1)*ut.all_bsplines(Tx, px, x, span_x)[0]*ut.all_bsplines(Ty, py, y1, span_y)[0]
		z2 = phi(x, y2)*ut.all_bsplines(Tx, px, x, span_x)[0]*ut.all_bsplines(Ty, py, y2, span_y)[0]
		return (b-a)/2*(z1+z2)
	elif points==3:
		x1 = (b+a)/2-np.sqrt(3./5)*(b-a)/2
		x2 = (b+a)/2
		x3 = (b+a)/2+np.sqrt(3./5)*(b-a)/2
		z1 = phi(x, y1)*ut.all_bsplines(Tx, px, x, span_x)[0]*ut.all_bsplines(Ty, py, y1, span_y)[0]
		z2 = phi(x, y2)*ut.all_bsplines(Tx, px, x, span_x)[0]*ut.all_bsplines(Ty, py, y2, span_y)[0]
		z3 = phi(x, y3)*ut.all_bsplines(Tx, px, x, span_x)[0]*ut.all_bsplines(Ty, py, y3, span_y)[0]
		return(b-a)/2*(5./9*z1+8./9*z2+5./9*z3)

