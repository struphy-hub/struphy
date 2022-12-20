import sys

import numpy as np
import matplotlib.pyplot as plt

from struphy.geometry import domains
from struphy.fields_background.mhd_equil.analytical import ScrewPinch
from struphy.diagnostics.continuous_spectra import get_mhd_continua_2d

from struphy.eigenvalue_solvers.spline_space import Spline_space_1d, Tensor_spline_space
from struphy.eigenvalue_solvers.mhd_axisymmetric_main import solve_mhd_ev_problem_2d


# ----- numerical parameters ----
Nel      = [32, 24]
p        = [2, 3]
spl_kind = [False, True]
nq_el    = [6, 6]
nq_pr    = [6, 6]
bc       = [['f', 'd'], ['f', 'f']]
polar_ck = 1

num_params = {'Nel' : Nel, 'p' : p, 'spl_kind' : spl_kind, 'nq_el' : nq_el, 'nq_pr' : nq_pr, 'bc' : bc, 'polar_ck' : polar_ck}

n_tor = -1
b_tor = 'i'

# ------ mapping parameters -----
dom_type    = 'PoloidalSplineCylinder'
a          = 1.
R0         = 5.
dom_params = {'a' : a, 'R0' : R0, 'Nel' : Nel, 'p' : p, 'spl_kind' : spl_kind}


# -- MHD equilibrium parameters --
params_mhd = {}

# minor/major radius
params_mhd['a']  = a 
params_mhd['R0'] = R0

# toroidal field
params_mhd['B0'] = 1.

# safety factor q(r) = q0 + (q1 - q0)*(r/a)^2
params_mhd['q0'] = 0.80
params_mhd['q1'] = 1.85

# number density n(r) = (1 - na)*(1 - (r/a)^n1)^n2 + na
params_mhd['n1'] = 4. 
params_mhd['n2'] = 3.
params_mhd['na'] = 0.

# create domain
domain_class = getattr(domains, dom_type)
domain = domain_class(dom_params)

# for plotting
etaplot = [np.linspace(0., 1., 201), np.linspace(0., 1., 101)]

xplot = domain.evaluate(etaplot[0], etaplot[1], 0., 'x')
yplot = domain.evaluate(etaplot[0], etaplot[1], 0., 'y')
rplot = etaplot[0]*a

# set up 1d spline spaces and set projectors
fem_1d_1 = Spline_space_1d(Nel[0], p[0], spl_kind[0], nq_el[0], bc[0])
fem_1d_2 = Spline_space_1d(Nel[1], p[1], spl_kind[1], nq_el[1], bc[1])

fem_1d_1.set_projectors(nq_pr[0])
fem_1d_2.set_projectors(nq_pr[1])

# set up 2d tensor product space with polar splines
fem_2d = Tensor_spline_space([fem_1d_1, fem_1d_2], polar_ck, domain.cx[:, :, 0], domain.cy[:, :, 0], n_tor, b_tor)

fem_2d.set_projectors('general')

# load MHD equilibrium 
eq_mhd = ScrewPinch(params_mhd, domain)

# solve eigenvalue problem
omega2_eig, U_eig, MAT = solve_mhd_ev_problem_2d(num_params, eq_mhd, n_tor, b_tor)


# analyze spectrum and plot continua
A, S = get_mhd_continua_2d(fem_2d, domain, omega2_eig, U_eig, [0, 12], params_mhd['B0']/params_mhd['R0'], 1e-3, 3)


fig, ax = plt.subplots(2, 3)
fig.set_figheight(12)
fig.set_figwidth(14)

ax[0, 0].plot(rplot, eq_mhd.q(rplot))
ax[0, 1].plot(rplot, eq_mhd.pr(rplot))
ax[0, 2].plot(rplot, eq_mhd.nr(rplot))

ax[0, 0].set_xlabel('$r$')
ax[0, 1].set_xlabel('$r$')
ax[0, 2].set_xlabel('$r$')

ax[0, 0].set_ylabel('$q$')
ax[0, 1].set_ylabel('$p$')
ax[0, 2].set_ylabel('$n$')

ax[0, 0].set_title('Safety factor')
ax[0, 1].set_title('Pressure')
ax[0, 2].set_title('Number density')


xgrid = domain.evaluate(fem_2d.el_b[0], fem_2d.el_b[1], 0., 'x')
ygrid = domain.evaluate(fem_2d.el_b[0], fem_2d.el_b[1], 0., 'y')

for i in range(xgrid.shape[0]):
    ax[1, 0].plot(xgrid[i, :], ygrid[i, :], 'k')
    
for i in range(xgrid.shape[1]):
    ax[1, 0].plot(xgrid[:, i], ygrid[:, i], 'r')
    
ax[1, 0].set_xlabel('x')
ax[1, 0].set_ylabel('y')
ax[1, 0].set_title(r'Grid : $N_\mathrm{el}=$' + str(fem_2d.Nel[:2]), pad=10)


#  ====================== plot shear Alfvén continuum =========================
exponent_A = 2
norm = params_mhd['B0']/R0

ms_plot = [1, 2]

for m in ms_plot:  
    ax[1, 1].plot(A[m][0]*a, (np.sqrt(A[m][1])/norm)**exponent_A, '+', label='m = ' + str(m))
    
ax[1, 1].set_xlabel('$r$')

if exponent_A == 1:
    ax[1, 1].set_ylabel('$\omega/\omega_A$')
else:
    ax[1, 1].set_ylabel('$\omega^2/\omega_A^2$')

ax[1, 1].set_xlim((0., a))
ax[1, 1].set_ylim((0., 1.0))
ax[1, 1].legend()
ax[1, 1].set_title('Shear Alfvén continuum', pad=10)
# =========================================================================


# ==================== plot slow sound continuum ==========================
exponent_S = 2
norm = params_mhd['B0']/R0

ms_plot = [0, 1, 2]

for m in ms_plot:  
    ax[1, 2].plot(S[m][0]*a, (np.sqrt(S[m][1])/norm)**exponent_S, '+', label='m = ' + str(m))

ax[1, 2].set_xlabel('$r$')

if exponent_S == 1:
    ax[1, 2].set_ylabel('$\omega/\omega_A$')
else:
    ax[1, 2].set_ylabel('$\omega^2/\omega_A^2$')

ax[1, 2].set_xlim((0., a))
ax[1, 2].set_ylim((0., 0.05))
ax[1, 2].legend()
ax[1, 2].set_title('Slow sound continuum', pad=10)
# =========================================================================

plt.subplots_adjust(wspace=0.4)
plt.subplots_adjust(hspace=0.5)
plt.show()