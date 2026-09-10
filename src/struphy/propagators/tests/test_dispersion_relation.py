import logging

import numpy as np
import cunumpy as xp
from matplotlib import pyplot as plt
from feectools.ddm.mpi import mpi as MPI

from struphy import (
    BinningPlot,
    BoundaryParameters,
    LoadingParameters,
    WeightsParameters,
    domains,
    equils,
    perturbations,
    set_logging_level,
)
from struphy.feec.mass import L2Projector, WeightedMassOperators
from struphy.feec.basis_projection_ops import BasisProjectionOperators
from struphy.feec.psydac_derham import Derham
from struphy.geometry.base import Domain
from struphy.io.options import DerhamOptions
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.fields_background.projected_equils import ProjectedFluidEquilibriumWithB
from struphy.propagators.perturbation_system_cold import ColdPlasmaPerturbation
from struphy.topology.grids import TensorProductGrid
from struphy.utils.pyccel import Pyccelkernel

logger = logging.getLogger("struphy")
set_logging_level(logging.DEBUG)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
plt.rcParams.update({"font.size": 22})

tol: float = 1e-13

m_e: float = 9.1094e-31 # kg
e: float = 1.51874e-14 # kg^1/2 m^3/2 s^{-1}
c: float = 299792458 # m/s
mu0: float = 4*np.pi / (c**2)
kB: float = 1.380649e-23 # kg m^2 s^{-2} K^{-1}

N: float = 1e18 # m^{-3}

kT = 10*1.602176621e-19 #J
T: float = kT / kB

eratio: float = 1.602e-19 / e # e SI over e Gauss
B: float = 1e-4 # Tesla
B_gauss: float = B * c * eratio

omega_pe_scale = np.sqrt((4*np.pi*N*(e**2)/(m_e)))

debye: float = np.sqrt(kT / (4*np.pi * N * (e**2)))

L = c / omega_pe_scale

Vcyclo: float = L*e*B_gauss/(2*np.pi*m_e * c) # cyclotron speed
Valfven: float = B_gauss / np.sqrt(m_e*N*mu0*c) # Alfven speed

V: float = np.sqrt(kT / m_e)

t: float = L/V
print(f"{t=}")

E = kT / (e*L)

B = E * c / V

alpha: float = (debye / L)**2

c_normalized = c/V
print(f"{c_normalized=}")

omega: float = 100000000. * t
print(f"{omega=}")
rhobar: float = 0.
mass: float = 1.
theta: float = 0. # 200.
zeta: float = np.pi * 3/5 # between 0 and pi/2
B0: float = 5.
coszeta = np.cos(zeta)
sinzeta = np.sin(zeta)

B_x = 0. # B0*sinzeta
B_y = 0.
B_z = 0. # B0*coszeta

omega_pe: float = np.sqrt(rhobar) / mass
print(f"{omega_pe=}")

omega_pe_normalized: float = omega_pe * omega_pe_scale * t
print(f"{omega_pe_normalized=}")

d_omega = np.sqrt(np.abs(omega**2 - omega_pe_normalized**2))
k_light = d_omega / c_normalized

# c_sound_normalized = np.sqrt(theta / mass)  # only activate k_sound and anything related to it when theta is nonzero
# k_sound = d_omega / c_sound_normalized

c0: float = (c_normalized**2) * alpha
c1: float = (omega**2) * alpha

print(f"{k_light/L=}")
# print(f"{k_sound/L=}")

def test_dispersion_relation_1d():
    ksquared = lambda k1,k2: ((k1**2)+(k2**2))*id3x3

    ktensork = lambda k1,k2: np.array([[k1**2, k1*k2, 0],
                                        [k2*k1, k2**2, 0],
                                        [0, 0, 0]])

    rotB = np.array([[0, -B_z, B_y],
                    [B_z, 0, -B_x],
                    [-B_y, B_x, 0]])

    id3x3 = np.identity(3)

    def matrix(k1,k2):
        mat = np.zeros((6,6), dtype="complex")

        mat[:3,:3] = id3x3 - theta / (mass * (omega**2)) * ktensork(k1,k2) - 1j / (mass*c_normalized*omega) * rotB
        mat[:3,3:6] = 1j * np.sqrt(4*np.pi) * omega_pe * id3x3
        mat[3:6,:3] = - 1j * omega_pe / (np.sqrt(4*np.pi)*alpha) * id3x3
        mat[3:6,3:6] = (omega**2) * id3x3 - (c_normalized**2) * (ksquared(k1,k2) - ktensork(k1,k2))

        return mat

    determinant = lambda k1, k2: np.linalg.det(matrix(k1,k2))

    print(determinant(k_light,0)/(c_normalized**2))
    # print(determinant(k_sound,0)/(c_normalized**2))
    # print(determinant((k_light+k_sound)/2,0))

    if d_omega == 0.:
        kmax = 0.5
    else:
        kmax = 1.5 * k_light # np.maximum(k_light, k_sound)

    Nel = 450

    k = np.linspace(-kmax, kmax, Nel)

    k1, k2 = np.meshgrid(k, k, indexing='ij')

    det = np.zeros((Nel,Nel))

    for i in range(Nel):
        for j in range(Nel):
            val = np.real(determinant(k1[i,j],k2[i,j]))
            det[i,j] = 0. if np.abs(val) <= tol else val

    det /= c_normalized**2

    detslice = (det[int(Nel/2),:]+det[int(Nel/2)+1,:])/2

    # ax = plt.figure().add_subplot(projection='3d')

    # surface = ax.plot_surface(k1,k2,det,linewidth=0)

    # plt.contour(k1, k2, det, levels=[0.])

    # plt.colorbar()

    plt.figure(1)
    plotyscale = np.max(detslice)
    plotxscale = 2*kmax/L
    plt.title("Determinant of the system of oscillations")
    plt.plot(k/L, detslice, label="Horizontal slice of determinant")
    plt.axhline(y=0.,color="black")
    plt.axvline(x=k_light/L, ymin=0.05, linestyle='--',color="black")
    plt.axvline(x=-k_light/L, ymin=0.05, linestyle='--',color="black")
    # plt.axvline(x=k_sound/L, ymin=0.05, linestyle='--',color="black")
    # plt.axvline(x=-k_sound/L, ymin=0.05, linestyle='--',color="black")
    plt.xlabel("Wave vectors [$m^{-1}$]")
    plt.text(k_light/L - 0.02*plotxscale,-0.035*plotyscale,f"+$k_{{\\mathrm{{light}}}}$")
    plt.text(-k_light/L - 0.02*plotxscale,-0.035*plotyscale,f"-$k_{{\\mathrm{{light}}}}$")
    # plt.text(k_sound/L - 0.02*plotxscale,-0.035*plotyscale,f"+$k_{{\\mathrm{{therminc}}}}$")
    # plt.text(-k_sound/L - 0.02*plotxscale,-0.035*plotyscale,f"-$k_{{\\mathrm{{thermic}}}}$")
    plt.legend()

    # plt.show()
    # exit()

    p = 3
    Ngrid = 2**8

    Nfft = 100
    k_cutoff = kmax
    dk = k_cutoff / Nfft
    maxL: float = 2*np.pi / dk # so that our largest wavenumber value is included

    domain = domains.Cuboid(l1=-maxL/2 ,r1=maxL/2)
    equil = equils.HomogenSlab(B0x=B_x, B0y=B_y, B0z=B_z)
    equil.domain = domain

    e = np.linspace(0., 1., Nel)
    e_x, e_y, e_z = domain(e,0.,0.) # the values the field will be sampled on that will correspond exactly to the k array after the FFT
    e_x = e_x[:,0,0]
    print(e_x)
    cellsize = (maxL / Nel)
    e_k = np.linspace(-Nel/2 * dk, (Nel/2-1) * dk, Nel)

    J0: float = 1.
    j_physical = lambda x,y,z: J0 * np.sinc(k_cutoff/np.pi * x)
    zeroes = lambda x,y,z: 0.*(x+y+z)
    print(f"{k_cutoff/np.pi *maxL=}")
    zeroes = lambda x,y,z: 0. * (x+y+z)

    j_pulled_1 = lambda e1,e2,e3: domain.pull([j_physical,j_physical,zeroes],e1,e2,e3,kind="1", squeeze_out=False)[0]
    j_pulled_2 = lambda e1,e2,e3: domain.pull([j_physical,j_physical,zeroes],e1,e2,e3,kind="1", squeeze_out=False)[1]
    j_pulled_3 = lambda e1,e2,e3: domain.pull([j_physical,j_physical,zeroes],e1,e2,e3,kind="1", squeeze_out=False)[2]

    print(f"{np.shape(e)=}")
    print(f"{np.shape(e_x)=}")
    print(f"{np.shape(j_pulled_1(e,0.,0.)[:,0,0])=}")
    print(f"{np.shape(j_physical(e_x,0.,0.))=}")

    jdiff1 = lambda e1,e2,e3: j_pulled_1(e1,e2,e3) / maxL - j_physical(domain(e1,e2,e3)[0],domain(e1,e2,e3)[1],domain(e1,e2,e3)[2])
    jdiff2 = lambda e1,e2,e3: j_pulled_2(e1,e2,e3) - j_physical(domain(e1,e2,e3)[0],domain(e1,e2,e3)[1],domain(e1,e2,e3)[2])
    jdiff3 = lambda e1,e2,e3: j_pulled_3(e1,e2,e3) - zeroes(domain(e1,e2,e3)[0],domain(e1,e2,e3)[1],domain(e1,e2,e3)[2])

    plt.figure(2)
    plt.subplot(1,3,1)
    plt.plot(e, jdiff1(e,0.,0.)[:,0,0],label="pulled j1")
    plt.legend()
    plt.subplot(1,3,2)
    plt.plot(e, jdiff2(e,0.,0.)[:,0,0],label="pulled j2")
    plt.legend()
    plt.subplot(1,3,3)
    plt.plot(e, jdiff3(e,0.,0.)[:,0,0],label="pulled j3")
    plt.legend()

    plt.figure(3)
    plt.subplot(2,3,1)
    plt.plot(e, j_physical(e_x,0.,0.),label="physical j1")
    plt.legend()
    plt.subplot(2,3,2)
    plt.plot(e, j_physical(e_x,0.,0.),label="physical j2")
    plt.legend()
    plt.subplot(2,3,3)
    plt.plot(e, zeroes(e_x,0.,0.),label="physical j3")
    plt.legend()
    plt.subplot(2,3,4)
    plt.plot(e, j_pulled_1(e,0.,0.)[:,0,0],label="pulled j1")
    plt.legend()
    plt.subplot(2,3,5)
    plt.plot(e, j_pulled_2(e,0.,0.)[:,0,0],label="pulled j2")
    plt.legend()
    plt.subplot(2,3,6)
    plt.plot(e, j_pulled_3(e,0.,0.)[:,0,0],label="pulled j3")
    plt.legend()

    # plt.show()
    # exit()

    degree = (p,1,1)
    num_elements = (Ngrid,1,1)
    bcs = (("dirichlet","dirichlet"), None, None)

    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs)
    derham = Derham(grid=grid, options=derham_opts, comm=comm)
    projected_equil = ProjectedFluidEquilibriumWithB(equil=equil, derham=derham)

    mass_ops = WeightedMassOperators(derham=derham, domain=domain)
    basis_ops = BasisProjectionOperators(derham=derham, domain=domain)

    Propagator.derham = derham
    Propagator.domain = domain
    Propagator.mass_ops = mass_ops
    Propagator.basis_ops = basis_ops
    Propagator.projected_equil = projected_equil

    J = FEECVariable(space="Hcurl")
    J.allocate(derham=derham, domain=domain)
    J.spline.vector = derham.P1([j_pulled_1,j_pulled_2,j_pulled_3])

    ee1, ee2, ee3 = np.meshgrid(np.linspace(0.,1.,4),np.linspace(0.,1.,5),np.linspace(0.,1.,6), indexing="ij")
    # print(f"{np.shape(sincheck(ee1,ee2,ee3))=}")
    print(f"{np.shape(j_pulled_1(ee1,ee2,ee3))=}")
    print(f"{e=}")
    # exit()

    plt.figure(4)
    # plt.plot(e, J0.spline(e,0.,0.)[:,0,0],label="projected j1")
    # plt.plot(e, j_pulled_1(e,0.,0.)[:,0,0],'x',label="j1")
    plt.subplot(3,3,1)
    plt.plot(e, j_pulled_1(e,0,0.)[:,0,0] ,label="j1")
    plt.legend()
    plt.subplot(3,3,2)
    plt.plot(e, j_pulled_2(e,0,0.)[:,0,0],label="j2")
    plt.legend()
    plt.subplot(3,3,3)
    plt.plot(e, j_pulled_3(e,0,0.)[:,0,0],label="j3")
    plt.legend()
    plt.subplot(3,3,4)
    plt.plot(e, J.spline(e,0.,0.)[0][:,0,0],label="projected j1")
    plt.legend()
    plt.subplot(3,3,5)
    plt.plot(e, J.spline(e,0.,0.)[1][:,0,0],label="projected j2")
    plt.legend()
    plt.subplot(3,3,6)
    plt.plot(e, J.spline(e,0.,0.)[2][:,0,0],label="projected j3")
    plt.legend()
    plt.subplot(3,3,7)
    plt.plot(e, J.spline(e,0.,0.)[0][:,0,0] - j_pulled_1(e,0,0.)[:,0,0],label="difference j1")
    plt.legend()
    plt.subplot(3,3,8)
    plt.plot(e, J.spline(e,0.,0.)[1][:,0,0] - j_pulled_2(e,0,0.)[:,0,0],label="difference j2")
    plt.legend()
    plt.subplot(3,3,9)
    plt.plot(e, J.spline(e,0.,0.)[2][:,0,0] - j_pulled_3(e,0,0.)[:,0,0],label="difference j3")
    # plt.show()
    # exit()

    plt.figure(5)
    # plt.title("Fourier transform of source term")
    plt.subplot(3,1,1)
    plt.plot(e_k/L, cellsize * np.fft.fftshift(np.abs(np.fft.fft(domain.push(J.spline,e,0.,0.,kind="1")[0][:,0,0]))),label="FFT of j1")
    plt.legend()
    plt.subplot(3,1,2)
    plt.plot(e_k/L, cellsize * np.fft.fftshift(np.abs(np.fft.fft(domain.push(J.spline,e,0.,0.,kind="1")[1][:,0,0]))),label="FFT of j2")
    plt.legend()
    plt.subplot(3,1,3)
    plt.plot(e_k/L, cellsize * np.fft.fftshift(np.abs(np.fft.fft(domain.push(J.spline,e,0.,0.,kind="1")[2][:,0,0]))),label="FFT of j3")
    plt.legend()
    # plt.show()
    # exit()

    solver_params = SolverParameters(
        tol=1e-10,
        maxiter=1600,
        info=True,
        recycle=True,
    )

    _rhosin = FEECVariable(space="H1")
    _rhosin.allocate(derham=derham, domain=domain)

    _rhocos = FEECVariable(space="H1")
    _rhocos.allocate(derham=derham, domain=domain)

    _usin = FEECVariable(space="Hcurl")
    _usin.allocate(derham=derham, domain=domain)

    _ucos = FEECVariable(space="Hcurl")
    _ucos.allocate(derham=derham, domain=domain)

    _Esin = FEECVariable(space="Hcurl")
    _Esin.allocate(derham=derham, domain=domain)

    _Ecos = FEECVariable(space="Hcurl")
    _Ecos.allocate(derham=derham, domain=domain)

    _Bsin = FEECVariable(space="Hdiv")
    _Bsin.allocate(derham=derham, domain=domain)

    _Bcos = FEECVariable(space="Hdiv")
    _Bcos.allocate(derham=derham, domain=domain)

    solver = ColdPlasmaPerturbation()
    solver.variables.rhosin = _rhosin
    solver.variables.rhocos = _rhocos
    solver.variables.usin = _usin
    solver.variables.ucos = _ucos
    solver.variables.Esin = _Esin
    solver.variables.Ecos = _Ecos
    solver.variables.Bsin = _Bsin
    solver.variables.Bcos = _Bcos

    solver.options = solver.Options(
        J=J,
        omega=omega,
        c0=c0,
        c1=c1,
        mass=mass,
        mu=0.,
        nu=0.,
        rhobar=rhobar,
        theta=theta,
        Ebar=[zeroes, zeroes, zeroes],
        solver="gmres",
        solver_params=solver_params,
    )

    solver.allocate()

    dt=1.0
    print("Hi man")
    solver(dt)
    print("Bye man")

    Esinvalues = domain.push(_Esin.spline, e, 0., 0., kind="1")
    Ecosvalues = domain.push(_Ecos.spline, e, 0., 0., kind="1")

    usinvalues = domain.push(_usin.spline, e, 0., 0., kind="1")
    ucosvalues = domain.push(_ucos.spline, e, 0., 0., kind="1")

    rhosinvalues = domain.push(_rhosin.spline, e, 0., 0., kind="0")
    rhocosvalues = domain.push(_rhocos.spline, e, 0., 0., kind="0")

    print(Esinvalues.shape)
    print(Ecosvalues.shape)

    print(usinvalues.shape)
    print(ucosvalues.shape)

    print(rhosinvalues.shape)
    print(rhocosvalues.shape)

    Esinvalues1 = Esinvalues[0,:,0,0]
    Esinvalues2 = Esinvalues[1,:,0,0]
    Esinvalues3 = Esinvalues[2,:,0,0]

    Ecosvalues1 = Esinvalues[0,:,0,0]
    Ecosvalues2 = Esinvalues[1,:,0,0]
    Ecosvalues3 = Esinvalues[2,:,0,0]

    usinvalues1 = usinvalues[0,:,0,0]
    usinvalues2 = usinvalues[1,:,0,0]
    usinvalues3 = usinvalues[2,:,0,0]

    ucosvalues1 = usinvalues[0,:,0,0]
    ucosvalues2 = usinvalues[1,:,0,0]
    ucosvalues3 = usinvalues[2,:,0,0]

    Esinvalues1_fft = np.fft.fftshift(np.fft.fft(Esinvalues1)) / cellsize
    print("Evaluated FFT of Esin1")
    Esinvalues2_fft = np.fft.fftshift(np.fft.fft(Esinvalues2)) / cellsize
    print("Evaluated FFT of Esin2")
    Esinvalues3_fft = np.fft.fftshift(np.fft.fft(Esinvalues3)) / cellsize
    print("Evaluated FFT of Esin3")

    Ecosvalues1_fft = np.fft.fftshift(np.fft.fft(Ecosvalues1)) / cellsize
    print("Evaluated FFT of Ecos1")
    Ecosvalues2_fft = np.fft.fftshift(np.fft.fft(Ecosvalues2)) / cellsize
    print("Evaluated FFT of Ecos2")
    Ecosvalues3_fft = np.fft.fftshift(np.fft.fft(Ecosvalues3)) / cellsize
    print("Evaluated FFT of Ecos3")

    usinvalues1_fft = np.fft.fftshift(np.fft.fft(usinvalues1)) / cellsize
    print("Evaluated FFT of usin1")
    usinvalues2_fft = np.fft.fftshift(np.fft.fft(usinvalues2)) / cellsize
    print("Evaluated FFT of usin2")
    usinvalues3_fft = np.fft.fftshift(np.fft.fft(usinvalues3)) / cellsize
    print("Evaluated FFT of usin3")

    ucosvalues1_fft = np.fft.fftshift(np.fft.fft(ucosvalues1)) / cellsize
    print("Evaluated FFT of ucos1")
    ucosvalues2_fft = np.fft.fftshift(np.fft.fft(ucosvalues2)) / cellsize
    print("Evaluated FFT of ucos2")
    ucosvalues3_fft = np.fft.fftshift(np.fft.fft(ucosvalues3)) / cellsize
    print("Evaluated FFT of ucos3")

    E_abs = np.sqrt(Esinvalues1 * np.conjugate(Esinvalues1) + Esinvalues2 * np.conjugate(Esinvalues2) + Esinvalues3 * np.conjugate(Esinvalues3) \
        + Ecosvalues1 * np.conjugate(Ecosvalues1) + Ecosvalues2 * np.conjugate(Ecosvalues2) + Ecosvalues3 * np.conjugate(Ecosvalues3))

    E_abs_fft = cellsize * np.sqrt(Esinvalues1_fft * np.conjugate(Esinvalues1_fft) + Esinvalues2_fft * np.conjugate(Esinvalues2_fft) + Esinvalues3_fft * np.conjugate(Esinvalues3_fft) \
        + Ecosvalues1_fft * np.conjugate(Ecosvalues1_fft) + Ecosvalues2_fft * np.conjugate(Ecosvalues2_fft) + Ecosvalues3_fft * np.conjugate(Ecosvalues3_fft))
    print("Evaluated square modulus of FFT of E")
    print(np.max(E_abs_fft))

    u_abs = np.sqrt(usinvalues1 * np.conjugate(usinvalues1) + usinvalues2 * np.conjugate(usinvalues2) + usinvalues3 * np.conjugate(usinvalues3) \
        + ucosvalues1 * np.conjugate(ucosvalues1) + ucosvalues2 * np.conjugate(ucosvalues2) + ucosvalues3 * np.conjugate(ucosvalues3))

    u_abs_fft = cellsize * np.sqrt(usinvalues1_fft * np.conjugate(usinvalues1_fft) + usinvalues2_fft * np.conjugate(usinvalues2_fft) + usinvalues3_fft * np.conjugate(usinvalues3_fft) \
        + ucosvalues1_fft * np.conjugate(ucosvalues1_fft) + ucosvalues2_fft * np.conjugate(ucosvalues2_fft) + ucosvalues3_fft * np.conjugate(ucosvalues3_fft))
    print("Evaluated square modulus of FFT of u")
    print(np.max(u_abs))
    print(np.max(u_abs_fft))

    # ax = plt.figure().add_subplot(projection='3d')
    # ax.plot(k,k,E_abs)

    print(f"{d_omega=}")
    print(f"{k_light/L=}")
    # print(f"{k_sound/L=}")

    plt.figure(6)
    plotyscale = np.max(u_abs_fft*V)
    plotxscale = dk * Nel / L
    plt.axhline(y=0.,color="black")
    plt.axvline(x=k_light/L, ymin=0.05, linestyle='--',color="black")
    plt.axvline(x=-k_light/L, ymin=0.05, linestyle='--',color="black")
    # plt.axvline(x=k_sound/L, ymin=0.05, linestyle='--',color="black")
    # plt.axvline(x=-k_sound/L, ymin=0.05, linestyle='--',color="black")
    plt.plot(e_k/L,u_abs_fft*V,label="u")
    plt.xlabel("Wave vectors [$m^{-1}$]")
    plt.text(k_light/L - 0.02*plotxscale,-0.035*plotyscale,f"+$k_{{\\mathrm{{light}}}}$")
    plt.text(-k_light/L - 0.02*plotxscale,-0.035*plotyscale,f"-$k_{{\\mathrm{{light}}}}$")
    # plt.text(k_sound/L - 0.02*plotxscale,-0.035*plotyscale,f"+$k_{{\\mathrm{{thermic}}}}$")
    # plt.text(-k_sound/L - 0.02*plotxscale,-0.035*plotyscale,f"-$k_{{\\mathrm{{thermic}}}}$")
    plt.ylabel("Velocity field magnitude")
    plt.title(f"Fourier Transform of u with $\\bar{{\\rho}}$={(rhobar*m_e*N*1000):.1f} $g cm^{{-3}}$, $\\bar{{\\theta}}$={(theta*T):.0f} K, \
        $k_{{\\mathrm{{light}}}}$={k_light/L:.3f} $m^{{-1}}$")
    plt.legend()
    # $k_{{\\mathrm{{thermic}}}}$={k_sound/L:.3f} $m^{{-1}}$

    plt.figure(7)
    plotyscale = np.max(E_abs_fft*E)
    plotxscale = dk * Nel / L
    plt.axhline(y=0.,color="black")
    plt.axvline(x=k_light/L, ymin=0.05, linestyle='--',color="black")
    plt.axvline(x=-k_light/L, ymin=0.05, linestyle='--',color="black")
    # plt.axvline(x=k_sound/L, ymin=0.05, linestyle='--',color="black")
    # plt.axvline(x=-k_sound/L, ymin=0.05, linestyle='--',color="black")
    plt.plot(e_k/L,E_abs_fft*E,label="E")
    plt.xlabel("Wave vectors [$m^{-1}$]")
    plt.text(k_light/L - 0.02*plotxscale,-0.035*plotyscale,f"+$k_{{\\mathrm{{light}}}}$")
    plt.text(-k_light/L - 0.02*plotxscale,-0.035*plotyscale,f"-$k_{{\\mathrm{{light}}}}$")
    # plt.text(k_sound/L - 0.02*plotxscale,-0.035*plotyscale,f"+$k_{{\\mathrm{{thermic}}}}$")
    # plt.text(-k_sound/L - 0.02*plotxscale,-0.035*plotyscale,f"-$k_{{\\mathrm{{thermic}}}}$")
    plt.ylabel("Electric field strength")
    plt.title(f"Fourier Transform of E with $\\bar{{\\rho}}$={(rhobar*m_e*N*1000):.1f} $g cm^{{-3}}$, $\\bar{{\\theta}}$={(theta*T):.0f} K, \
        $k_{{\\mathrm{{light}}}}$={k_light/L:.3f} $m^{{-1}}$")
    plt.legend()
    # $k_{{\\mathrm{{thermic}}}}$={k_sound/L:.3f} $m^{{-1}}$

    plt.show()


def test_dispersion_relation_2d():
    ksquared = lambda k1,k2: ((k1**2)+(k2**2))*id3x3

    ktensork = lambda k1,k2: np.array([[k1**2, k1*k2, 0],
                                        [k2*k1, k2**2, 0],
                                        [0, 0, 0]])

    rotB = np.array([[0, -B_z, B_y],
                    [B_z, 0, -B_x],
                    [-B_y, B_x, 0]])

    id3x3 = np.identity(3)

    def matrix(k1,k2):
        mat = np.zeros((6,6), dtype="complex")

        mat[:3,:3] = id3x3 - theta / (mass * (omega**2)) * ktensork(k1,k2) # - 1j * omega_cyclo / omega * rotB
        mat[:3,3:6] = 1j * omega_pe_normalized * id3x3
        mat[3:6,:3] = - 1j * omega_pe_normalized * id3x3
        mat[3:6,3:6] = (omega**2) * id3x3 - (c_normalized**2) * (ksquared(k1,k2) - ktensork(k1,k2))

        return mat

    determinant = lambda k1, k2: np.linalg.det(matrix(k1,k2))

    print(determinant(k_light,0)/(c_normalized**2))
    # print(determinant(k_sound,0)/(c_normalized**2))
    # print(determinant((k_light+k_sound)/2,0))

    if d_omega == 0.:
        kmax = 0.5
    else:
        kmax = 1.6 * k_light # np.maximum(k_light, k_sound)

    Nel = 200

    k = np.linspace(-kmax, kmax, Nel)

    k1, k2 = np.meshgrid(k, k, indexing='ij')

    det = np.zeros((Nel,Nel))

    for i in range(Nel):
        for j in range(Nel):
            val = np.real(determinant(k1[i,j],k2[i,j]))
            det[i,j] = 0. if np.abs(val) <= tol else val

    det /= c_normalized**2

    detslice = (det[int(Nel/2),:]+det[int(Nel/2)+1,:])/2

    # ax = plt.figure().add_subplot(projection='3d')

    # surface = ax.plot_surface(k1,k2,det,linewidth=0)

    # plt.contour(k1, k2, det, levels=[0.])

    # plt.colorbar()

    plt.figure(1)
    plotyscale = np.max(detslice)
    plotxscale = 2*kmax/L
    plt.title("Determinant of the system of oscillations")
    plt.plot(k/L, detslice, label="Horizontal slice of determinant")
    plt.axhline(y=0.,color="black")
    plt.axvline(x=k_light/L, ymin=0.05, linestyle='--',color="black")
    plt.axvline(x=-k_light/L, ymin=0.05, linestyle='--',color="black")
    # plt.axvline(x=k_sound/L, ymin=0.05, linestyle='--',color="black")
    # plt.axvline(x=-k_sound/L, ymin=0.05, linestyle='--',color="black")
    plt.xlabel("Wave vectors [$m^{-1}$]")
    plt.text(k_light/L - 0.02*plotxscale,-0.035*plotyscale,f"+$k_{{\\mathrm{{light}}}}$")
    plt.text(-k_light/L - 0.02*plotxscale,-0.035*plotyscale,f"-$k_{{\\mathrm{{light}}}}$")
    # plt.text(k_sound/L - 0.02*plotxscale,-0.035*plotyscale,f"+$k_{{\\mathrm{{therminc}}}}$")
    # plt.text(-k_sound/L - 0.02*plotxscale,-0.035*plotyscale,f"-$k_{{\\mathrm{{thermic}}}}$")
    plt.legend()


    p = 3
    Ngrid = 2**6

    Nfft = 64
    k_cutoff = kmax
    dk = k_cutoff / Nfft
    maxL: float = 2*np.pi / dk # so that our largest wavenumber value is included

    domain = domains.Cuboid(l1=-maxL/2 ,r1=maxL/2,l2=-maxL/2 ,r2=maxL/2)
    equil = equils.HomogenSlab(B0x=B_x, B0y=B_y, B0z=B_z)
    equil.domain = domain

    e = np.linspace(0., 1., Nel)
    e_x, e_y, e_z = domain(e,e,0.) # the values the field will be sampled on that will correspond exactly to the k array after the FFT
    e_x = e_x[:,:,0]
    e_y = e_y[:,:,0]
    print(e_x)
    print(e_y)
    cellsize = (maxL / Nel)**2
    e_k = np.linspace(-Nel/2 * dk, (Nel/2-1) * dk, Nel)

    j_physical = lambda x,y,z: k_cutoff/np.pi * np.sinc(k_cutoff/np.pi * x) * k_cutoff/np.pi * np.sinc(k_cutoff/np.pi * y)
    # j_y = lambda x,y,z: np.exp(-((x-0.5)**2)/(0.1**2))
    # j_physical = lambda x,y,z: np.sinc(64/maxL *x)
    print(f"{k_cutoff/np.pi *maxL=}")
    zeroes = lambda x,y,z: 0. * (x+y+z)

    j_pulled_1 = lambda e1,e2,e3: domain.pull([j_physical,j_physical,j_physical],e1,e2,e3,kind="1", squeeze_out=False)[0]
    j_pulled_2 = lambda e1,e2,e3: domain.pull([j_physical,j_physical,j_physical],e1,e2,e3,kind="1", squeeze_out=False)[1]
    j_pulled_3 = lambda e1,e2,e3: domain.pull([j_physical,j_physical,j_physical],e1,e2,e3,kind="1", squeeze_out=False)[2]

    print(f"{np.shape(e)=}")
    print(f"{np.shape(e_x)=}")
    print(f"{np.shape(e_y)=}")
    print(f"{np.shape(j_pulled_1(e,e,0.)[:,:,0])=}")
    print(f"{np.shape(j_physical(e_x,e_y,0.))=}")

    jdiff1 = lambda e1,e2,e3: j_pulled_1(e1,e2,e3) / maxL - j_physical(domain(e1,e2,e3)[0],domain(e1,e2,e3)[1],domain(e1,e2,e3)[2])
    jdiff2 = lambda e1,e2,e3: j_pulled_2(e1,e2,e3) / maxL - j_physical(domain(e1,e2,e3)[0],domain(e1,e2,e3)[1],domain(e1,e2,e3)[2])
    jdiff3 = lambda e1,e2,e3: j_pulled_3(e1,e2,e3) / maxL - j_physical(domain(e1,e2,e3)[0],domain(e1,e2,e3)[1],domain(e1,e2,e3)[2])

    plt.figure(2)
    plt.subplot(1,3,1)
    plt.pcolormesh(e,e, jdiff1(e,e,0.)[:,:,0],label="pulled j1")
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.pcolormesh(e,e, jdiff2(e,e,0.)[:,:,0],label="pulled j2")
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.pcolormesh(e,e, jdiff3(e,e,0.)[:,:,0],label="pulled j3")
    plt.colorbar()

    plt.figure(3)
    plt.subplot(2,3,1)
    plt.pcolormesh(e,e, j_physical(e_x,e_y,0.),label="physical j1")
    plt.colorbar()
    plt.subplot(2,3,2)
    plt.pcolormesh(e,e, j_physical(e_x,e_y,0.),label="physical j2")
    plt.colorbar()
    plt.subplot(2,3,3)
    plt.pcolormesh(e,e, j_physical(e_x,e_y,0.),label="physical j3")
    plt.colorbar()
    plt.subplot(2,3,4)
    plt.pcolormesh(e,e, j_pulled_1(e,e,0.)[:,:,0],label="physical j1")
    plt.colorbar()
    plt.subplot(2,3,5)
    plt.pcolormesh(e,e, j_pulled_2(e,e,0.)[:,:,0],label="physical j2")
    plt.colorbar()
    plt.subplot(2,3,6)
    plt.pcolormesh(e,e, j_pulled_3(e,e,0.)[:,:,0],label="physical j3")
    plt.colorbar()
    # plt.show()
    # exit()

    degree = (p,p,1)
    num_elements = (Ngrid,Ngrid,1)
    bcs = (("dirichlet","dirichlet"), ("dirichlet","dirichlet"), None)

    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs)
    derham = Derham(grid=grid, options=derham_opts, comm=comm)
    projected_equil = ProjectedFluidEquilibriumWithB(equil=equil, derham=derham)

    mass_ops = WeightedMassOperators(derham=derham, domain=domain)
    basis_ops = BasisProjectionOperators(derham=derham, domain=domain)

    Propagator.derham = derham
    Propagator.domain = domain
    Propagator.mass_ops = mass_ops
    Propagator.basis_ops = basis_ops
    Propagator.projected_equil = projected_equil

    J = FEECVariable(space="Hcurl")
    J.allocate(derham=derham, domain=domain)
    J.spline.vector = derham.P1([j_pulled_1,j_pulled_2,j_pulled_3])

    ee1, ee2, ee3 = np.meshgrid(np.linspace(0.,1.,4),np.linspace(0.,1.,5),np.linspace(0.,1.,6), indexing="ij")
    # print(f"{np.shape(sincheck(ee1,ee2,ee3))=}")
    print(f"{np.shape(j_pulled_1(ee1,ee2,ee3))=}")
    print(f"{e=}")

    plt.figure(3)
    # plt.plot(e, J0.spline(e,0.,0.)[:,0,0],label="projected j1")
    # plt.plot(e, j_pulled_1(e,0.,0.)[:,0,0],'x',label="j1")
    plt.subplot(2,3,1)
    plt.pcolormesh(e,e, j_pulled_1(e,e,0.)[:,:,0],label="j1")
    plt.colorbar()
    plt.subplot(2,3,2)
    plt.pcolormesh(e,e, j_pulled_2(e,e,0.)[:,:,0],label="j2")
    plt.colorbar()
    plt.subplot(2,3,3)
    plt.pcolormesh(e,e, j_pulled_3(e,e,0.)[:,:,0],label="j3")
    plt.colorbar()
    plt.subplot(2,3,4)
    plt.pcolormesh(e,e, J.spline(e,e,0.)[0][:,:,0],label="projected j1")
    plt.colorbar()
    plt.subplot(2,3,5)
    plt.pcolormesh(e,e, J.spline(e,e,0.)[1][:,:,0],label="projected j2")
    plt.colorbar()
    plt.subplot(2,3,6)
    plt.pcolormesh(e,e, J.spline(e,e,0.)[2][:,:,0],label="projected j3")
    plt.colorbar()
    # plt.show()
    # exit()

    plt.figure(4)
    # plt.title("Fourier transform of source term")
    plt.subplot(3,1,1)
    plt.contour(e_k/L, e_k/L, cellsize * np.fft.fftshift(np.abs(np.fft.fft2(domain.push(J.spline,e,e,0.,kind="1")[0][:,:,0]))),levels=50)
    plt.colorbar()
    plt.subplot(3,1,2)
    plt.contour(e_k/L, e_k/L, cellsize * np.fft.fftshift(np.abs(np.fft.fft2(domain.push(J.spline,e,e,0.,kind="1")[1][:,:,0]))),levels=50)
    plt.colorbar()
    plt.subplot(3,1,3)
    plt.contour(e_k/L, e_k/L, cellsize * np.fft.fftshift(np.abs(np.fft.fft2(domain.push(J.spline,e,e,0.,kind="1")[2][:,:,0]))),levels=50)
    plt.colorbar()
    # plt.show()
    # exit()

    solver_params = SolverParameters(
        tol=1e-16,
        maxiter=3000,
        info=True,
        recycle=True,
    )

    _rhosin = FEECVariable(space="H1")
    _rhosin.allocate(derham=derham, domain=domain)

    _rhocos = FEECVariable(space="H1")
    _rhocos.allocate(derham=derham, domain=domain)

    _usin = FEECVariable(space="Hcurl")
    _usin.allocate(derham=derham, domain=domain)

    _ucos = FEECVariable(space="Hcurl")
    _ucos.allocate(derham=derham, domain=domain)

    _Esin = FEECVariable(space="Hcurl")
    _Esin.allocate(derham=derham, domain=domain)

    _Ecos = FEECVariable(space="Hcurl")
    _Ecos.allocate(derham=derham, domain=domain)

    _Bsin = FEECVariable(space="Hdiv")
    _Bsin.allocate(derham=derham, domain=domain)

    _Bcos = FEECVariable(space="Hdiv")
    _Bcos.allocate(derham=derham, domain=domain)

    solver = ColdPlasmaPerturbation()
    solver.variables.rhosin = _rhosin
    solver.variables.rhocos = _rhocos
    solver.variables.usin = _usin
    solver.variables.ucos = _ucos
    solver.variables.Esin = _Esin
    solver.variables.Ecos = _Ecos
    solver.variables.Bsin = _Bsin
    solver.variables.Bcos = _Bcos

    solver.options = solver.Options(
        J=J,
        omega=omega,
        c0=c0,
        c1=c1,
        mass=mass,
        mu=0.,
        nu=0.,
        rhobar=rhobar,
        theta=theta,
        Ebar=[zeroes, zeroes, zeroes],
        solver="gmres",
        solver_params=solver_params,
    )

    solver.allocate()

    dt=1.0
    print("Hi man")
    solver(dt)
    print("Bye man")

    Esinvalues = domain.push(_Esin.spline, e, e, 0., kind="1")
    Ecosvalues = domain.push(_Ecos.spline, e, e, 0., kind="1")

    usinvalues = domain.push(_usin.spline, e, e, 0., kind="1")
    ucosvalues = domain.push(_ucos.spline, e, e, 0., kind="1")

    rhosinvalues = domain.push(_rhosin.spline, e, e, 0., kind="0")
    rhocosvalues = domain.push(_rhocos.spline, e, e, 0., kind="0")

    print(Esinvalues.shape)
    print(Ecosvalues.shape)

    print(usinvalues.shape)
    print(ucosvalues.shape)

    print(rhosinvalues.shape)
    print(rhocosvalues.shape)

    Esinvalues1 = Esinvalues[0,:,:,0]
    Esinvalues2 = Esinvalues[1,:,:,0]
    Esinvalues3 = Esinvalues[2,:,:,0]

    Ecosvalues1 = Esinvalues[0,:,:,0]
    Ecosvalues2 = Esinvalues[1,:,:,0]
    Ecosvalues3 = Esinvalues[2,:,:,0]

    usinvalues1 = usinvalues[0,:,:,0]
    usinvalues2 = usinvalues[1,:,:,0]
    usinvalues3 = usinvalues[2,:,:,0]

    ucosvalues1 = usinvalues[0,:,:,0]
    ucosvalues2 = usinvalues[1,:,:,0]
    ucosvalues3 = usinvalues[2,:,:,0]

    Esinvalues1_fft = np.fft.fftshift(np.fft.fft2(Esinvalues1)) / cellsize
    print("Evaluated FFT of Esin1")
    Esinvalues2_fft = np.fft.fftshift(np.fft.fft2(Esinvalues2)) / cellsize
    print("Evaluated FFT of Esin2")
    Esinvalues3_fft = np.fft.fftshift(np.fft.fft2(Esinvalues3)) / cellsize
    print("Evaluated FFT of Esin3")

    Ecosvalues1_fft = np.fft.fftshift(np.fft.fft2(Ecosvalues1)) / cellsize
    print("Evaluated FFT of Ecos1")
    Ecosvalues2_fft = np.fft.fftshift(np.fft.fft2(Ecosvalues2)) / cellsize
    print("Evaluated FFT of Ecos2")
    Ecosvalues3_fft = np.fft.fftshift(np.fft.fft2(Ecosvalues3)) / cellsize
    print("Evaluated FFT of Ecos3")

    usinvalues1_fft = np.fft.fftshift(np.fft.fft2(usinvalues1)) / cellsize
    print("Evaluated FFT of usin1")
    usinvalues2_fft = np.fft.fftshift(np.fft.fft2(usinvalues2)) / cellsize
    print("Evaluated FFT of usin2")
    usinvalues3_fft = np.fft.fftshift(np.fft.fft2(usinvalues3)) / cellsize
    print("Evaluated FFT of usin3")

    ucosvalues1_fft = np.fft.fftshift(np.fft.fft2(ucosvalues1)) / cellsize
    print("Evaluated FFT of ucos1")
    ucosvalues2_fft = np.fft.fftshift(np.fft.fft2(ucosvalues2)) / cellsize
    print("Evaluated FFT of ucos2")
    ucosvalues3_fft = np.fft.fftshift(np.fft.fft2(ucosvalues3)) / cellsize
    print("Evaluated FFT of ucos3")

    E_abs = np.sqrt(Esinvalues1 * np.conjugate(Esinvalues1) + Esinvalues2 * np.conjugate(Esinvalues2) + Esinvalues3 * np.conjugate(Esinvalues3) \
        + Ecosvalues1 * np.conjugate(Ecosvalues1) + Ecosvalues2 * np.conjugate(Ecosvalues2) + Ecosvalues3 * np.conjugate(Ecosvalues3))

    E_abs_fft = cellsize * np.sqrt(Esinvalues1_fft * np.conjugate(Esinvalues1_fft) + Esinvalues2_fft * np.conjugate(Esinvalues2_fft) + Esinvalues3_fft * np.conjugate(Esinvalues3_fft) \
        + Ecosvalues1_fft * np.conjugate(Ecosvalues1_fft) + Ecosvalues2_fft * np.conjugate(Ecosvalues2_fft) + Ecosvalues3_fft * np.conjugate(Ecosvalues3_fft))
    print("Evaluated square modulus of FFT of E")
    print(np.max(E_abs_fft))

    u_abs = np.sqrt(usinvalues1 * np.conjugate(usinvalues1) + usinvalues2 * np.conjugate(usinvalues2) + usinvalues3 * np.conjugate(usinvalues3) \
        + ucosvalues1 * np.conjugate(ucosvalues1) + ucosvalues2 * np.conjugate(ucosvalues2) + ucosvalues3 * np.conjugate(ucosvalues3))

    u_abs_fft = cellsize * np.sqrt(usinvalues1_fft * np.conjugate(usinvalues1_fft) + usinvalues2_fft * np.conjugate(usinvalues2_fft) + usinvalues3_fft * np.conjugate(usinvalues3_fft) \
        + ucosvalues1_fft * np.conjugate(ucosvalues1_fft) + ucosvalues2_fft * np.conjugate(ucosvalues2_fft) + ucosvalues3_fft * np.conjugate(ucosvalues3_fft))
    print("Evaluated square modulus of FFT of u")
    print(np.max(u_abs))
    print(np.max(u_abs_fft))

    # ax = plt.figure().add_subplot(projection='3d')
    # ax.plot(k,k,E_abs)

    print(f"{d_omega=}")
    print(f"{k_light/L=}")
    # print(f"{k_sound/L=}")

    plt.figure(5)
    plotyscale = np.max(u_abs_fft*V)
    plotxscale = dk * Nel / L
    plt.axhline(y=0.,color="black")
    plt.axvline(x=k_light/L, ymin=0.05, linestyle='--',color="black")
    plt.axvline(x=-k_light/L, ymin=0.05, linestyle='--',color="black")
    # plt.axvline(x=k_sound/L, ymin=0.05, linestyle='--',color="black")
    # plt.axvline(x=-k_sound/L, ymin=0.05, linestyle='--',color="black")
    plt.contour(e_k/L,e_k/L,u_abs_fft*V,levels=100,label="u")
    plt.xlabel("Wave vectors [$m^{-1}$]")
    plt.text(k_light/L - 0.02*plotxscale,-0.035*plotyscale,f"+$k_{{\\mathrm{{light}}}}$")
    plt.text(-k_light/L - 0.02*plotxscale,-0.035*plotyscale,f"-$k_{{\\mathrm{{light}}}}$")
    # plt.text(k_sound/L - 0.02*plotxscale,-0.035*plotyscale,f"+$k_{{\\mathrm{{therminc}}}}$")
    # plt.text(-k_sound/L - 0.02*plotxscale,-0.035*plotyscale,f"-$k_{{\\mathrm{{thermic}}}}$")
    plt.ylabel("Velocity field magnitude")
    plt.title(f"Fourier Transform of u with $\\bar{{\\rho}}$={(rhobar*m_e*N*1000):.1f} $g cm^{{-3}}$, $\\bar{{\\theta}}$={(theta*T):.0f} K, \
        $k_{{\\mathrm{{light}}}}$={k_light/L:.3f} $m^{{-1}}$")
    plt.legend()
    plt.colorbar()
    # $k_{{\\mathrm{{thermic}}}}$={k_sound/L:.3f} $m^{{-1}}$

    plt.figure(6)
    plotyscale = np.max(E_abs_fft*E)
    plotxscale = dk * Nel / L
    plt.axhline(y=0.,color="black")
    plt.axvline(x=k_light/L, ymin=0.05, linestyle='--',color="black")
    plt.axvline(x=-k_light/L, ymin=0.05, linestyle='--',color="black")
    # plt.axvline(x=k_sound/L, ymin=0.05, linestyle='--',color="black")
    # plt.axvline(x=-k_sound/L, ymin=0.05, linestyle='--',color="black")
    plt.contour(e_k/L,e_k/L,E_abs_fft*E,levels=100,label="E")
    plt.xlabel("Wave vectors [$m^{-1}$]")
    plt.text(k_light/L - 0.02*plotxscale,-0.035*plotyscale,f"+$k_{{\\mathrm{{light}}}}$")
    plt.text(-k_light/L - 0.02*plotxscale,-0.035*plotyscale,f"-$k_{{\\mathrm{{light}}}}$")
    # plt.text(k_sound/L - 0.02*plotxscale,-0.035*plotyscale,f"+$k_{{\\mathrm{{therminc}}}}$")
    # plt.text(-k_sound/L - 0.02*plotxscale,-0.035*plotyscale,f"-$k_{{\\mathrm{{thermic}}}}$")
    plt.ylabel("Electric field strength")
    plt.title(f"Fourier Transform of E with $\\bar{{\\rho}}$={(rhobar*m_e*N*1000):.1f} $g cm^{{-3}}$, $\\bar{{\\theta}}$={(theta*T):.0f} K, \
        $k_{{\\mathrm{{light}}}}$={k_light/L:.3f} $m^{{-1}}$")
    plt.legend()
    plt.colorbar()
    # $k_{{\\mathrm{{thermic}}}}$={k_sound/L:.3f} $m^{{-1}}$

    plt.show()




def test_convergence_1d(
    show_plot: bool = False,
):
    """Test of the solver on 1d problem by means of manufactured solution"""

    domain: Domain = domains.Cuboid()
    equil = equils.HomogenSlab(B0x=B_x, B0y=B_y, B0z=B_z)
    equil.domain = domain

    bcs = (None, None, None)

    pmax = 3
    Nmin = 4
    Nmax = 7

    J0: float = 1.5 # times eNV
    
    denom_light: float = 1. / (omega**2 - rhobar / (alpha * (mass**2)) - (c_normalized**2) * 16*(xp.pi**2))
    denom_sound: float = 1. / (omega**2 - rhobar / (alpha * (mass**2)) - (theta / mass) * 36*(xp.pi**2))

    E0: float = omega * J0 * denom_light / alpha
    u0: float = J0 * denom_sound / (mass * alpha)

    zeroes = lambda x,y,z: 0.*(x+y+z)
    
    j_exact_x = lambda x,y,z: J0 * xp.cos(6*xp.pi*x)
    j_exact_y = lambda x,y,z: J0 * xp.sin(4*xp.pi*x)

    E_exact_x = lambda x,y,z: - (1 + rhobar / (alpha * (mass**2)) * denom_sound) * J0 / (omega * alpha) * xp.cos(6*xp.pi*x)
    E_exact_y = lambda x,y,z: - E0 * xp.sin(4*xp.pi*x)

    u_exact_x = lambda x,y,z: - u0 * xp.cos(6*xp.pi*x)
    u_exact_y = lambda x,y,z: - E0 / (mass * omega) * xp.sin(4*xp.pi*x)

    rho_exact = lambda x,y,z: - 6*xp.pi * rhobar * u0 / omega * xp.sin(6*xp.pi*x)

    B_exact_z = lambda x,y,z: - 4*xp.pi / omega * E0 * xp.cos(4*xp.pi*x)


    # Test over spline degree and grid resolution
    Nels = [2**n for n in range(Nmin, Nmax + 1)]

    e1 = xp.linspace(0.0, 1.0, 64)
    e2 = 0.0
    e3 = 0.0

    ee1, ee2, ee3 = xp.meshgrid(e1, e2, e3, indexing="ij")

    for p in range(2, pmax + 1):
        errors_Esin = []
        errors_Ecos = []
        errors_usin = []
        errors_ucos = []
        errors_rhosin = []
        errors_rhocos = []
        errors_Bsin = []
        errors_Bcos = []
        errors = []
        h_vec = []

        for n, Nel in enumerate(Nels):

            degree = (p, 1, 1)
            num_elements = (Nel, 1, 1)

            grid = TensorProductGrid(num_elements=num_elements)
            derham_opts = DerhamOptions(degree=degree, bcs=bcs)
            derham = Derham(grid=grid, options=derham_opts, comm=comm)
            projected_equil = ProjectedFluidEquilibriumWithB(equil=equil, derham=derham)

            mass_ops = WeightedMassOperators(derham=derham, domain=domain)
            basis_ops = BasisProjectionOperators(derham=derham, domain=domain)

            Propagator.derham = derham
            Propagator.domain = domain
            Propagator.mass_ops = mass_ops
            Propagator.basis_ops = basis_ops
            Propagator.projected_equil = projected_equil

            J = FEECVariable(space="Hcurl")
            J.allocate(derham=derham, domain=domain)
            J.spline.vector = derham.P1([j_exact_x, j_exact_y, zeroes])

            solver_params = SolverParameters(
                tol=1e-16,
                maxiter=3000,
                info=True,
                recycle=False,
            )

            _Esin = FEECVariable(space="Hcurl")
            _Esin.allocate(derham=derham, domain=domain)
            _Ecos = FEECVariable(space="Hcurl")
            _Ecos.allocate(derham=derham, domain=domain)
            _usin = FEECVariable(space="Hcurl")
            _usin.allocate(derham=derham, domain=domain)
            _ucos = FEECVariable(space="Hcurl")
            _ucos.allocate(derham=derham, domain=domain)
            _rhosin = FEECVariable(space="H1")
            _rhosin.allocate(derham=derham, domain=domain)
            _rhocos = FEECVariable(space="H1")
            _rhocos.allocate(derham=derham, domain=domain)
            _Bsin = FEECVariable(space="Hdiv")
            _Bsin.allocate(derham=derham, domain=domain)
            _Bcos = FEECVariable(space="Hdiv")
            _Bcos.allocate(derham=derham, domain=domain)

            solver = ColdPlasmaPerturbation()
            solver.variables.Esin = _Esin
            solver.variables.Ecos = _Ecos
            solver.variables.usin = _usin
            solver.variables.ucos = _ucos
            solver.variables.rhosin = _rhosin
            solver.variables.rhocos = _rhocos
            solver.variables.Bsin = _Bsin
            solver.variables.Bcos = _Bcos

            solver.options = solver.Options(
                J=J,
                omega=omega,
                c0=c0,
                c1=c1,
                mass=mass,
                mu=0.,
                nu=0.,
                rhobar=rhobar,
                theta=theta,
                Ebar=[zeroes, zeroes, zeroes],
                solver="gmres",
                solver_params=solver_params,
            )

            solver.allocate()

            dt = 1.0
            solver(dt)

            Esin_calculated = xp.array(_Esin.spline(ee1, ee2, ee3))
            logger.info(f"{Esin_calculated.shape = }")
            Esin_analytical = xp.array([E_exact_x(ee1, ee2, ee3), E_exact_y(ee1, ee2, ee3), zeroes(ee1, ee2, ee3)])
            logger.info(f"{Esin_analytical.shape = }")

            Ecos_calculated = xp.array(_Ecos.spline(ee1, ee2, ee3))
            logger.info(f"{Ecos_calculated.shape = }")
            Ecos_analytical = xp.array([zeroes(ee1, ee2, ee3), zeroes(ee1, ee2, ee3), zeroes(ee1, ee2, ee3)])
            logger.info(f"{Ecos_analytical.shape = }")

            usin_calculated = xp.array(_usin.spline(ee1, ee2, ee3))
            logger.info(f"{usin_calculated.shape = }")
            usin_analytical = xp.array([zeroes(ee1, ee2, ee3), zeroes(ee1, ee2, ee3), zeroes(ee1, ee2, ee3)])
            logger.info(f"{usin_analytical.shape = }")

            ucos_calculated = xp.array(_ucos.spline(ee1, ee2, ee3))
            logger.info(f"{ucos_calculated.shape = }")
            print(_ucos.spline(1/xp.e,0,0))
            ucos_analytical = xp.array([u_exact_x(ee1, ee2, ee3), u_exact_y(ee1, ee2, ee3), zeroes(ee1, ee2, ee3)])
            logger.info(f"{ucos_analytical.shape = }")

            Bsin_calculated = xp.array(_Bsin.spline(ee1, ee2, ee3))
            logger.info(f"{Bsin_calculated.shape = }")
            Bsin_analytical = xp.array([zeroes(ee1, ee2, ee3), zeroes(ee1, ee2, ee3), zeroes(ee1, ee2, ee3)])
            logger.info(f"{Bsin_analytical.shape = }")

            Bcos_calculated = xp.array(_Bcos.spline(ee1, ee2, ee3))
            logger.info(f"{Bcos_calculated.shape = }")
            Bcos_analytical = xp.array([zeroes(ee1, ee2, ee3), zeroes(ee1, ee2, ee3), B_exact_z(ee1, ee2, ee3)])
            logger.info(f"{Bcos_analytical.shape = }")

            rhosin_calculated = xp.array(_rhosin.spline(ee1, ee2, ee3))
            logger.info(f"{rhosin_calculated.shape = }")
            rhosin_analytical = rho_exact(ee1, ee2, ee3)
            logger.info(f"{rhosin_analytical.shape = }")

            rhocos_calculated = xp.array(_rhocos.spline(ee1, ee2, ee3))
            logger.info(f"{rhocos_calculated.shape = }")
            rhocos_analytical = zeroes(ee1, ee2, ee3)
            logger.info(f"{rhocos_analytical.shape = }")

            if show_plot:
                plt.figure(f"Esin[0] error for degree {p =}, analytical amplitude {(1 + rhobar / (alpha * (mass**2)) * denom_sound) * J0 / (omega * alpha)}", figsize=(12, 8))
                plt.subplot(2, int((Nmax - Nmin) / 2 + 1), n + 1)
                plt.plot(e1, Esin_calculated[0][:, 0, 0], "o", label=f"{Nel}, numerical")
                plt.plot(e1, Esin_analytical[0][:, 0, 0], "k--", label=f"{Nel}, analytical")
                plt.legend()

                plt.figure(f"Esin[1] error for degree {p =}, analytical amplitude {E0}", figsize=(12, 8))
                plt.subplot(2, int((Nmax - Nmin) / 2 + 1), n + 1)
                plt.plot(e1, Esin_calculated[1][:, 0, 0], "o", label=f"{Nel}, numerical")
                plt.plot(e1, Esin_analytical[1][:, 0, 0], "k--", label=f"{Nel}, analytical")
                plt.legend()

                plt.figure(f"Esin[2] error for degree {p =}", figsize=(12, 8))
                plt.subplot(2, int((Nmax - Nmin) / 2 + 1), n + 1)
                plt.plot(e1, Esin_calculated[2][:, 0, 0], "o", label=f"{Nel}, numerical")
                plt.plot(e1, Esin_analytical[2][:, 0, 0], "k--", label=f"{Nel}, analytical")
                plt.legend()

                plt.figure(f"Ecos[0] error for degree {p =}", figsize=(12, 8))
                plt.subplot(2, int((Nmax - Nmin) / 2 + 1), n + 1)
                plt.plot(e1, Ecos_calculated[0][:, 0, 0], "o", label=f"{Nel}, numerical")
                plt.plot(e1, Ecos_analytical[0][:, 0, 0], "k--", label=f"{Nel}, analytical")
                plt.legend()

                plt.figure(f"Ecos[1] error for degree {p =}", figsize=(12, 8))
                plt.subplot(2, int((Nmax - Nmin) / 2 + 1), n + 1)
                plt.plot(e1, Ecos_calculated[1][:, 0, 0], "o", label=f"{Nel}, numerical")
                plt.plot(e1, Ecos_analytical[1][:, 0, 0], "k--", label=f"{Nel}, analytical")
                plt.legend()

                plt.figure(f"Ecos[2] error for degree {p =}", figsize=(12, 8))
                plt.subplot(2, int((Nmax - Nmin) / 2 + 1), n + 1)
                plt.plot(e1, Ecos_calculated[2][:, 0, 0], "o", label=f"{Nel}, numerical")
                plt.plot(e1, Ecos_analytical[2][:, 0, 0], "k--", label=f"{Nel}, analytical")
                plt.legend()

                plt.figure(f"usin[0] error for degree {p =}", figsize=(12, 8))
                plt.subplot(2, int((Nmax - Nmin) / 2 + 1), n + 1)
                plt.plot(e1, usin_calculated[0][:, 0, 0], "o", label=f"{Nel}, numerical")
                plt.plot(e1, usin_analytical[0][:, 0, 0], "k--", label=f"{Nel}, analytical")
                plt.legend()

                plt.figure(f"usin[1] error for degree {p =}", figsize=(12, 8))
                plt.subplot(2, int((Nmax - Nmin) / 2 + 1), n + 1)
                plt.plot(e1, usin_calculated[1][:, 0, 0], "o", label=f"{Nel}, numerical")
                plt.plot(e1, usin_analytical[1][:, 0, 0], "k--", label=f"{Nel}, analytical")
                plt.legend()

                plt.figure(f"usin[2] error for degree {p =}", figsize=(12, 8))
                plt.subplot(2, int((Nmax - Nmin) / 2 + 1), n + 1)
                plt.plot(e1, usin_calculated[2][:, 0, 0], "o", label=f"{Nel}, numerical")
                plt.plot(e1, usin_analytical[2][:, 0, 0], "k--", label=f"{Nel}, analytical")
                plt.legend()

                plt.figure(f"ucos[0] error for degree {p =}, analytical amplitude {u0}", figsize=(12, 8))
                plt.subplot(2, int((Nmax - Nmin) / 2 + 1), n + 1)
                plt.plot(e1, ucos_calculated[0][:, 0, 0], "o", label=f"{Nel}, numerical")
                plt.plot(e1, ucos_analytical[0][:, 0, 0], "k--", label=f"{Nel}, analytical")
                plt.legend()

                plt.figure(f"ucos[1] error for degree {p =}, analytical amplitude {E0 / (mass * omega)}", figsize=(12, 8))
                plt.subplot(2, int((Nmax - Nmin) / 2 + 1), n + 1)
                plt.plot(e1, ucos_calculated[1][:, 0, 0], "o", label=f"{Nel}, numerical")
                plt.plot(e1, ucos_analytical[1][:, 0, 0], "k--", label=f"{Nel}, analytical")
                plt.legend()

                plt.figure(f"ucos[2] error for degree {p =}", figsize=(12, 8))
                plt.subplot(2, int((Nmax - Nmin) / 2 + 1), n + 1)
                plt.plot(e1, ucos_calculated[2][:, 0, 0], "o", label=f"{Nel}, numerical")
                plt.plot(e1, ucos_analytical[2][:, 0, 0], "k--", label=f"{Nel}, analytical")
                plt.legend()

                plt.figure(f"rhosin error for degree {p =}, analytical amplitude {6*xp.pi * rhobar * u0 / omega}", figsize=(12, 8))
                plt.subplot(2, int((Nmax - Nmin) / 2 + 1), n + 1)
                plt.plot(e1, rhosin_calculated[:, 0, 0], "o", label=f"{Nel}, numerical")
                plt.plot(e1, rhosin_analytical[:, 0, 0], "k--", label=f"{Nel}, analytical")
                plt.legend()

                plt.figure(f"rhocos error for degree {p =}", figsize=(12, 8))
                plt.subplot(2, int((Nmax - Nmin) / 2 + 1), n + 1)
                plt.plot(e1, rhocos_calculated[:, 0, 0], "o", label=f"{Nel}, numerical")
                plt.plot(e1, rhocos_analytical[:, 0, 0], "k--", label=f"{Nel}, analytical")
                plt.legend()

                plt.figure(f"Bsin[0] error for degree {p =}", figsize=(12, 8))
                plt.subplot(2, int((Nmax - Nmin) / 2 + 1), n + 1)
                plt.plot(e1, Bsin_calculated[0][:, 0, 0], "o", label=f"{Nel}, numerical")
                plt.plot(e1, Bsin_analytical[0][:, 0, 0], "k--", label=f"{Nel}, analytical")
                plt.legend()

                plt.figure(f"Bsin[1] error for degree {p =}", figsize=(12, 8))
                plt.subplot(2, int((Nmax - Nmin) / 2 + 1), n + 1)
                plt.plot(e1, Bsin_calculated[1][:, 0, 0], "o", label=f"{Nel}, numerical")
                plt.plot(e1, Bsin_analytical[1][:, 0, 0], "k--", label=f"{Nel}, analytical")
                plt.legend()

                plt.figure(f"Bsin[2] error for degree {p =}", figsize=(12, 8))
                plt.subplot(2, int((Nmax - Nmin) / 2 + 1), n + 1)
                plt.plot(e1, Bsin_calculated[2][:, 0, 0], "o", label=f"{Nel}, numerical")
                plt.plot(e1, Bsin_analytical[2][:, 0, 0], "k--", label=f"{Nel}, analytical")
                plt.legend()

                plt.figure(f"Bcos[0] error for degree {p =}", figsize=(12, 8))
                plt.subplot(2, int((Nmax - Nmin) / 2 + 1), n + 1)
                plt.plot(e1, Bcos_calculated[0][:, 0, 0], "o", label=f"{Nel}, numerical")
                plt.plot(e1, Bcos_analytical[0][:, 0, 0], "k--", label=f"{Nel}, analytical")
                plt.legend()

                plt.figure(f"Bcos[1] error for degree {p =}", figsize=(12, 8))
                plt.subplot(2, int((Nmax - Nmin) / 2 + 1), n + 1)
                plt.plot(e1, Bcos_calculated[1][:, 0, 0], "o", label=f"{Nel}, numerical")
                plt.plot(e1, Bcos_analytical[1][:, 0, 0], "k--", label=f"{Nel}, analytical")
                plt.legend()

                plt.figure(f"Bcos[2] error for degree {p =}, analytical amplitude {4*xp.pi / omega * E0}", figsize=(12, 8))
                plt.subplot(2, int((Nmax - Nmin) / 2 + 1), n + 1)
                plt.plot(e1, Bcos_calculated[2][:, 0, 0], "o", label=f"{Nel}, numerical")
                plt.plot(e1, Bcos_analytical[2][:, 0, 0], "k--", label=f"{Nel}, analytical")
                plt.legend()

                # if n == 0:
                #     plt.show()
                #     exit()

            error_Esin = xp.max(xp.abs(Esin_calculated - Esin_analytical))
            errors_Esin.append(error_Esin)
            error_Ecos = xp.max(xp.abs(Ecos_calculated - Ecos_analytical))
            errors_Ecos.append(error_Ecos)

            error_usin = xp.max(xp.abs(usin_calculated - usin_analytical))
            errors_usin.append(error_usin)
            error_ucos = xp.max(xp.abs(ucos_calculated - ucos_analytical))
            errors_ucos.append(error_ucos)

            error_rhosin = xp.max(xp.abs(rhosin_calculated - rhosin_analytical))
            errors_rhosin.append(error_rhosin)
            error_rhocos = xp.max(xp.abs(rhocos_calculated - rhocos_analytical))
            errors_rhocos.append(error_rhocos)

            error_Bsin = xp.max(xp.abs(Bsin_calculated - Bsin_analytical))
            errors_Bsin.append(error_Bsin)
            error_Bcos = xp.max(xp.abs(Bcos_calculated - Bcos_analytical))
            errors_Bcos.append(error_Ecos)

            error = xp.max([error_Esin,error_Ecos,error_usin,error_ucos,error_rhosin,error_rhocos,error_Bsin,error_Bcos])
            errors.append(error)

            h = 1 / Nel
            h_vec.append(h)

        m, _ = xp.polyfit(xp.log(Nels), xp.log(errors), deg=1)
        logger.info(f"For {p =}, solution converges with rate {-m =} ")

        if show_plot:
            plt.figure(f"Esin Convergence for degree {p =}", figsize=(12, 8))
            plt.title(f"Esin Convergence rate for degree {p =}")
            plt.plot(h_vec, errors_Esin, "o", label=f"Calculated Esin error, {m =}")
            plt.plot(
                h_vec,
                [(h ** (p)) / (h_vec[0] ** (p)) * errors[0] for h in h_vec],
                "k--",
                label="Theoretical Esin error, rate = p",
            )
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Grid spacing h")
            plt.ylabel("Error")
            plt.legend()

            plt.figure(f"Ecos Convergence for degree {p =}", figsize=(12, 8))
            plt.title(f"Ecos Convergence rate for degree {p =}")
            plt.plot(h_vec, errors_Ecos, "o", label=f"Calculated Ecos error, {m =}")
            plt.plot(
                h_vec,
                [(h ** (p)) / (h_vec[0] ** (p)) * errors[0] for h in h_vec],
                "k--",
                label="Theoretical Ecos error, rate = p",
            )
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Grid spacing h")
            plt.ylabel("Error")
            plt.legend()

            plt.figure(f"usin Convergence for degree {p =}", figsize=(12, 8))
            plt.title(f"usin Convergence rate for degree {p =}")
            plt.plot(h_vec, errors_usin, "o", label=f"Calculated usin error, {m =}")
            plt.plot(
                h_vec,
                [(h ** (p)) / (h_vec[0] ** (p)) * errors[0] for h in h_vec],
                "k--",
                label="Theoretical usin error, rate = p",
            )
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Grid spacing h")
            plt.ylabel("Error")
            plt.legend()

            plt.figure(f"ucos Convergence for degree {p =}", figsize=(12, 8))
            plt.title(f"ucos Convergence rate for degree {p =}")
            plt.plot(h_vec, errors_Esin, "o", label=f"Calculated ucos error, {m =}")
            plt.plot(
                h_vec,
                [(h ** (p)) / (h_vec[0] ** (p)) * errors[0] for h in h_vec],
                "k--",
                label="Theoretical ucos error, rate = p",
            )
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Grid spacing h")
            plt.ylabel("Error")
            plt.legend()

            plt.figure(f"rhosin Convergence for degree {p =}", figsize=(12, 8))
            plt.title(f"rhosin Convergence rate for degree {p =}")
            plt.plot(h_vec, errors_rhosin, "o", label=f"Calculated rhosin error, {m =}")
            plt.plot(
                h_vec,
                [(h ** (p + 1)) / (h_vec[0] ** (p + 1)) * errors[0] for h in h_vec],
                "k--",
                label="Theoretical rhosin error, rate = p + 1",
            )
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Grid spacing h")
            plt.ylabel("Error")
            plt.legend()

            plt.figure(f"rhocos Convergence for degree {p =}", figsize=(12, 8))
            plt.title(f"rhocos Convergence rate for degree {p =}")
            plt.plot(h_vec, errors_rhocos, "o", label=f"Calculated rhocos error, {m =}")
            plt.plot(
                h_vec,
                [(h ** (p + 1)) / (h_vec[0] ** (p + 1)) * errors[0] for h in h_vec],
                "k--",
                label="Theoretical rhocos error, rate = p + 1",
            )
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Grid spacing h")
            plt.ylabel("Error")
            plt.legend()

            plt.figure(f"Bsin Convergence for degree {p =}", figsize=(12, 8))
            plt.title(f"Bsin Convergence rate for degree {p =}")
            plt.plot(h_vec, errors_Bsin, "o", label=f"Calculated Bsin error, {m =}")
            plt.plot(
                h_vec,
                [(h ** (p - 1)) / (h_vec[0] ** (p - 1)) * errors[0] for h in h_vec],
                "k--",
                label="Theoretical Esin error, rate = p - 1",
            )
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Grid spacing h")
            plt.ylabel("Error")
            plt.legend()

            plt.figure(f"Bcos Convergence for degree {p =}", figsize=(12, 8))
            plt.title(f"Bcos Convergence rate for degree {p =}")
            plt.plot(h_vec, errors_Bcos, "o", label=f"Calculated Bcos error, {m =}")
            plt.plot(
                h_vec,
                [(h ** (p - 1)) / (h_vec[0] ** (p - 1)) * errors[0] for h in h_vec],
                "k--",
                label="Theoretical Bcos error, rate = p - 1",
            )
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Grid spacing h")
            plt.ylabel("Error")
            plt.legend()

            plt.figure(f"Convergence for degree {p =}", figsize=(12, 8))
            plt.title(f"Convergence rate for degree {p =}")
            plt.plot(h_vec, errors, "o", label=f"Calculated error, {m =}")
            plt.plot(
                h_vec,
                [(h ** (p)) / (h_vec[0] ** (p)) * errors[0] for h in h_vec],
                "k--",
                label="Theoretical error, rate = p",
            )
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Grid spacing h")
            plt.ylabel("Error")
            plt.legend()

            plt.figure("Difference between Esin and ucos")
            plt.plot(e1, ucos_calculated[1][:,0,0]-Esin_calculated[1][:, 0, 0]/(mass*omega), '.')

            plt.show()

        tolerance: float = 0.07
        assert -m > (p - 1 - tolerance)


if __name__ == "__main__":
    test_dispersion_relation_1d()
    # test_convergence_1d(show_plot=True)