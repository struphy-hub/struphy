from psydac.linalg.stencil import StencilMatrix

from psydac.api.discretization import discretize

from sympde.topology import Line, Square, Cube, Derham

from sympy import sqrt

import struphy.feec.bsplines_kernels as bsp

import numpy as np

from mpi4py import MPI

from time import sleep

mpi_comm = MPI.COMM_WORLD
rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()


# parameters 1d
Nel = [16]
p = [3]
spl_kind = [True]

Np_per_cell = 8

# create and discretize domain
F_symb = Line('L', bounds=(0, 1))
# F_symb = Square('S', bounds1=(0, 1), bounds2=(0, 1))

derham_symb = Derham(F_symb)
# derham_symb = Derham(F_symb, sequence=('H1', 'Hcurl', 'L2'))

domain_h = discretize(F_symb, ncells=Nel, comm=mpi_comm)

# create discrete derham
derham = discretize(derham_symb, domain_h, degree=p, periodic=spl_kind)

# domain decomposition according to psydac (V0 space)
nel_start = derham.V0.local_domain[0][0]
nel_end   = derham.V0.local_domain[1][0]
nel_loc   = nel_end - nel_start + 1

brk_start = derham.V0.breaks[0][nel_start]
brk_end   = derham.V0.breaks[0][nel_end + 1]
brk_delta = brk_end - brk_start

# array (nproc, 3) with nel_start, nel_end and # of elements of each process
domain_decomp = np.zeros(3*mpi_size, dtype=int)
mpi_comm.Allgather(np.array([nel_start, nel_end, nel_loc]), domain_decomp)
domain_decomp = domain_decomp.reshape(mpi_size, 3)

# print(domain_decomp)
# exit()

mpi_comm.Barrier()
print()

print(f'process # {rank} : Nel from {nel_start} to {nel_end} | local # cells {nel_loc} | local domain from {brk_start} to {brk_end}')
# exit()

# particle loading
np.random.seed(1607)

particles_loc = np.random.rand(Np_per_cell*nel_loc)*brk_delta + brk_start

print()
print(f'process # {rank} : local particles \n {particles_loc}\n')
# exit()


# accumulation matrix as stencil matrix
A = StencilMatrix(derham.V0.vector_space, derham.V0.vector_space)

starts = A.domain.starts[0]
ends = A.domain.ends[0]
pads = A.domain.pads[0]

# mpi_comm.Barrier()
# print(f'\n process # {rank} : shape of StencilMatrix {A.shape}')
# mpi_comm.Barrier()
# print(f'\n process # {rank} : shape of full StencilMatrix {A.toarray().shape}')

# mpi_comm.Barrier()
# print(f'\n process # {rank} : StencilMatrix : \n {A._data}')

# exit()

mpi_comm.Barrier()
print(f'\n process # {rank} : spline indices from {starts} to {ends} | total # of splines : {ends - starts + 1}')

# mpi_comm.Barrier()
# print(f'\n process # {rank} : domain indices from {A.domain.starts[0]} to {A.domain.ends[0]}')
# print(f'\n process # {rank} : codomain indices from {A.codomain.starts[0]} to {A.codomain.ends[0]}')

# exit()

A._data[:, pads] = (rank + 1)*2

mpi_comm.Barrier()
if rank == 0:
    print('========= before any ghosting ==================')
mpi_comm.Barrier()
print(f'\n process # {rank} : StencilMatrix : \n {A._data}')

# exit()

# mpi_comm.Barrier()
# print(f'\n process # {rank} : left ghosts before updating : \n {A._data[:pads,:]}')

# A.update_ghost_regions()

# mpi_comm.Barrier()
# print(f'\n process # {rank} : left ghosts after updating : \n {A._data[:pads,:]}')

# mpi_comm.Barrier()
# print(f'\n process # {rank} : shape of left ghosts : {np.shape(A._data[:pads,:])} c.f. {pads}, {2*p[0]+1}')

# exit()


mpi_comm.Barrier()
if rank == 0:
    print('\n Now comes the ghosting \n')

mpi_comm.Barrier()

l = 71

# send left ghosts -> become right ghosts for the other process
# mpi_comm.Isend(A._data[:pads,:], dest=(rank-1)%mpi_size, tag=rank + 10)
mpi_comm.Isend(A._data[:pads,:].copy(), dest=(rank-1)%mpi_size, tag=rank + 10)
# send_l = np.random.rand(30*l)*(1-rank)
# mpi_comm.Isend(np.random.rand(30*l)*(1-rank), dest=(rank-1)%mpi_size, tag=rank + 10)
# send right ghosts -> become left ghosts for the other process
# send_r = A._data[-pads:,:]
# mpi_comm.Isend(send_r, dest=(rank+1)%mpi_size, tag=rank + 30)
mpi_comm.Isend(A._data[-pads:,:], dest=(rank+1)%mpi_size, tag=rank + 30)


# receive ghosts from the left -> right ghosts from the other process
ghosts_l = np.zeros((pads, 2*p[0]+1), dtype=float)
req_l = mpi_comm.Irecv(ghosts_l, source=(rank-1)%mpi_size, tag= (rank-1)%mpi_size + 30)
re_l = False
while re_l == False :
    re_l = MPI.Request.Test(req_l)

# print(f'\n process # {rank} has received left ghosts : \n {ghosts_l}')

# receive ghosts from the right -> left ghosts from the other process
ghosts_r = np.zeros((pads, 2*p[0]+1), dtype=float)
# ghosts_r = np.zeros((30*l), dtype=float)
request_r = mpi_comm.Irecv(ghosts_r, source=(rank+1)%mpi_size, tag= (rank+1)%mpi_size + 10)
re_r = False
while re_r == False :
    re_r = MPI.Request.Test(request_r)

print(f'rank {rank} has received max value of {np.max(ghosts_r)}')

# A._data[pads:2*pads, :] += ghosts_r[:pads, :2*p[0]+1]
# ghosts_r = ghosts_r.reshape(30,l)
A._data[pads:2*pads, :] += ghosts_r[:pads, :2*p[0]+1]

# print(f'\n process # {rank} has received right ghosts : \n {ghosts_r}')

print('ghosting done')

mpi_comm.Barrier()
if rank == 0:
    print('========= after the ghosting ==================')
mpi_comm.Barrier()
print(f'\n process # {rank} : StencilMatrix : \n {A._data}')


exit()

A._data[pads:2*pads,:] += np.roll(ghosts_l, axis=1, shift=1)
A._data[-2*pads:-pads,:] += np.roll(ghosts_r, axis=1, shift=1)




mpi_comm.Barrier()
if rank == 0:
    print('========= after the ghosting ==================')
mpi_comm.Barrier()
print(f'\n process # {rank} : StencilMatrix : \n {A._data}')
sleep(0.1)

if rank == 0:
    print('\n ========= and now updating yields ==================')
A.update_ghost_regions()
mpi_comm.Barrier()
print(f'\n process # {rank} : StencilMatrix : \n {A._data}')
mpi_comm.Barrier()
print(f'\n process # {rank} : StencilMatrix : \n {A.toarray()}')
# print(f'\n process # {rank} : shape StencilMatrix : {A.toarray().shape}')
