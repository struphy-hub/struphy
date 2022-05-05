from struphy.geometry.domain_3d import Domain

from sympde.topology import Cube, Derham

from psydac.api.discretization import discretize
from psydac.fem.tensor import TensorFemSpace 
from psydac.fem.vector import ProductFemSpace
from psydac.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix
from psydac.linalg.block import BlockVectorSpace, BlockVector, BlockMatrix

from mpi4py import MPI
import numpy as np

# mpi communicator
MPI_COMM = MPI.COMM_WORLD
mpi_rank = MPI_COMM.Get_rank()
MPI_COMM.Barrier()

# Psydac symbolic logical domain
DOMAIN_symb = Cube('C', bounds1=(0, 1), bounds2=(0, 1), bounds3=(0, 1))

# Psydac symbolic Derham
DERHAM_symb = Derham(DOMAIN_symb)

# grid parameters
Nel      = [16, 16, 2]
p        = [3, 3, 2]
spl_kind = [True, True, True] 
n_quad   = [4, 4, 4]
bc       = [['f', 'f'], None, None]

if mpi_rank == 0:
    print(f'\nRank: {mpi_rank}, Nel={Nel}, p={p}, spl_kind={spl_kind}')

# Psydac discrete domain
DOMAIN  = discretize(DOMAIN_symb, ncells=Nel, comm=MPI.COMM_WORLD) # The parallelism is initiated here.

# Psydac discrete Derham
DERHAM  = discretize(DERHAM_symb, DOMAIN, degree=p, periodic=spl_kind)

# Discrete paces
V0 = DERHAM.V0
V1 = DERHAM.V1
V2 = DERHAM.V2
V3 = DERHAM.V3

assert isinstance(V0, TensorFemSpace)
assert isinstance(V1, ProductFemSpace)
assert isinstance(V2, ProductFemSpace)
assert isinstance(V3, TensorFemSpace)

# Associated vector spaces
assert isinstance(V0.vector_space, StencilVectorSpace)
assert isinstance(V1.vector_space, BlockVectorSpace)
assert isinstance(V2.vector_space, BlockVectorSpace)
assert isinstance(V3.vector_space, StencilVectorSpace)

# Stencil objects (distributed)
x0 = StencilVector(V0.vector_space)
A0 = StencilMatrix(V0.vector_space, V0.vector_space)

print(f'Rank: {mpi_rank} | A domain  : {A0.domain}, dimension: {A0.domain.dimension}')
print(f'Rank: {mpi_rank} | A codomain: {A0.codomain}, dimension: {A0.codomain.dimension}')

# Global indices of each process, and paddings
gl_s = x0.starts 
gl_e = x0.ends
pads = x0.pads

gl_s_dom = A0.domain.starts 
gl_e_dom = A0.domain.ends
pads_dom = A0.domain.pads

gl_s_codom = A0.codomain.starts 
gl_e_codom = A0.codomain.ends
pads_codom = A0.codomain.pads

print(f'Rank: {mpi_rank}, global start indices={gl_s}, global end indices={gl_e}, paddings={pads}')
MPI_COMM.Barrier()

assert gl_s==gl_s_dom
assert gl_e==gl_e_dom
assert pads==pads_dom

assert gl_s==gl_s_codom
assert gl_e==gl_e_codom
assert pads==pads_codom

# Data of Stencil objects
if mpi_rank==0:
    print('\nData of Stencil Vector:')
    print(f'Rank: {mpi_rank}, type(x0)={type(x0)}')
    print(f'Rank: {mpi_rank}, type(x0[:, :, :])={type(x0[:, :, :])}')
    print(f'Rank: {mpi_rank}, type(x0[:])={type(x0[:])}')
    print(f'Rank: {mpi_rank}, type(x0._data)={type(x0._data)}')
    print(f'Rank: {mpi_rank}, type(x0.toarray())={type(x0.toarray())}')
    print(f'Rank: {mpi_rank}, type(x0.toarray_local())={type(x0.toarray_local())}')
    print(f'Rank: {mpi_rank}, x0.shape={x0.shape}')
    print(f'Rank: {mpi_rank}, x0[:, :, :].shape={x0[:, :, :].shape}')
    print(f'Rank: {mpi_rank}, x0[:].shape={x0[:].shape}')
    print(f'Rank: {mpi_rank}, x0._data.shape={x0._data.shape}')
    print(f'Rank: {mpi_rank}, x0.toarray().shape={x0.toarray().shape}')
    print(f'Rank: {mpi_rank}, x0.toarray_local().shape={x0.toarray_local().shape}')

    print('\nData of Stencil Matrix:')
    print(f'Rank: {mpi_rank}, type(A0)={type(A0)}')
    print(f'Rank: {mpi_rank}, type(A0[:, :, :, :, :, :])={type(A0[:, :, :, :, :, :])}')
    print(f'Rank: {mpi_rank}, type(A0[:, :])={type(A0[:, :])}')
    print(f'Rank: {mpi_rank}, type(A0._data)={type(A0._data)}')
    print(f'Rank: {mpi_rank}, type(A0.toarray())={type(A0.toarray())}')
    print(f'Rank: {mpi_rank}, type(A0.toarray_local()) does not exist.')
    print(f'Rank: {mpi_rank}, A0.shape={A0.shape}')
    print(f'Rank: {mpi_rank}, A0[:, :, :, :, :, :].shape={A0[:, :, :, :, :, :].shape}')
    print(f'Rank: {mpi_rank}, A0[:, :].shape={A0[:, :].shape}')
    print(f'Rank: {mpi_rank}, A0._data.shape={A0._data.shape}')
    print(f'Rank: {mpi_rank}, A0.toarray().shape={A0.toarray().shape}')
    print(f'Rank: {mpi_rank}, A0.toarray_local().shape does not exist.')
MPI_COMM.Barrier()



y0 = x0.copy()
# Assign values directly to Stencil Vector (no padding + global indices !!)
# -------------------------------------------------------------------------
x0[gl_s[0] : gl_e[0] + 1, gl_s[1] : gl_e[1] + 1, gl_s[2]] = mpi_rank
x0.update_ghost_regions()

# try writing without end index:
z0 = x0.copy()
try:
    print(f'\nz0[:].shape[0]: {z0[:].shape[0]}') 
    for i in range(z0[:].shape[0]):
        z0[i] = i
except:
    print('\nWrong acces of Stencilvector (!): for i in range(x0[:].shape[0]) gets out of bounds.\n')

print(f'Rank: {mpi_rank}, x0[:, :, gl_s]={x0[:, :, gl_s[2]]}')
MPI_COMM.Barrier()

# Assign values to _data attribute (=numpy array) (padding + local indices !!)
# ----------------------------------------------------------------------------
y0._data[pads[0] : -pads[0], pads[1] : -pads[1], pads[2]] = mpi_rank
y0.update_ghost_regions()

# try writing without end index:
try:
    print(f'\nz0._data.shape[0]: {z0._data.shape[0]}')
    for i in range(z0._data.shape[0]):
        z0._data[i] = i
except:
    print('\nWrong acces of ._data (!): for i in range(x0._data.shape[0]).\n')

#print(f'Rank: {mpi_rank}, y0[:, :, gl_s]={y0[:, :, gl_s[2]]}')
#MPI_COMM.Barrier()

assert np.allclose(x0.toarray_local(), y0.toarray_local())



B0 = A0.copy()
# Assign values directly to Stencil Matrix
# ---------------------------------------- 
# rows:   no padding + global indices
# colums: no padding + from -p to p, diagonal is at index 0 !!)
for n in range(2*p[0] + 1):
    A0[gl_s[0] : gl_e[0] + 1, gl_s[1], gl_s[2], n - p[0], 0, 0] = (n - p[0])*mpi_rank 
A0.update_ghost_regions()

print(f'Rank: {mpi_rank}, A0[:, gl_s, gl_s, :, 0, 0]={A0[:, gl_s[1], gl_s[2], :, 0, 0]}')
MPI_COMM.Barrier()

# Assign values to _data attribute (=numpy array)
# -----------------------------------------------
# rows:   padding + local indices 
# colums: padding + from -p to p, diagonal is at index 0 !!
for n in range(2*p[0] + 1):
    B0._data[pads[0] : -pads[0], pads[1], pads[2], pads[0] + n - p[0], pads[1], pads[2]] = (n - p[0])*mpi_rank
B0.update_ghost_regions()

#print(f'Rank: {mpi_rank}, B0[:, gl_s, gl_s, :, 0, 0]={B0[:, gl_s[1], gl_s[2], :, 0, 0]}')
#MPI_COMM.Barrier()

assert np.allclose(A0.toarray(), B0.toarray())

# Create and test Identity matrix
Id = StencilMatrix(V0.vector_space, V0.vector_space)
u0 = StencilVector(V0.vector_space)

np.random.seed(42)
u0._data[:] = np.random.random(u0._data.shape)

# Put 1 on the diagonal
Id._data[pads[0] : -pads[0], pads[1] : -pads[1], pads[2] : -pads[2], pads[0], pads[1], pads[2]] = 1.
MPI_COMM.Barrier()

assert np.allclose(Id.dot(u0).toarray(), u0.toarray())
assert np.allclose((Id.dot(u0) - u0).toarray(), np.zeros(u0.toarray().shape))



# Updating ghost regions:
# -----------------------
y0 = x0.copy()
# Writing into the ghost regions does nothing:
if mpi_rank==0:
    y0._data[:pads[0], pads[1] : -pads[1], pads[2]] = 999
y0.update_ghost_regions()

#print(f'Rank: {mpi_rank}, x0[:, :, gl_s]={x0[:, :, gl_s[2]]}')
#MPI_COMM.Barrier()

assert np.allclose(x0.toarray_local(), y0.toarray_local())

# Writing inside the padding and then updating:
if mpi_rank==0:
    x0._data[pads[0] : -pads[0], pads[1] : -pads[1], pads[2]] = 999
x0.update_ghost_regions()

print(f'Rank: {mpi_rank}, x0[:, :, gl_s]={x0[:, :, gl_s[2]]}')
MPI_COMM.Barrier()


