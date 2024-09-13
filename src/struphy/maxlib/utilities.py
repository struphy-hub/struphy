# coding: utf-8

import numpy as np
from math                  import sqrt
from psydac.linalg.stencil import StencilVectorSpace, StencilVector
from psydac.linalg.block   import BlockVector, BlockVectorSpace

__all__ = ('array_to_psydac', 'petsc_to_psydac', '_sym_ortho')

def array_to_psydac(x, Xh):
    """ converts a numpy array to StencilVector or BlockVector format"""

    if isinstance(Xh, BlockVectorSpace):
        u = BlockVector(Xh)
        if isinstance(Xh.spaces[0], BlockVectorSpace):
            for d in range(len(Xh.spaces)):
                starts = [np.array(V.starts) for V in Xh.spaces[d].spaces]
                ends   = [np.array(V.ends)   for V in Xh.spaces[d].spaces]

                for i in range(len(starts)):
                    g = tuple(slice(s,e+1) for s,e in zip(starts[i], ends[i]))
                    shape = tuple(ends[i]-starts[i]+1)
                    u[d][i][g] = x[:np.product(shape)].reshape(shape)
                    x       = x[np.product(shape):]

        else:
            starts = [np.array(V.starts) for V in Xh.spaces]
            ends   = [np.array(V.ends)   for V in Xh.spaces]

            for i in range(len(starts)):
                g = tuple(slice(s,e+1) for s,e in zip(starts[i], ends[i]))
                shape = tuple(ends[i]-starts[i]+1)
                u[i][g] = x[:np.product(shape)].reshape(shape)
                x       = x[np.product(shape):]

    elif isinstance(Xh, StencilVectorSpace):

        u =  StencilVector(Xh)
        starts = np.array(Xh.starts)
        ends   = np.array(Xh.ends)
        g = tuple(slice(s, e+1) for s,e in zip(starts, ends))
        shape = tuple(ends-starts+1)
        u[g] = x[:np.product(shape)].reshape(shape)
    else:
        raise ValueError('Xh must be a StencilVectorSpace or a BlockVectorSpace')

    u.update_ghost_regions()
    return u


# --------------------------------------------------------------------------- # 
# New by Max START

from line_profiler import profile
from hashlib import sha256
import hashlib
import pickle

# Cache dictionary
cache = {}

# @profile
# def get_cache_key(vec, inds=None):
#     """Generate a unique key for the PETSc vector based on its properties, contents, and indices."""
#     # Collect relevant properties from the PETSc vector
#     size = vec.getSize()
#     vtype = vec.getType()

#     # Retrieve a summary of the vector data
#     vector_array = vec.getArray()
#     vector_summary = (size, vtype)

#     # Hash the vector summary to ensure uniqueness
#     summary_hash = hashlib.sha256(pickle.dumps(vector_summary)).hexdigest()
    
#     # Include indices in the key if provided
#     if inds is not None:
#         inds_hash = hashlib.sha256(pickle.dumps(inds)).hexdigest()
#     else:
#         inds_hash = ''
    
#     # Create a unique key combining size, type, data hash, and indices hash
#     key = {
#         'size': size,
#         'type': vtype,
#         'summary_hash': summary_hash,
#         'inds_hash': inds_hash
#     }
    
#     # Convert key to a string and hash it
#     key_str = pickle.dumps(key)
#     key_hash = hashlib.sha256(key_str).hexdigest()
#     return key_hash

@profile
def get_cache_key(vec, inds):
    """Generate a unique key for the PETSc vector based on its properties and indices."""
    # Collect relevant properties from the PETSc vector
    size = vec.getSize()
    vtype = vec.getType()

    # Ensure 'inds' is treated as a tuple
    if isinstance(inds, (int, np.integer)):
        inds = (inds,)
    else:
        inds = tuple(inds)

    # Create a unique key as a tuple
    key_parts = (size, vtype, inds)

    # Hash the tuple representation directly
    key_bytes = str(key_parts).encode('utf-8')
    key_hash = hashlib.blake2b(key_bytes, digest_size=16).hexdigest()  # Reduced digest size

    return key_hash

@profile
def petsc_to_psydac_new(vec, Xh):
    """ converts a petsc Vec object to a StencilVector or a BlockVector format.
        We gather the petsc global vector in all the processes and extract the chunk owned by the Psydac Vector.
        .. warning: This function will not work if the global vector does not fit in the process memory.
    """
    save_in_cache = True
    if isinstance(Xh, BlockVectorSpace):
        u = BlockVector(Xh)
        if isinstance(Xh.spaces[0], BlockVectorSpace):

            comm       = u[0][0].space.cart.global_comm
            dtype      = u[0][0].space.dtype
            sendcounts = np.array(comm.allgather(len(vec.array)))
            recvbuf    = np.empty(sum(sendcounts), dtype=dtype)

            # gather the global array in all the procs
            comm.Allgatherv(sendbuf=vec.array, recvbuf=(recvbuf, sendcounts))

            inds = 0
            for d,space in enumerate(Xh.spaces):
                # replace Xh.spaces[d] with space
                starts = [np.array(V.starts) for V in Xh.spaces[d].spaces]
                ends   = [np.array(V.ends)   for V in Xh.spaces[d].spaces]

                for i,start in enumerate(starts):
                    # replace starts[i] with start
                    idx = tuple( slice(m*p,-m*p) for m,p in zip(u.space.spaces[d].spaces[i].pads, u.space.spaces[d].spaces[i].shifts) )
                    shape = tuple(ends[i]-starts[i]+1)
                    npts  = Xh.spaces[d].spaces[i].npts
                    
                    key = get_cache_key(vec)
                    if key in cache and save_in_cache:
                        # print('Cache hit 2')
                        indices, idx = cache[key]
                    else:
                        # print('Cache miss 2')
                        # compute the global indices of the coefficents owned by the process using starts and ends
                        indices = np.array([np.ravel_multi_index( [s+x for s,x in zip(starts[i], xx)], dims=npts,  order='C' ) for xx in np.ndindex(*shape)] )
                        cache[key] = (indices, idx)
                    
                    vals = recvbuf[indices+inds]
                    u[d][i]._data[idx] = vals.reshape(shape)
                    inds += np.product(npts)

        else:
            comm       = u[0].space.cart.global_comm
            dtype      = u[0].space.dtype
            sendcounts = np.array(comm.allgather(len(vec.array)))
            recvbuf    = np.empty(sum(sendcounts), dtype=dtype)

            # gather the global array in all the procs
            comm.Allgatherv(sendbuf=vec.array, recvbuf=(recvbuf, sendcounts))

            inds = 0
            starts = [np.array(V.starts) for V in Xh.spaces]
            ends   = [np.array(V.ends)   for V in Xh.spaces]
            for i,start in enumerate(starts):
                # replace starts[i] with start

                idx = tuple( slice(m*p,-m*p) for m,p in zip(u.space.spaces[i].pads, u.space.spaces[i].shifts) )
                shape = tuple(ends[i]-starts[i]+1)
                npts  = Xh.spaces[i].npts
                # compute the global indices of the coefficents owned by the process using starts and ends
                #print(f"{i = }, {start = }, {idx = }, {shape = }, {npts = } {vec = } {inds = }")
                key = get_cache_key(vec, inds)
                if key in cache and save_in_cache:
                    print('Cache hit 2')
                    indices = cache[key]
                else:
                    print('Cache miss 2')
                    # compute the global indices of the coefficents owned by the process using starts and ends
                    indices = np.array([np.ravel_multi_index( [s+x for s,x in zip(starts[i], xx)], dims=npts,  order='C' ) for xx in np.ndindex(*shape)] )
                    cache[key] = indices
                print(f"{idx = }")
                vals = recvbuf[indices+inds]
                u[i]._data[idx] = vals.reshape(shape)
                inds += np.product(npts)

    elif isinstance(Xh, StencilVectorSpace):

        u          = StencilVector(Xh)
        comm       = u.space.cart.global_comm
        dtype      = u.space.dtype
        sendcounts = np.array(comm.allgather(len(vec.array)))
        recvbuf    = np.empty(sum(sendcounts), dtype=dtype)

        # gather the global array in all the procs
        comm.Allgatherv(sendbuf=vec.array, recvbuf=(recvbuf, sendcounts))

        # compute the global indices of the coefficents owned by the process using starts and ends
        starts = np.array(Xh.starts)
        ends   = np.array(Xh.ends)
        shape  = tuple(ends-starts+1)
        npts   = Xh.npts
        indices = np.array([np.ravel_multi_index( [s+x for s,x in zip(starts, xx)], dims=npts,  order='C' ) for xx in np.ndindex(*shape)] )
        idx = tuple( slice(m*p,-m*p) for m,p in zip(u.space.pads, u.space.shifts) )
        vals = recvbuf[indices]
        u._data[idx] = vals.reshape(shape)

    else:
        raise ValueError('Xh must be a StencilVectorSpace or a BlockVectorSpace')

    u.update_ghost_regions()
    return u
# New by Max END
# --------------------------------------------------------------------------- # 

@profile
def petsc_to_psydac(vec, Xh):
    """ converts a petsc Vec object to a StencilVector or a BlockVector format.
        We gather the petsc global vector in all the processes and extract the chunk owned by the Psydac Vector.
        .. warning: This function will not work if the global vector does not fit in the process memory.
    """

    if isinstance(Xh, BlockVectorSpace):
        u = BlockVector(Xh)
        if isinstance(Xh.spaces[0], BlockVectorSpace):

            comm       = u[0][0].space.cart.global_comm
            dtype      = u[0][0].space.dtype
            sendcounts = np.array(comm.allgather(len(vec.array)))
            recvbuf    = np.empty(sum(sendcounts), dtype=dtype)

            # gather the global array in all the procs
            comm.Allgatherv(sendbuf=vec.array, recvbuf=(recvbuf, sendcounts))

            inds = 0
            for d in range(len(Xh.spaces)):
                starts = [np.array(V.starts) for V in Xh.spaces[d].spaces]
                ends   = [np.array(V.ends)   for V in Xh.spaces[d].spaces]

                for i in range(len(starts)):
                    idx = tuple( slice(m*p,-m*p) for m,p in zip(u.space.spaces[d].spaces[i].pads, u.space.spaces[d].spaces[i].shifts) )
                    shape = tuple(ends[i]-starts[i]+1)
                    npts  = Xh.spaces[d].spaces[i].npts
                    # compute the global indices of the coefficents owned by the process using starts and ends
                    indices = np.array([np.ravel_multi_index( [s+x for s,x in zip(starts[i], xx)], dims=npts,  order='C' ) for xx in np.ndindex(*shape)] )
                    vals = recvbuf[indices+inds]
                    u[d][i]._data[idx] = vals.reshape(shape)
                    inds += np.product(npts)

        else:
            comm       = u[0].space.cart.global_comm
            dtype      = u[0].space.dtype
            sendcounts = np.array(comm.allgather(len(vec.array)))
            recvbuf    = np.empty(sum(sendcounts), dtype=dtype)

            # gather the global array in all the procs
            comm.Allgatherv(sendbuf=vec.array, recvbuf=(recvbuf, sendcounts))

            inds = 0
            starts = [np.array(V.starts) for V in Xh.spaces]
            ends   = [np.array(V.ends)   for V in Xh.spaces]
            for i in range(len(starts)):
                idx = tuple( slice(m*p,-m*p) for m,p in zip(u.space.spaces[i].pads, u.space.spaces[i].shifts) )
                shape = tuple(ends[i]-starts[i]+1)
                npts  = Xh.spaces[i].npts
                # compute the global indices of the coefficents owned by the process using starts and ends
                indices = np.array([np.ravel_multi_index( [s+x for s,x in zip(starts[i], xx)], dims=npts,  order='C' ) for xx in np.ndindex(*shape)] )
                vals = recvbuf[indices+inds]
                u[i]._data[idx] = vals.reshape(shape)
                inds += np.product(npts)

    elif isinstance(Xh, StencilVectorSpace):

        u          = StencilVector(Xh)
        comm       = u.space.cart.global_comm
        dtype      = u.space.dtype
        sendcounts = np.array(comm.allgather(len(vec.array)))
        recvbuf    = np.empty(sum(sendcounts), dtype=dtype)

        # gather the global array in all the procs
        comm.Allgatherv(sendbuf=vec.array, recvbuf=(recvbuf, sendcounts))

        # compute the global indices of the coefficents owned by the process using starts and ends
        starts = np.array(Xh.starts)
        ends   = np.array(Xh.ends)
        shape  = tuple(ends-starts+1)
        npts   = Xh.npts
        indices = np.array([np.ravel_multi_index( [s+x for s,x in zip(starts, xx)], dims=npts,  order='C' ) for xx in np.ndindex(*shape)] )
        idx = tuple( slice(m*p,-m*p) for m,p in zip(u.space.pads, u.space.shifts) )
        vals = recvbuf[indices]
        u._data[idx] = vals.reshape(shape)

    else:
        raise ValueError('Xh must be a StencilVectorSpace or a BlockVectorSpace')

    u.update_ghost_regions()
    return u

def _sym_ortho(a, b):
    """
    Stable implementation of Givens rotation.
    This function was taken from the scipy repository
    https://github.com/scipy/scipy/blob/master/scipy/sparse/linalg/isolve/lsqr.py

    Notes
    -----
    The routine 'SymOrtho' was added for numerical stability. This is
    recommended by S.-C. Choi in [1]_.  It removes the unpleasant potential of
    ``1/eps`` in some important places (see, for example text following
    "Compute the next plane rotation Qk" in minres.py).

    References
    ----------
    .. [1] S.-C. Choi, "Iterative Methods for Singular Linear Equations
           and Least-Squares Problems", Dissertation,
           http://www.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf
    """
    if b == 0:
        return np.sign(a), 0, abs(a)
    elif a == 0:
        return 0, np.sign(b), abs(b)
    elif abs(b) > abs(a):
        tau = a / b
        s = np.sign(b) / sqrt(1 + tau * tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = np.sign(a) / sqrt(1+tau*tau)
        s = c * tau
        r = a / c
    return c, s, r