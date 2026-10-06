#pragma once
// ABI-compatible with cunumpy array views; replaced by its header on migration.
// Shapes and strides use 64-bit integers; strides are measured in elements.
//
// Kernels index a view like the pyccel kernels index the array: a(i, j) for a[i, j]. They never
// read the strides themselves. The strides are set on the host, from the CuPy array's own
// strides (arr.strides // arr.itemsize): in CudaKernel.__call__ (struphy/utils/kernel_backends.py)
// for kernel arguments and in Argument._pack (struphy/utils/cuda_arguments.py) for struct fields.
//
// The layout is NumPy's: for a C-contiguous (row-major) array of shape (n0, n1, n2),
// strides = (n1 * n2, n2, 1), so the last index runs fastest; for shape (n_markers, n_cols),
// strides = (n_cols, 1) and markers(ip, j) is data[ip * n_cols + j]. A sliced or transposed
// CuPy array keeps its own strides, so it is indexed correctly without a copy.
template<class T> struct Array1D {
    T* data;
    long long shape[1], strides[1];
    /**
     * Element (i) of the array, as the pyccel kernels index it with a[i].
     *
     * @param i Index along axis 0; not bounds-checked.
     * @return Reference to data[i * strides[0]], for reads and writes.
     */
    __device__ inline T& operator()(long long i) const {
        return data[i*strides[0]];
    }
};

template<class T> struct Array2D {
    T* data;
    long long shape[2], strides[2];
    /**
     * Element (i, j) of the array, as the pyccel kernels index it with a[i, j].
     *
     * @param i Index along axis 0; not bounds-checked.
     * @param j Index along axis 1; not bounds-checked.
     * @return Reference to data[i * strides[0] + j * strides[1]], for reads and writes.
     */
    __device__ inline T& operator()(long long i, long long j) const {
        return data[i*strides[0]+j*strides[1]];
    }
};

template<class T> struct Array3D {
    T* data;
    long long shape[3], strides[3];
    /**
     * Element (i, j, k) of the array, as the pyccel kernels index it with a[i, j, k].
     *
     * @param i Index along axis 0; not bounds-checked.
     * @param j Index along axis 1; not bounds-checked.
     * @param k Index along axis 2; not bounds-checked.
     * @return Reference to data[i * strides[0] + j * strides[1] + k * strides[2]], for reads and writes.
     */
    __device__ inline T& operator()(long long i, long long j, long long k) const {
        return data[i*strides[0]+j*strides[1]+k*strides[2]];
    }
};
