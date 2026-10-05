#pragma once
// ABI-compatible with cunumpy array views; replaced by its header on migration.
// Shapes and strides use 64-bit integers; strides are measured in elements.
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
