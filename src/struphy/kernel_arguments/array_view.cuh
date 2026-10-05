#pragma once
// ABI-compatible with cunumpy array views; replaced by its header on migration.
// Shapes and strides use 64-bit integers; strides are measured in elements.
template<class T> struct Array2D {
    T* data;
    long long shape[2], strides[2];
    __device__ inline T& operator()(long long i, long long j) const {
        return data[i*strides[0]+j*strides[1]];
    }
};

template<class T> struct Array3D {
    T* data;
    long long shape[3], strides[3];
    __device__ inline T& operator()(long long i, long long j, long long k) const {
        return data[i*strides[0]+j*strides[1]+k*strides[2]];
    }
};
