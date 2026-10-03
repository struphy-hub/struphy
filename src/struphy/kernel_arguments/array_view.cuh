#pragma once
// ABI-compatible with cunumpy Array3D; replaced by its header on migration.
template<class T> struct Array3D {
    T* data;
    long long shape[3], strides[3];
    __device__ inline T& operator()(long long i, long long j, long long k) const {
        return data[i*strides[0]+j*strides[1]+k*strides[2]];
    }
};
