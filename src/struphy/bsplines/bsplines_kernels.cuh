#pragma once
namespace struphy_cuda {
constexpr int MAX_SPLINE_DEGREE = 8;
__device__ inline int find_span(const double* t, int n, int p, double eta) {
    int low=p, high=n-1-p;
    if(eta<=t[low]) return low;
    if(eta>=t[high]) return high-1;
    int span=(low+high)/2;
    while(eta<t[span] || eta>=t[span+1]) {
        if(eta<t[span]) high=span; else low=span;
        span=(low+high)/2;
    }
    return span;
}
__device__ inline void basis_funs(const double* t, int p, double eta, int span, double* values) {
    double left[MAX_SPLINE_DEGREE], right[MAX_SPLINE_DEGREE];
    values[0]=1.;
    for(int j=0;j<p;++j) {
        left[j]=eta-t[span-j]; right[j]=t[span+1+j]-eta;
        double saved=0.;
        for(int r=0;r<=j;++r) {
            double temp=values[r]/(right[r]+left[j-r]);
            values[r]=saved+right[r]*temp; saved=left[j-r]*temp;
        }
        values[j+1]=saved;
    }
}
__device__ inline void b_d_splines_slim(const double* t, int p, double eta, int span, double* bn, double* bd) {
    double left[MAX_SPLINE_DEGREE], right[MAX_SPLINE_DEGREE];
    bn[0]=1.;
    for(int j=0;j<p;++j) {
        left[j]=eta-t[span-j]; right[j]=t[span+1+j]-eta;
        double saved=0.;
        if(j==p-1) for(int i=0;i<p;++i) bd[p-1-i]=p/(t[span-i+p]-t[span-i])*bn[p-1-i];
        for(int r=0;r<=j;++r) {
            double temp=bn[r]/(right[r]+left[j-r]);
            bn[r]=saved+right[r]*temp; saved=left[j-r]*temp;
        }
        bn[j+1]=saved;
    }
}
}
