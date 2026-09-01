"""CUDA kernels for dynamic weighted basis-projection matrices."""

_ASSEMBLE_SRC = r"""
extern "C" __global__
void assemble_weighted_basis_3d_cuda(
 const long long* row1,const long long* row2,const long long* row3,
 const long long* span1,const long long* span2,const long long* span3,
 const double* w1,const double* w2,const double* w3,
 const double* b1,const double* b2,const double* b3,const double* fun,
 const int ni1,const int ni2,const int ni3,const int nq1,const int nq2,const int nq3,
 const int p1,const int p2,const int p3,const int so1,const int so2,const int so3,
 const int pi1,const int pi2,const int pi3,const int po1,const int po2,const int po3,
 const int dimi1,const int dimi2,const int dimi3,const int dimo1,const int dimo2,const int dimo3,
 const int pout1,const int pout2,const int pout3,double* mat,
 const int md2,const int md3,const int md4,const int md5,const int md6)
{
 long long tid=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 const long long nb=(long long)(p1+1)*(p2+1)*(p3+1);
 const long long nq=(long long)nq1*nq2*nq3;
 const long long total=(long long)ni1*ni2*ni3*nq*nb;
 if(tid>=total)return;
 long long t=tid; const long long bb=t%nb;t/=nb; const long long qq=t%nq;t/=nq;
 const int kk=t%ni3;t/=ni3; const int jj=t%ni2;const int ii=t/ni2;
 const int b3i=bb%(p3+1),b2i=(bb/(p3+1))%(p2+1),b1i=bb/((p2+1)*(p3+1));
 const int q3=qq%nq3,q2=(qq/nq3)%nq2,q1=qq/(nq2*nq3);
 const int i=(int)row1[ii],j=(int)row2[jj],k=(int)row3[kk];
 int m=(int)span1[ii*nq1+q1]-p1+b1i;
 int n=(int)span2[jj*nq2+q2]-p2+b2i;
 int o=(int)span3[kk*nq3+q3]-p3+b3i;
 const int cut1=dimo1<=dimi1?p1:pout1,cut2=dimo2<=dimi2?p2:pout2,cut3=dimo3<=dimi3?p3:pout3;
 int d=m-(i+so1);if(d>cut1)m-=dimi1;else if(d<-cut1)m+=dimi1;
 d=n-(j+so2);if(d>cut2)n-=dimi2;else if(d<-cut2)n+=dimi2;
 d=o-(k+so3);if(d>cut3)o-=dimi3;else if(d<-cut3)o+=dimi3;
 const int c1=pi1+m-(i+so1),c2=pi2+n-(j+so2),c3=pi3+o-(k+so3);
 const long long fi=((long long)(ii*nq1+q1)*(ni2*nq2)+jj*nq2+q2)*(ni3*nq3)+kk*nq3+q3;
 const double value=fun[fi]*w1[ii*nq1+q1]*w2[jj*nq2+q2]*w3[kk*nq3+q3]
  *b1[((long long)ii*nq1+q1)*(p1+1)+b1i]
  *b2[((long long)jj*nq2+q2)*(p2+1)+b2i]
  *b3[((long long)kk*nq3+q3)*(p3+1)+b3i];
 const long long mi=(((((long long)(po1+i)*md2+(po2+j))*md3+(po3+k))*md4+c1)*md5+c2)*md6+c3;
 atomicAdd(&mat[mi],value);
}
"""

_kernel = None


def assemble_dofs_for_weighted_basisfuns_3d_gpu(
    mat,
    starts_in,
    ends_in,
    pads_in,
    starts_out,
    ends_out,
    pads_out,
    fun,
    weights,
    spans,
    bases,
    subs,
    dims_in,
    dims_out,
    degrees_out,
):
    import cupy as cp
    import numpy as np

    global _kernel
    if _kernel is None:
        _kernel = cp.RawKernel(_ASSEMBLE_SRC, "assemble_weighted_basis_3d_cuda")
    spans = tuple(cp.ascontiguousarray(cp.asarray(x, dtype=cp.int64)) for x in spans)
    weights = tuple(cp.ascontiguousarray(cp.asarray(x, dtype=cp.float64)) for x in weights)
    bases = tuple(cp.ascontiguousarray(cp.asarray(x, dtype=cp.float64)) for x in bases)
    rows = tuple(cp.arange(len(x), dtype=cp.int64) - cp.cumsum(cp.asarray(x, dtype=cp.int64)) for x in subs)
    fun = cp.ascontiguousarray(fun)
    mat.fill(0.0)
    ni = tuple(x.shape[0] for x in spans)
    nq = tuple(x.shape[1] for x in spans)
    degree = tuple(x.shape[2] - 1 for x in bases)
    total = int(np.prod(ni) * np.prod(nq) * np.prod([p + 1 for p in degree]))
    threads = 256
    _kernel(
        ((total + threads - 1) // threads,),
        (threads,),
        (
            *rows,
            *spans,
            *weights,
            *bases,
            fun,
            *(np.int32(x) for x in (*ni, *nq, *degree)),
            *(np.int32(x) for x in starts_out),
            *(np.int32(x) for x in pads_in),
            *(np.int32(x) for x in pads_out),
            *(np.int32(x) for x in dims_in),
            *(np.int32(x) for x in dims_out),
            *(np.int32(x) for x in degrees_out),
            mat,
            *(np.int32(x) for x in mat.shape[1:]),
        ),
    )
