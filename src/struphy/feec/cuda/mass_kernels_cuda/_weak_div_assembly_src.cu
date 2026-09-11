extern "C" __global__
void weak_div_assemble_cuda(
    const long long* s1, const long long* s2, const long long* s3,
    const int ne1, const int ne2, const int ne3,
    const int pi1, const int pi2, const int pi3,
    const int pj1, const int pj2, const int pj3,
    const int st1, const int st2, const int st3,
    const int pad1, const int pad2, const int pad3,
    const double* w1, const double* w2, const double* w3,
    const int nq1, const int nq2, const int nq3,
    const double* bi1, const double* bi2, const double* bi3,
    const double* bj1, const double* bj2, const double* bj3,
    const int ndi1, const int ndi2, const int ndi3,
    const int ndj1, const int ndj2, const int ndj3,
    const double* weight, const double* dl1, const double* dl2, const double* dl3,
    const int component, double* data,
    const int dd2, const int dd3, const int dd4, const int dd5, const int dd6)
{
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long ni = (long long)(pi1+1)*(pi2+1)*(pi3+1);
    const long long nj = (long long)(pj1+1)*(pj2+1)*(pj3+1);
    const long long total = (long long)ne1*ne2*ne3*ni*nj;
    if (tid >= total) return;
    long long t=tid;
    const long long j=t%nj; t/=nj;
    const long long i=t%ni; t/=ni;
    const int e3=t%ne3; t/=ne3;
    const int e2=t%ne2; const int e1=t/ne2;
    const int il3=i%(pi3+1), il2=(i/(pi3+1))%(pi2+1), il1=i/((pi2+1)*(pi3+1));
    const int jl3=j%(pj3+1), jl2=(j/(pj3+1))%(pj2+1), jl1=j/((pj2+1)*(pj3+1));
    const int c1=pad1+(int)s1[e1]-pi1+il1-st1;
    const int c2=pad2+(int)s2[e2]-pi2+il2-st2;
    const int c3=pad3+(int)s3[e3]-pi3+il3-st3;
    const int o1=pad1+jl1-il1, o2=pad2+jl2-il2, o3=pad3+jl3-il3;
    double value=0.0;
    for(int q1=0;q1<nq1;++q1) {
      const long long ib1=((long long)(e1*(pi1+1)+il1)*ndi1)*nq1+q1;
      const long long jb1=((long long)(e1*(pj1+1)+jl1)*ndj1)*nq1+q1;
      const double ti1=bi1[ib1], tn1=bj1[jb1], td1=bj1[jb1+nq1];
      for(int q2=0;q2<nq2;++q2) {
        const long long ib2=((long long)(e2*(pi2+1)+il2)*ndi2)*nq2+q2;
        const long long jb2=((long long)(e2*(pj2+1)+jl2)*ndj2)*nq2+q2;
        const double ti12=ti1*bi2[ib2], tn2=bj2[jb2], td2=bj2[jb2+nq2];
        for(int q3=0;q3<nq3;++q3) {
          const long long ib3=((long long)(e3*(pi3+1)+il3)*ndi3)*nq3+q3;
          const long long jb3=((long long)(e3*(pj3+1)+jl3)*ndj3)*nq3+q3;
          const double tn3=bj3[jb3], td3=bj3[jb3+nq3];
          const double basis=tn1*tn2*tn3;
          const double deriv=component==0 ? td1*tn2*tn3 : (component==1 ? tn1*td2*tn3 : tn1*tn2*td3);
          const long long qi=((long long)(e1*nq1+q1)*(ne2*nq2)+e2*nq2+q2)*(ne3*nq3)+e3*nq3+q3;
          const double dl=component==0 ? dl1[qi] : (component==1 ? dl2[qi] : dl3[qi]);
          value += w1[e1*nq1+q1]*w2[e2*nq2+q2]*w3[e3*nq3+q3]
                   * weight[qi] * ti12*bi3[ib3] * (deriv+basis*dl);
        }
      }
    }
    const long long di=(((((long long)c1*dd2+c2)*dd3+c3)*dd4+o1)*dd5+o2)*dd6+o3;
    atomicAdd(&data[di],value);
}

