Example of of a simple 3D stellarator case, run with GVEC for 10000 iterations
 
Files:
* `parameter_ellipStell.ini`: input parameter file for GVEC
* `GVEC_ellipStell_profile_update_State_0000_00010000.dat`: output/restart file, values needed are **marked with** `(*)` in the commented lines `##`, comments can be modified
* `GVEC_ellipStell_modes_U0_LA__0000_00010000.csv`: visualization of each mode (m,n) of X1 sampled in (s)
* `GVEC_ellipStell_modes_U0_X1__0000_00010000.csv`: visualization of each mode (m,n) of X2 sampled in (s)
* `GVEC_ellipStell_modes_U0_X2__0000_00010000.csv`: visualization of each mode (m,n) of LA sampled in (s)
* `GVEC_ellipStell_visu_BC_0000_00000000.vtu`   : paraview visualization file of the boundary (s=1,theta,zeta)
* `GVEC_ellipStell_visu_planes_0000_00010000.vtu` : paraview visualzation file of multiple poloidal planes (s,theta,zeta=const)

