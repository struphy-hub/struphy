## Version 2.2.0

### Core changes

* Add matrixless option for weighted mass operator, allow to change None type blocks and update preconditionner. Add the option to compute the matrix vector product for a weighted mass matrix without explicitly assembling the `StencilMatrix`. To use this option pass `matrix_free=True` when initiating the operator. This was implemented using a new class `StencilMatrixlessMassOperator` which is used instead of the StencilMatrix and calls a new kernel `kernel_3d_matrixless` to perform the dot product. !466

* Updated the `MassMatrixPreconditioner` in order to use it also with `WeightedMassOperator` where the weights are not callable but np.ndarray. Added a new parameter `dim_reduce` that can be used in order to approximate the weights as constant along another dimension that the first one as done before. !466

* Faster kernels for assembling mass matrix: improve `kernel_3d_mat` by using smaller array to store part or single dimension of multidimensional array. !461

* Added a new flag `--cprofile` to the console run command. The new default will be to run without Cprofile. Wrote a function `recursive_get_files()` that searches for all `.yml` files in all sub-folders. Same for batch scripts. !460

* Get rid of modes from Jacobian in initial function (pushed-forward). Addition of new option `physical_at_eta` in the `comps` specification of the initial FEEC field. !459

* Added new `Shear_x`, `Shear_y` and `Shear_z` initial conditions. !457

* The format of `parameters.yml` has been changed in order to have more general FEEC initialization. Instead of giving `coords` and `comps` separately, now the p-form of the function from `initial/perturbations.py` must be given at the `comps` where we want to initialize. Choices are:
`null`, `'physical'`, `'0'`, `'3'` for scalar spaces,
`null`, `'physical'`, `'1'`, `'2'`, `'v'`, 'norm' for vector-valued spaces.
Moreover, different mode numbers and amplitudes can be given for different components of the FEEC variables. !456

* Markers can be initialized from the output data of another simulation. !456

* Faster evaluation of splines on interpolation/histopolation/integration grid. Added a new method `Field.eval_tp_fixed` which calls the new faster kernel `eval_mpi_tensor_product_fixed`. !450

* Re-factored the L2 projection; added all de Rham spaces.
The method `L2Projector.get_dofs` replaces old method `WeightedMassOperator.assemble_vec`. Added geom.-projection and L2-integration grid infos as attributes to the Derham class.
`pts`, `wts` for both quad and projector grids; `spans` and `bases` for quad grids only. !452

* Enable lower dimensional simulations by setting `Nel=1`, `p=1` and `spl_kind=True` in the negligible space directions. 
The option `squeeze_out=False` had to be set as the new default in push/pull and MHD equilibrium evaluations. !451

* Bugfix: Added an `Allreduce` in `BasisProjectionOperator` to decide whether to compute the matrix or not based on the value of the weight on all processors (and not just depending on the local processor). !449

### Model specific changes

* `Poisson` was added in toy models. In the `ImplicitDiffusion` propagator a new parameter `A_mat` is created so that one can select a weighted mass matrix in the elliptic operator. `M1_perp` is added in mass.py. Inverse Jacobian is also added in mass.py as `DFinv`. !439

* Added model `VariationalMHD` with the new propagator `VariationalMagFieldAdvection`. !455

* Added fully compressible model `VariationalCompressibleFluid`. Small optimization : make the `WeightedMassMatrix` an attribute of the model instead of being an one from the propagator to avoid assembling repeatedly the same operator. !454

### Documentation, tutorials, testing, etc.

* Added `Almalinux` and `Opensuse` docker images. !462

* Provide an extensive example of Struphy discretization in the section Numerics.
The whole doc has been refactored a bit, tried to improve the browsing experience with more TOCs. !468

* Added dockerfiles and extended `.gitlab-ci.yml` testing for Ubuntu, Fedora, and CentOS. !453

