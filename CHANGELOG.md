## Version 2.1.0

### Core changes

**1) The default language for compilation of kernels is now "c" (previously "fortran"). This should enable easier use of Struphy with macOS. !430**

The status of Struphy kernels can be displayed on screen via `struphy compile --status`.

OpenMP for both PIC and FEEC is disabled by default and has to be called with the respective flags, see `struphy compile -h`. The option `--compiler`  can now take the four compilers available in Pyccel ("GNU" is the default, "intel", "PGI", "nvidia").
Moreover, a JSON file can be specified to define a [custom compiler](https://github.com/pyccel/pyccel/blob/devel/docs/compiler.md).

It was necessary to add temporaries for c-slicing to avoid memory leaks in the c-compiled kernels. Moreover, we now enforce c-ordering in `geometry.base.Domain._evaluate_metric_coefficient()`.

Struphy now requires `pyccel>=1.10.0`.

**2) Use `psydac.linalg.solvers.inverse()` for solving $Ax=b$ from now on. Availbale solvers as for now are: `cg`, `pcg`, `bicg`, `bicgstab`, `pbicgstab`, `minres`, `lsmr`, `gmres`. !432**

All inverse operators have been adapted and `struphy/linear_algebra/iterative_solvers.py` has been deprecated.

**3) Auto-detection of kernels and their dependencies. !437**

Struphy files that need to be compiled must now contain the sub-strings `'kernels'` and `'.py'` in the filename (some re-namings where thus necessary). Such files, and their dependencies, are found automatically during compilation (`struphy compile`). 

Kernels and their dependencies can be displayed on screen via `struphy compile --dependencies`. 

In order for the auto-detection of dependencies to work robustly, the imports of kernels into other kernels has to be done in a specific way, namely:
```  
import struphy.module.your_kernels as your_kernels
``` 
This format is now mandatory for all kernel imports in kernels. Files have been changed accordingly.
 
The kernel names are stored in the new state dictionary under `state['kernels']`. The `Makefile` was drastically shortened. New kernels (and their dependencies) will be recognized automatically.

**4) Other core changes:**

* The new method `LinOpWithTransp.toarray_struphy(..., is_spars=False)` allows for conversion of any linear operator (defined solely by a matrix-vector product) into a dense or sparse matrix, repsectively. This works also with MPI.
Eventually this method will be added as `LinearOperator.toarray()` and `LinearOperator.tosparse()` in Psydac. !426 and !431

* `MassMatrixPreconditioner` is now a `LinearOperator`. !426

* The struphy version of Psydac has been upgraded to 0.1.6, which now features:

    - kernels can be compiled with "c" now, in addition to "fortran". !430
    - all inverse operators have a setter method for changing the matrix $A$ !432 
    - preconditioners are passed as `LinearOperators` to `inverse()` !432
    - use new Psydac `IdentityOperator` that does not change the type of a `LinearOperator` under composition !432
    - added `out=` to `transpose()` and `copy()` of `BlockLinearOperator` and `StencilMatrix` !435

* Removed `Sum`, `Multiply` and `Compose`, use Psydac magic commands `+`, `*` and `@` instead. !432

* Added the factor $\sqrt 2$ in particle drawing - thanks @dobell for spotting this:
    ```
    self.velocities = sp.erfinv(2*self.velocities - 1) * np.sqrt(2) * v_th + u_mean
    ```
    In the old version of Struphy there was no factor 2 in the denominator of exp of Gaussian.

* Enable the use of `np.ndarray` to prescribe the weights in `BasisprojectionOperator` !433

* A new `update_weights` method was added to `BasisProjectionOperator`.
This method allow to recycle `BasisProjectionOperator` by updating the coefficient in iterative algorithms. !436

    In the process :

    - Rename `assemble_mat` in `_assemble_mat` as it should now only been called inside the class for updating the matrix.

    - Made `dof_mat` a private attribute (`self._dof_mat`) of the class so that `_assemble_mat` doesn't take any argument and most importantly, `_dof_operator` is automaticaly updated when `_dof_mat` is (no need for creating a new `ComposedLinearOperator`). `dof_mat` is now created when instantiating the class and updated in `_assemble_mat` (before was created in the later function). 

    - Created a flag `use_cache` that allow for recycling some information computed in the assembly of the matrix for later use in case of repeated call to `update_weights`.

* New properties are added to the `Derham` class: `grad_bcfree`, `curl_bcfree` and `div_bcfree` which are Psydac discrete derivatives without boundary operators. !434

* New callable functions of mhd_equilibrium `gradB` and `curl_unit_b` are added for the class `CartesianMHDequilibrium` and `AxisymmMHDequilibrium`. Currently not implemented for `GVECequilibrium` yet. !434

* New velocity sampling for `Particles5D` !434:

    For the parallel velocity, one velocity is drawn according to CDF of 1D Maxwellian (same with `Particles6D`).

    For the perpendicular velocity, the perpendicular speed $v_\perp = \sqrt{v_1^2 + v_2^2}`$ is drawn from the CDF of coordinate transformed $(v_1, v_2) \rightarrow (v_\perp, \theta)$ polar 2D Maxwellian.

* `Maxwellian5DITPA` class is implemented as a kinetic_background of `Particles5D` !434

* Moved test files into the respective source code folders and deleted folder `src/struphy/tests`. !437

    Added the new test groups `models`, `propagators`, `tutorials` and `timings` available for `struphy test` command.

    The option `--monitor` enables the use of `pytest-monitor` to time and memory-check all tests.
    Added automatic `.html` and `.json` generation via `struphy test timings` (when test were monitored). 
    These files are stored under `_pymon/` in the struphy install path and can be used for further studies. 
    Timings can be print on screen via `struphy test timings --verbose`.

* Introduced `state.yml` which replaces `_path.txt` files. The state dictionary collects all info related to paths and compilation status. !437

* Tutorials are now executed as unit tests, and can thus be timed. Launch tutorials via `struphy test tutorials (-n N)`; the command `struphy tutorials` has been removed. Doc has been adapted. !437

* Two new `LinearOperator` classes `CoordinateProjector` and `CoordinateInclusion`: !438

    `CoordinateProjector` correspond to the projection $(H^1)^3 \mapsto (H^1)$ that maps $u = (u_1, u_2, u_3)$ to $u_i$ where $i=1,2,3$ is given when building the projector.
    `CoordinateInclusion` is the transpose of the former one, that is the operator $(H^1) \mapsto (H^1)^3$ that maps $v$ to $(v,0,0)$ if $i=1$, $(0,v,0)$ if $i=2$ and $(0,0,v)$ if $i=3$. 
    Even if I only presented them is the case of $(H^1)^3$ they can be built and use for every space which is a Cartesian product $X=X_1 \times X_2 \times ... $ as projector $X \mapsto X_i$ and the corresponding inclusion $X_i \mapsto X$.

* Provide new templates for Struphy issues and merge requests. !440

* Added an `out` and a `tmp` optional parameter to the `__call__()` method of `Field`. This allows to avoid creating useless tmps when calling the same Field on the same grid repetitively. !443 

* Add the option `--Tend` to `struphy test MODEL` in order to quickly check for memory leaks !441

### Model specific changes

**1) A variational solver for the Burgers equation is implemented as a first simple step toward Variational Non-Linear MHD. !438**

The new model is `VariationalBurgers` in `models.toy`, it relies on only one propagator : `VariationalVelocityAdvection`.

This propagator implements a Crank-Nicolson step for self-advection term in Burger equation,

$$ \int_{\Omega} \partial_t u \cdot v - \frac{1}{3} \int_{\Omega} u \cdot [u,v] dx = 0 \,.$$

**2) Implementent a variational solver for the pressurless compressible fluid model as a second step towards non-linear MHD. !442**

The new model is `VariationalPressurelessFluid` in `models.toy`, it relies on two propagators : `VariationalMomentumAdvection`and `VariationalDensityEvolvePL`

The first propagator implements a Crank-Nicolson step for self-advection term in euler equation,

$$ \int_{\Omega} \partial_t \rho u \cdot v - \rho u \cdot [u,v] dx = 0 \,.$$

And the second one a Crank-Nicolson step for the density advection and corresponding term in the momentum equation 

$$ \int_{\Omega} \partial_t \rho u \cdot v + \frac{|u|^2}{2} \nabla \cdot (\rho v) dx = 0 \,, $$

$$ \partial_t \rho + \nabla \cdot (\rho u) = 0 \,. $$

**3) Implementent a variational solver for the barotropic compressible fluid model as a third step towards non-linear MHD. !444**

The new model is `VariationalBarotropicFluid` in `models.toy`, it relies on two propagators : `VariationalMomentumAdvection`and `VariationalDensityEvolve`

The first propagator implements a Crank-Nicolson step for self-advection term in euler equation,

$$ \int_{\Omega} \partial_t \rho u \cdot v - \rho u \cdot [u,v] dx = 0 $$

And was already implemented in the previous MR !442

And the second one a Crank-Nicolson step for the density advection and corresponding term in the momentum equation 

$$ \int_{\Omega} \partial_t \rho u \cdot v + \big( \frac{|u|^2}{2} - \frac{\partial \rho e}{\partial \rho} \big) \nabla \cdot (\rho v) dx = 0 $$

$$ \partial_t \rho + \nabla \cdot (\rho u) = 0 $$

where e is the internal enegy, here $e=\rho/2$. Modified the already existing `VariationalDensityEvolve` to allow choosing the model and avoid creating another very similar propagator.

**4) Oher model specific changes:**

* Propagator `CurrentCoupling5DGradBxB_dg` is removed and related kernels are commented. !434

* `CurrentCoupling5DDensity` propagator is implemented. !434

* openMP decorators are added for the model `LinearMHDDriftkineticCC` !434


### Documentation, tutorials, etc.



### Struphy-simulations, new files:

None.
