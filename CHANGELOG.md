## Version 2.4.3

### Headlines

* New explicit s-stage Runge Kutta solvers for FEEC variables; takes ODE vector fields (right-hand sides) as Python callable inputs !677
* Improved profiling: extended the profiling capabilities by including time traces which can be enabled with `--time-traces`; this generates some nice plots too !681
* Added model `HasegawaWakatani` for plasma turbulence simulations !682



### Other user news

* Update variational MHD with pressure variable, add linear and delta-f variational MHD models !683
* Optional install of physics packages (like `gvec` and `desc`) via `[phys]`; this makes for an easier base install with less dependencies !688
* Clean-up of variational propagators for MHD; readability is much improved !685
* Include [PSYDAC update #493](https://github.com/pyccel/psydac/pull/493): `LinearOperator` now as the new method `dot_inner`; `Vector` has `dot` replaced by `inner` for the inner product !698



### Developer news

* Faster push pipeline !674
* Adding `local_projectors` to Derham setup !670
* Make macOS jobs run concurrently in CI !671
* move `Derham.Field` outside of Derham class; rename `Field` -> `SplineFunction`; rename `Derham.create_field` -> `Derham.create_spline_function` !682
* allow for 5d array as list entry in weights for `create_weighted_mass` !682
* added new pre-commit-hook `nbstripout` that checks for clean notebooks !682
* allow for callable in right-hand-side of new `Poisson` propagator; the callable must return a V0 StencilVector !682
* `WeightedMassOperators.toarray` now works if extraction and boundary operators are identity !667
* replace `exit()` with `sys.exit(0)` for expected exit and `sys.exit(1)` for failures !696


### Bug fixes

* Resolve error in `reflect` particle boundary condition !672
* Resolve bug in assembly of basis projection operators !680

