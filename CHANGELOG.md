## Version 2.0.0

* Release of parallel Struphy 2
* Updated instructions for multipass virtual machine: added instructions for M1 Mac users, X11 forwarding, and connecting to VS Code. !335
* Updated `quickstart` section of documentation. !336
* New propagator `ShearAlfvenB1` to be used e.g. in model `LinearExtendedMHD`. !337
* Update Mac OS requirements in documentation. !338
* Model `LinearExtendedMHD` fully functional: added hall propagator, sonic ion propagator, sonic electron propagator and basis projection operator K3. !341
* New MHD equilibrium `AdhocTorusQPsi` based on an analytical expression for the safety factor profile in terms of the normalized polidal flux `q=q(psi_norm)`. !342
* Renamed velocity variables: `vx, vy, vz` --> `v1, v2, v3`. !343
* Improved post-processing and diagnostics. !344
* First draft of Vlasov-Maxwell with delta-f. !345
* First three notebook tutorials. !348
* Split `models.py` into four sub-files: `fluid.py`, `kinetic.py`, `hybrid.py`, `toy.py`. !350
* Analytical dispersion relation for model `LinearExtendedMHD`. !351
* Added file `test_mhd_equils.py` for testing all evaluation methods in the MHDequilibrium base class for different equilibrium-domain pairs. !352
* The new console option `struphy --set-io PATH` enables users to define a new default I/O path: The template folders `io/inp` and `io/batch` are copied to this path, and the folder `PATH/io/out/` is created.A command like `struphy run MODEL -i FOO -o BAR -b HUI` will target the new default I/O folder for input, output and batch script. !353
* Enable auto-relase in `gitlab-ci.yml`. !356