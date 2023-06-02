## Version 1.9.9

* Add `dims_mask` to grid parameters; lets you choose spatial directions to decompose with MPI, e.g. `dims_mask : [True, True, False]` does not decompose the third direction. !321
* Improved batch run mode; for batch script runs, the batch script is now first copied to the output folder and submitted in this output folder. With this, the files `sim.out`, `sim.err` and `struphy.out` are written in the output folder, and also the batch script information is not lost. !322
* A new function `print_plasma_params` is implemented; print volume averaged plasma parameters(characteristics) for each species of the model. (When there is at least one species). !323

    ```
    Global parameters:
        - plasma volume
        - transit length
        - magnetic field

    Species dependent parameters:
        - mass
        - charge
        - density
        - pressure
        - thermal energy
        - thermal speed
        - thermal frequency
        - cyclotron frequency
        - larmour radius
        - rho/L
        - plasma frequency

    in case of MHD species
        - alfven speed
        - alfven frequency
    ```
* Time resolution flag in output data and post processing; new flag `-s, --save-step` in struphy run command to enable skipping in output data, e.g. `struphy run Maxwell -s 4` saves output data every fourth time step. New flag `-s, --step` to enable skipping in creation of post-processing data, e.g. `struphy pproc -d sim_1 -s 2` creates post-processing data (grids, evaluated fields, vtk files, etc.) for every second step in the hdf5 files. !326
* Reform Particles Class; new abstract methods:
`draw_markers`, `svol` and `s0`. !327  