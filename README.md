# Welcome to hylife!

*The Python Finite Element library for structure-preserving kinetic-fluid hybrid simulations.*

hylife provides

- discrete de Rham sequence in 1,2 or 3 dimensions
- tensor-product B-spline basis functions
- commuting projectors based on inter-/histopolation at Greville points
- grad, curl, div operators
- Fortran kernels generated via [pyccel](https://github.com/pyccel/pyccel)
- pullback and push-forward to mapped domain
- C^1 analytical mappings and C^1 IGA-compatible spline mappings
- particle sampling routines
- particle accumulation routines 
- basis evaluation routines

Currently, there is 1 code delivered with the hylife repository:

- STRUPHY (STRUcture-Preserving HYbrid code for MHD-kinetic current coupling)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

The requirements are the same as for Pyccel and can be found here:

- [Pyccel: https://github.com/pyccel/pyccel](https://github.com/pyccel/pyccel)

### Installing Pyccel

We recommend installation in development mode.
Choose a directory for the `pyccel` repository, go there and execute the following commands:

```
git clone https://github.com/pyccel/pyccel.git
cd pyccel
python3 -m pip install --user -e .
```

In order to be able to execute `pyccel` globally you need to add its path to your `$PATH` variable:

```
export PATH="$PATH:$HOME/.local/bin"
```

If you want to add the path permanently, add the above line to your `.bashrc` file in `$HOME`.
In order to test the installation go to an arbitrary directory and type `pyccel --version`. You should see something like

```
pyccel 0.9.16 from ...your_pyccel_location.../pyccel/pyccel (python 3.8)
```


### Installing Hylife

Choose a directory for your `hylife` repository and execute the following command there:

```
git clone https://gitlab.mpcdf.mpg.de/clapp/hylife.git
```

### Setting up STRUPHY

In order to get the STRUPHY code running perform the following steps:

1. Create a directory for your STRUPHY simulations. The absolute path to this directory will be called `$all_sim` in what follows.
   This directory need not be located in your hylife repository.
  
2. In your hylife repository execute
  ```
  cp -r simulations/example_analytical $all_sim/name_of_run
  ```
  This copies a simulation template from your `hylife` repository to your simulation folder located at `$all_sim` and creates a folder `name_of_run` for the current simulation.
  Simulation input, output and source files for the current simulation will be in `name_of_run`.
  
3. In your hylife repository, in the file `run.sh` set the correct paths in lines 5 and 6
  ```
  all_sim=$all_sim 
  run_dir=name_of_run
  ```
  
4. STRUPHY can now be run by executing `./run.sh` in your hylife repository. Results will be stored in `$all_sim/name_of_run`.
  

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

