# Welcome to Hylife!

*The Python Finite Element library for structure-preserving fluid-kinetic hybrid simulations.*

Currently, there is 1 code delivered with the Hylife repository:

- STRUPHY_3D (STRUcture-Preserving HYbrid code for MHD-kinetic current coupling)

# Requirements

* Working gfortran compiler
```
sudo apt install gfortran
```

* [Python3: python.org](https://python.org)

* Python packages:
```
pip3 install numpy
pip3 install mpi4py
pip3 install h5py
pip3 install PyYAML
pip3 install scipy
pip3 install matplotlib
python3 -m pip install pyccel==0.10.1
```

These can be installed with
```
python3 -m pip install -r requirements.txt
```

* In case there are problems with the installation of mpi4py type
```
sudo apt install libopenmpi-dev
```
<!---
## Installing Pyccel

In order to be able to execute `pyccel` globally you need to add its path to your `$PATH` variable:

```
export PATH="$PATH:$HOME/.local/bin"
```

If you want to add the path permanently, add the above line to your `.bashrc` file in `$HOME`.
In order to test the installation go to an arbitrary directory and type `pyccel --version`. You should see something like

```
pyccel 0.9.16 from ...your_pyccel_location.../pyccel/pyccel (python 3.8)
```
-->

Specifics for the HPC system `cobra` at IPP:

- use the module `anaconda/3/2020.02` for installing pyccel and loading mpi4py.



# Installing Hylife
```
git clone https://gitlab.mpcdf.mpg.de/clapp/hylife.git
```

# Setting up STRUPHY

In order to get the STRUPHY code running perform the following steps:

1. Create a folder `my_output` for STRUPHY output. Output of specific simulations e.g. `sim_1`, `sim_2`, etc. will be stored in `my_output/sim_1`, `my_output/sim_2`, etc..
  
2. In your hylife repository execute
  ```
  cp -r simulations/template_python path_to_my_output/sim_1
  ```
  This copies a simulation template from your `hylife` repository to your simulation folder `sim_1`.
  
3. In your hylife repository, in the file `run_STRUPHY_3D.sh` set the correct paths in lines 5 and 6
  ```
  all_sim=path_to_my_output
  run_dir=sim_1
  ```
  
4. In your hylife repository run STRUPHY with
  ```
  ./run_STRUPHY_3D.sh
  ```

<!---
## Running tests

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
-->
