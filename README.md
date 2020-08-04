# Welcome to STRUPHY 

The STRUPHY (STRUcture-Preserving HYbrid) code simulates kinetic-MHD (magneto-hydrodynamic) hybrid models of variuos flavours, combining conforming finite element methods (finite element exterior calculus, FEEC) with particle-in-cell (PIC) methods.

The STRUPHY code features:

- Linear, ideal MHD equations with nonlinear coupling to full-orbit Vlasov equation (6D), current-coupling approach
- Regular C<sup>1</sup>-mappings to single patch
- Exact conservation of div**B**=0 and of magnetic helicity, reagardless of grid spacing and mapping
- Exact energy balance, reagardless of grid spacing and mapping
- Control variate method for PIC (optional)
- Implicit time stepping with operator splitting
- OpenMP parallelization of PIC

The low-level routines feature:

- B-spline bases and commuting projectors (inter- and histopolation) for the 3D de Rham complex
- Periodic and Dirichlet boundary conditions
- Local projection operators based on quasi-interpolation (optional)

STRUPHY is a [python](https://www.python.org/) code that uses [pyccel](https://github.com/pyccel/pyccel) to accelerate to Fortran speed.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

What things you need to install the software and how to install them

- [Pyccel: https://github.com/pyccel/pyccel](https://github.com/pyccel/pyccel)

We recommend installation in development mode: 

python3 -m pip install --user -e .

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

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

