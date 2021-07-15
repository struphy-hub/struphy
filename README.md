# Welcome to Hylife!

*The Python Finite Element library for structure-preserving fluid-kinetic hybrid simulations.*

Currently, the following plasma codes are delivered with the Hylife repository:

- [STRUPHY_cc_lin_6D.py](https://gitlab.mpcdf.mpg.de/clapp/hylife/-/wikis/home/struphy_cc_lin_6d) 

# Requirements

* Linux or MacOS
* Non standard libraries: `libopenmpi-dev`
```
sudo apt install libopenmpi-dev
```
* Recommended: `virtualenv`
```
python3 -m pip install --user virtualenv
```

# Quickstart on Linux

```
$ git clone git@gitlab.mpcdf.mpg.de:clapp/hylife.git
$ virtualenv .venv
$ source .venv/bin/activate
$ python -m pip install -r requirements.txt
$ ./STRUPHY_init.sh
$ ./run_STRUPHY_cc_lin_6D.sh
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
