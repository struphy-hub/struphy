# Repository

Struphy has two protected branches, **main** and **devel**. 
Nobody can push directly to these branches.

The **main** branch holds the current release of the code. 

**devel** is the branch for developers. Feature branches must be checked out and merged into **devel**.


# Dependency Bounds On PRs

Pull requests into **devel** are checked for stale dependency upper bounds in `pyproject.toml`.

The policy is intentionally narrow:

1. Only `project.dependencies` are checked.
2. From `project.optional-dependencies`, only `phys` and `mpi` is checked.
3. Patch-only updates are ignored.
4. A pull request fails when a newer **major** or **minor** stable release exists on PyPI beyond the declared upper bound.

When the check fails, the CI summary prints local remediation commands. In short, run the checker locally, run `python utils/update_dependency_bounds.py` on that report, and commit the updated `pyproject.toml`.


# Releases

Happen when pushed to **main**.


# Forking

Please create a **public fork** to be able to merge your code into Struphy!

You can create feature branches in your forked repo and create merge requests into the original Struphy repo.


# Contact

* [Mailing list](https://listserv.gwdg.de/mailman/listinfo/struphy)
* [MatrixChat developer's channel](https://matrix.to/#/!wqjcJpsUvAbTPOUXen:mpg.de?via=mpg.de&via=academiccloud.de)
* [Issue tracker](https://github.com/struphy-hub/struphy/issues) 
* [LinkedIn](https://www.linkedin.com/company/struphy/)
* [stefan.possanner@ipp.mpg.de](mailto:spossann@ipp.mpg.de)
* [max.lindqvist@ipp.mpg.de](mailto:max.lindqvist@ipp.mpg.de)
* [xin.wang@ipp.mpg.de](mailto:xin.wang@ipp.mpg.de)
