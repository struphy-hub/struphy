# Repository

Struphy has two protected branches, **main** and **devel**. 
Nobody can push directly to these branches.

The **main** branch holds the current release of the code. 

**devel** is the branch for developers. Feature branches must be checked out and merged into **devel**.


# Dependency Bounds On PRs

Every pull request into **devel** runs the workflow `pr-dependency-bounds.yml`, which checks the upper bounds of the
dependencies in `pyproject.toml` with `utils/check_dependency_bounds.py`:

1. Only `project.dependencies` and, from `project.optional-dependencies`, the groups `phys` and `mpi` are checked.
2. A bound is outdated when a newer **major** or **minor** stable release exists on PyPI (patch-only updates are
   ignored) or when no upper bound is declared.
3. Exact pins (`==`, e.g. `ruff`) are deliberate and not checked.

When the check fails, the PR author refreshes the bounds and pushes (the job log and summary print the same commands):

```bash
python utils/check_dependency_bounds.py   # writes dependency-bounds-report.json
python utils/update_dependency_bounds.py  # sets the outdated bounds to <=latest in pyproject.toml
git add pyproject.toml
git commit -m "Update dependency bounds"
git push
```

The update rewrites only the changed requirement strings, so comments and formatting of `pyproject.toml` are kept.
The PR's CI then tests the new bounds.


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
