# Repository

Struphy has two protected branches, **main** and **devel**. 
Nobody can push directly to these branches.

The **main** branch holds the current release of the code. 

**devel** is the branch for developers. Feature branches must be checked out and merged into **devel**.


# Dependency Bounds On PRs

Pull requests into **devel** are checked for stale dependency upper bounds in `pyproject.toml`.

The policy is intentionally narrow:

1. Only `project.dependencies` are checked.
2. From `project.optional-dependencies`, only `phys` is checked.
3. Patch-only updates are ignored.
4. A pull request fails when a newer **major** or **minor** stable release exists on PyPI beyond the declared upper bound.

If the pull request branch lives in the main Struphy repository, the failing freshness check triggers a follow-up workflow that:

1. creates a clean environment,
2. runs `pip install -U --upgrade-strategy eager .[phys]`,
3. runs `python utils/set_release_dependencies.py --optional-group phys`,
4. commits the resulting `pyproject.toml` change back to the pull request branch.

That push should retrigger the pull request pipeline through the normal `synchronize` event.

For pull requests opened from forks, the freshness check still fails, but the repository workflow does **not** push back to the fork branch automatically.


# Releases

Releases should be prepared from code that has already passed the `PR - model tests in Container` workflow on a pull request into **devel**.

The tested dependency set is recorded automatically from the PR test environment after the workflow installs dependencies with `pip install -U -e ".[phys,mpi,doc]"`.

To prepare a release:

1. Merge the tested pull request into **devel**.
2. Run the `Prepare Release Dependency Bounds` workflow and select the source ref to release from.
3. The workflow resolves the merged PR into **devel**, finds the successful `PR - model tests in Container` run for that PR head commit, downloads the recorded dependency snapshot, updates `pyproject.toml`, and opens or updates a PR into **main**.
4. Review that PR, including `.github/release/tested-dependencies.json`, and merge it into **main**.
5. The existing publish workflows on **main** then build and publish from that reviewed commit.

Pull requests into **main** are checked against `.github/release/tested-dependencies.json` to ensure that the release bounds in `pyproject.toml` match the tested dependency snapshot.


# Forking

Please create a **public fork** to be able to merge your code into Struphy!

You can create feature branches in your forked repo and create merge requests into the original Struphy repo.


# Contact

* [Mailing list](https://listserv.gwdg.de/mailman/listinfo/struphy)
* [MatrixChat developer's channel](https://matrix.to/#/!wqjcJpsUvAbTPOUXen:mpg.de?via=mpg.de&via=academiccloud.de)
* [Issue tracker](https://github.com/struphy-hub/struphy/issues) 
* [LinkedIn](https://www.linkedin.com/company/struphy/)
* [stefan.possanner@ipp.mpg.de](mailto:spossann@ipp.mpg.de)
* [max.lindqvist@ipp.mpg.de@ipp.mpg.de](mailto:max.lindqvist@ipp.mpg.de)
* [xin.wang@ipp.mpg.de](mailto:xin.wang@ipp.mpg.de)
