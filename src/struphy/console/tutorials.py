def struphy_tutorials(n=None):
    """
    Run Struphy simulation(s) for notebook tutorials. See `https://struphy.pages.mpcdf.de/struphy/sections/tutorials.html`_.
    Data is stored in the Struphy installation path under io/out/tutorial_<nr>.

    Parameters
    ----------
    n : int
        Number of specific tutorial simulation to run. If None, all tutorial simulations are run.
    """

    import struphy
    from struphy.tutorials import simulations

    libpath = struphy.__path__[0]

    sims = []
    if n is None:
        for word in dir(simulations):
            if 'tutorial' in word:
                sims += [word]
    else:
        assert n < 100
        sims += ['tutorial_' + str(n).zfill(2)]

    for sim in sims:
        func = getattr(simulations, sim)
        func()
