from struphy.post_processing.output import open_output


def struphy_output(
    action: str,
    path: str,
    physical: bool = False,
    parallel: bool = False,
    format: str = "markdown",
    directory: str | None = None,
):
    """Inspect, post-process or report a completed simulation output.

    Parameters
    ----------
    action : {"info", "keys", "pproc", "report"}
        ``info`` lists the evaluable products with descriptions, ``keys`` prints one key per line,
        ``pproc`` materializes the post-processed products, and ``report`` writes a data report.

    path : str
        The simulation output directory.

    physical : bool
        With ``pproc``, also create physical field components.

    parallel : bool
        With ``pproc``, post-process on all ranks of ``MPI.COMM_WORLD``.

    format : {"markdown", "html"}
        With ``report``, the report format.

    directory : str, optional
        With ``report``, where to write it (default ``post_processing/report/``).
    """
    out = open_output(path)
    if action == "info":
        out.info()
    elif action == "keys":
        for key in out.keys():
            print(key)
    elif action == "pproc":
        out.pproc(physical=physical, parallel=parallel)
    elif action == "report":
        print(out.report(directory, format=format))
    else:
        raise ValueError(f"Unknown action {action!r}")
