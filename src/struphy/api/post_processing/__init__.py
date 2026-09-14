from struphy.post_processing.post_processing_tools import PlottingData, PostProcessor


def post_process(
    sim=None,
    path_out: str = None,
    *,
    step: int = 1,
    celldivide=1,
    physical: bool = False,
    guiding_center: bool = False,
    classify: bool = False,
    create_vtk: bool = True,
    force: bool = True,
) -> PlottingData:
    """Post-process a completed run and return its loaded plotting data.

    This is the convenient serial entry point for the common process-then-load
    workflow. Use :class:`PostProcessor` and :class:`PlottingData` separately when
    processing under MPI or when the processed files should not be loaded into
    memory immediately.

    Parameters are the same as :meth:`PostProcessor.process`; identify the run with
    either ``sim`` or ``path_out``.
    """
    if sim is not None:
        return sim.pproc(
            step=step,
            celldivide=celldivide,
            physical=physical,
            guiding_center=guiding_center,
            classify=classify,
            create_vtk=create_vtk,
            force=force,
            load=True,
        )

    processor = PostProcessor(sim=sim, path_out=path_out)
    processor.process(
        step=step,
        celldivide=celldivide,
        physical=physical,
        guiding_center=guiding_center,
        classify=classify,
        create_vtk=create_vtk,
        force=force,
    )

    data = PlottingData(sim=sim, path_out=path_out)
    data.load()
    return data


__all__ = ["PostProcessor", "PlottingData", "post_process"]
