from struphy.post_processing.post_processing_tools import PostProcessor
from struphy.post_processing.run_output import RunOutput

PlottingData = RunOutput


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
    force: bool = False,
) -> RunOutput:
    """Process a completed run and return its lazy :class:`RunOutput`."""
    if sim is not None:
        sim.pproc(step=step, celldivide=celldivide, physical=physical,
                  guiding_center=guiding_center, classify=classify,
                  create_vtk=create_vtk, force=force, load=True)
        return RunOutput(sim=sim)
    if path_out is None:
        raise ValueError("path_out or sim is required")
    processor = PostProcessor(path_out=path_out)
    processor.process(step=step, celldivide=celldivide, physical=physical,
                      guiding_center=guiding_center, classify=classify,
                      create_vtk=create_vtk, force=force)
    return RunOutput(path_out=path_out)


__all__ = ["PostProcessor", "RunOutput", "PlottingData", "post_process"]
