try:
    import amrex.space3d as amr
except ImportError:
    amr = None


class Amrex:
    def __init__(self):
        if amr:
            # initialize pyAMReX

            amr.initialize(
                [
                    # print AMReX status messages
                    "amrex.verbose=1",
                    # # throw exceptions and create core dumps instead of
                    # # AMReX backtrace files: allows to attach to
                    # # debuggers
                    "amrex.throw_exception=0",
                    "amrex.signal_handling=1",
                ]
            )
        else:
            ModuleNotFoundError("pyAMReX must be installed")

    def finalize(self):
        amr.finalize()


def detect_amrex_gpu():
    try:
        import amrex.space3d as amr

        if amr.Config.have_gpu:
            import cupy as xp
        else:
            import numpy as xp
    except ImportError:
        amr = None
        try:
            import cupy as xp
        except ImportError:
            import numpy as xp
    return amr, xp
