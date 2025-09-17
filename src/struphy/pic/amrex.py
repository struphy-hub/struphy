try:
    import amrex.space3d as amr
except ImportError:
    amr = None


class Amrex:
    def __init__(self):
        if amr:
            # initialize AMReX

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
            raise ModuleNotFoundError("pyAMReX must be installed")

    def finalize(self):
        amr.finalize()
