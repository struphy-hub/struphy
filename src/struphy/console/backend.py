"""The ``struphy backend`` command: choose the cunumpy array backend (NumPy on the CPU or CuPy on the GPU).

cunumpy picks its backend from the environment variable ``CUNUMPY_BACKEND`` when it is imported. ``struphy backend
set cupy`` stores the choice in struphy's state file, and ``import struphy`` then sets ``CUNUMPY_BACKEND`` to it,
overwriting whatever the shell had: the stored choice is the single source of truth. The profiling driver runs
``struphy backend set`` in every job script before the simulation.
"""

import os

from struphy.utils import utils

ENV_VAR = "CUNUMPY_BACKEND"
BACKENDS = ("numpy", "cupy")
DEFAULT_BACKEND = "numpy"
STATE_KEY = "backend"


def configured_backend(state: dict | None = None) -> str:
    """The backend stored with ``struphy backend set``, or the default ``"numpy"``."""
    if state is None:
        state = utils.read_state()
    backend = state.get(STATE_KEY, DEFAULT_BACKEND)
    if backend not in BACKENDS:
        raise ValueError(f"invalid backend {backend!r} in the state file; choose one of {BACKENDS}")
    return backend


def struphy_backend(action: str, backend: str | None = None, value: bool = False):
    """Show, store or forget the cunumpy backend of struphy runs.

    Parameters
    ----------
    action : {"get", "set", "unset"}
        ``get`` prints the stored backend and the backend cunumpy can provide on this machine. ``set`` stores
        ``backend`` in the state file. ``unset`` removes it; cunumpy then falls back to ``CUNUMPY_BACKEND`` from the
        shell, or NumPy.

    backend : {"numpy", "cupy"}, optional
        With ``set``, the backend to store.

    value : bool
        With ``get``, print only the stored backend, for use in shell scripts.
    """
    state = utils.read_state()

    if action == "get":
        stored = configured_backend(state)
        if value:
            print(stored)
            return
        import cunumpy as xp

        # what cunumpy could provide here: it falls back to NumPy when CuPy or CUDA is not functional
        effective = xp.get_backend()

        if STATE_KEY in state:
            print(f"Backend struphy runs use (struphy backend set):  {stored}")
        else:
            env = os.environ.get(ENV_VAR)
            if env is None:
                print(f"Backend struphy runs use:  {DEFAULT_BACKEND}  (nothing stored, cunumpy's default)")
            elif env in BACKENDS:
                print(f"Backend struphy runs use:  {env}  (nothing stored, from {ENV_VAR} in the shell)")
            else:
                print(
                    f"Backend struphy runs use:  {DEFAULT_BACKEND}  "
                    f"(nothing stored; {ENV_VAR}={env} in the shell is not a backend, cunumpy ignores it)"
                )
        print(f"Backend available on this machine:                {effective}")
        requested = stored if STATE_KEY in state else os.environ.get(ENV_VAR, DEFAULT_BACKEND)
        if requested in BACKENDS and effective != requested:
            print()
            print(
                f"Warning: {requested} is chosen but cunumpy can only provide {effective} here "
                f"(CuPy or CUDA is not functional on this machine). Runs on this machine use {effective}."
            )
    elif action == "set":
        if backend not in BACKENDS:
            raise ValueError(f"'struphy backend set' needs a backend, one of {BACKENDS}; got {backend!r}")
        state[STATE_KEY] = backend
        utils.save_state(state)
        print(f"struphy runs now use the {backend} backend ('import struphy' sets {ENV_VAR}={backend}).")
    elif action == "unset":
        state.pop(STATE_KEY, None)
        utils.save_state(state)
        print(f"stored backend removed; runs use {ENV_VAR} from the shell, or {DEFAULT_BACKEND}")
    else:
        raise ValueError(f"Unknown action {action!r}")
