"""The ``struphy backend`` command: choose the cunumpy array backend (NumPy on the CPU or CuPy on the GPU).

cunumpy picks its backend from the environment variable ``CUNUMPY_BACKEND`` when it is imported. ``struphy backend
set cupy`` stores the choice in struphy's state file; ``import struphy`` then sets ``CUNUMPY_BACKEND`` from it
unless the variable is already set, and the profiling driver exports it in every job script it writes.
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
        ``get`` prints the stored backend, the environment variable and the backend cunumpy actually uses in this
        process. ``set`` stores ``backend`` in the state file. ``unset`` removes it, back to the default ``numpy``.

    backend : {"numpy", "cupy"}, optional
        With ``set``, the backend to store.

    value : bool
        With ``get``, print only the stored backend, for use in shell scripts: ``export CUNUMPY_BACKEND=$(struphy
        backend get --value)``.
    """
    state = utils.read_state()

    if action == "get":
        stored = configured_backend(state)
        if value:
            print(stored)
            return
        import cunumpy as xp

        print(f"stored backend (struphy backend set): {stored}" + ("" if STATE_KEY in state else " (default)"))
        print(f"{ENV_VAR}: {os.environ.get(ENV_VAR, '(not set)')}")
        print(f"backend in this process: {xp.get_backend()}")
    elif action == "set":
        if backend not in BACKENDS:
            raise ValueError(f"'struphy backend set' needs a backend, one of {BACKENDS}; got {backend!r}")
        state[STATE_KEY] = backend
        utils.save_state(state)
        print(f"stored backend: {backend}")
        print(f"'import struphy' now sets {ENV_VAR}={backend} unless the variable is already set,")
        print(f"and the profiling driver exports {ENV_VAR}={backend} in its job scripts.")
        env = os.environ.get(ENV_VAR)
        if env is not None and env != backend:
            print(f"Note: {ENV_VAR}={env} is set in this shell and takes precedence; unset it or export {backend}.")
    elif action == "unset":
        state.pop(STATE_KEY, None)
        utils.save_state(state)
        print(f"stored backend removed; the default is {DEFAULT_BACKEND}")
    else:
        raise ValueError(f"Unknown action {action!r}")
