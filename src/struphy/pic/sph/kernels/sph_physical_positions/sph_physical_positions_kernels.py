"""Physical marker positions for SPH, see :func:`~struphy.pic.sph_eval_kernels.physical_positions`."""

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
from struphy.kernel_arguments.pusher_args_kernels import DomainArguments, MarkerArguments
from struphy.pic.sph_eval_kernels import physical_positions


def sph_physical_positions(
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    holes: "bool[:]",
    x_idx: "int",
):
    r"""Store :math:`\mathbf x_p = F(\boldsymbol\eta_p)` of all markers (including ghost particles) in
    ``markers[x_idx:x_idx + 3, :]``, see :func:`~struphy.pic.sph_eval_kernels.physical_positions`.

    Parameters
    ----------
    args_markers : MarkerArguments
        Marker arguments.

    args_domain : DomainArguments
        Arguments of the mapping.

    holes : bool[:]
        ``True`` for holes.

    x_idx : int
        First of the three marker columns for the physical position.
    """
    physical_positions(args_markers, args_domain, holes, x_idx)
