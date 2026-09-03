from dataclasses import dataclass

import cunumpy as xp
import numpy as np
from scipy.fft import irfft, rfft

from struphy.feec.psydac_derham import Derham
from struphy.io.options import LiteralOptions
from struphy.pic.accumulation.filter_kernels import apply_three_point_filter_3d


@dataclass
class FilterParameters:
    """Parameters for the AccumFilter class"""

    use_filter: LiteralOptions.OptsFilter | None = None
    modes: tuple[int, ...] = (1,)
    repeat: int = 1
    alpha: float = 0.5


class AccumFilter:
    """
    Callable filter that applies one of:
      - 'fourier_in_tor'
      - 'three_point'
      - 'hybrid' (three_point, then fourier_in_tor)
    """

    def __init__(self, params: FilterParameters, derham: Derham, space_id: str):
        self._params = params if params is not None else FilterParameters()
        self._derham = derham
        self._space_id = space_id

        self._form = derham.space_to_form[space_id]
        self._form_int = 0 if self._form == "v" else int(self._form)

    @property
    def params(self) -> FilterParameters:
        return self._params

    @property
    def derham(self):
        """Discrete Derham complex on the logical unit cube."""
        return self._derham

    @property
    def space_id(self):
        """Space identifier for the matrix/vector (H1, Hcurl, Hdiv, L2 or H1vec) to be accumulated into."""
        return self._space_id

    @property
    def form(self):
        """p-form("0", "1", "2", "3") to be accumulated into."""
        return self._form

    @property
    def form_int(self):
        """Integer notation of p-form("0", "1", "2", "3") to be accumulated into."""
        return self._form_int

    def __call__(self, vec):
        """
        Apply the chosen filter to `vec` in-place and return it.

        Parameters
        ----------
        vec : BlockVector
            Accumulated vector object.
        """
        use = self.params.use_filter
        if use is None:
            return vec  # nothing to do

        if use == "fourier_in_tor":
            self._apply_toroidal_fourier_filter(vec, self._params.modes)

        elif use == "three_point":
            self._apply_three_point(vec, repeat=self._params.repeat, alpha=self._params.alpha)

        elif use == "hybrid":
            self._apply_three_point(vec, repeat=self._params.repeat, alpha=self._params.alpha)
            self._apply_toroidal_fourier_filter(vec, self._params.modes)

        else:
            raise NotImplementedError("The type of filter must be 'fourier_in_tor', 'three_point', or 'hybrid'.")

        return vec

    def _yield_dir_components(self, vec):
        """
        Yields (axis, comp_vec, starts, ends) for each directions.
        - For scalar accumulations ('H1','L2'): yields (0, vec, starts, ends).
        - Otherwise: yields (axis, vec[axis], starts, ends) for axis=0,1,2.
        """
        if self.space_id in ("H1", "L2"):
            starts = self.derham.coeff_spaces[self.form].starts
            ends = self.derham.coeff_spaces[self.form].ends

            yield 0, vec, starts, ends

        else:
            for axis in range(3):
                starts = self.derham.coeff_spaces[self.form][axis].starts
                ends = self.derham.coeff_spaces[self.form][axis].ends

                yield axis, vec[axis], starts, ends

    def _apply_three_point(self, vec, repeat: int, alpha: float):
        """
        Applying three point smoothing filter to the spline coefficients of the accumulated vector (``._data`` of the StencilVector):

        Parameters
        ----------
        vec : StencilVector | BlockVector

        repeat : int
            Number of repeatition.

        alpha : float
            Alpha factor of the smoothing filter.

        """

        for _ in range(repeat):
            for axis, comp, starts, ends in self._yield_dir_components(vec):
                apply_three_point_filter_3d(
                    comp._data,
                    axis,
                    self.form_int,
                    xp.array(self.derham.num_elements),
                    xp.array(self.derham.spl_kind),
                    xp.array(self.derham.degree),
                    xp.array(starts),
                    xp.array(ends),
                    alpha=alpha,
                )

            vec.update_ghost_regions()

    def _apply_toroidal_fourier_filter(self, vec, modes: tuple[int, ...]):
        """
        Applying fourier filter to the spline coefficients of the accumulated vector (toroidal direction).

        Parameters
        ----------
        vec : BlockVector

        modes : tuple[int, ...]
            Mode numbers which are not filtered out.
        """

        # Host-only throughout: this is a per-(i,j)-line loop over scipy.fft calls (no
        # batched-2D API used here), never vectorized even under NumPy, so `modes`/`pn`/
        # `ir` are plain NumPy/Python (used only as Python loop bounds and fancy-index
        # arrays -- under the CuPy backend `xp.empty(...)` would make `ir` a CuPy array,
        # and `range(ir[0])` on a device scalar raises TypeError).
        tor_num_elements = self.derham.num_elements[2]
        modes = np.asarray(modes, dtype=int)

        assert tor_num_elements >= 2 * int(modes.max()), "num_elements[2] must be at least 2*max(modes)"
        assert self.derham.domain_decomposition.nprocs[2] == 1, "No domain decomposition along toroidal direction"

        pn = np.asarray(self.derham.degree, dtype=int)

        # rfft output length
        if (tor_num_elements % 2) == 0:
            vec_temp = np.zeros(int(tor_num_elements / 2) + 1, dtype=complex)
        else:
            vec_temp = np.zeros(int((tor_num_elements - 1) / 2) + 1, dtype=complex)

        for _, comp, starts, ends in self._yield_dir_components(vec):
            ir = [int(ends[i] + 1 - starts[i]) for i in range(3)]

            # Under the CuPy backend, comp._data lives on the device; pulled to host once
            # per component and pushed back once, rather than paying a device round-trip
            # on every one of the ir[0]*ir[1] lines below (also correct on the NumPy
            # backend, where to_numpy/asarray are no-ops).
            data = xp.to_numpy(comp._data)

            # filter along toroidal index (k direction)
            for i in range(ir[0]):
                ii = pn[0] + i
                for j in range(ir[1]):
                    jj = pn[1] + j

                    # forward FFT along toroidal line
                    line = rfft(data[ii, jj, pn[2] : pn[2] + ir[2]])
                    vec_temp[:] = 0
                    vec_temp[modes] = line[modes]  # keep selected modes only

                    # inverse FFT back to real space, write in-place
                    data[ii, jj, pn[2] : pn[2] + ir[2]] = irfft(vec_temp, n=tor_num_elements)

            comp._data[...] = xp.asarray(data)

            comp.update_ghost_regions()
