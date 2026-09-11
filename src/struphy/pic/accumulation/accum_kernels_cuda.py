"""Hand-written CUDA replacements for select accumulation (particle-to-grid
deposition) kernels, used only under ``ARRAY_BACKEND=cupy``.

Unlike the pusher kernels in :mod:`~struphy.pic.pushing.pusher_kernels_cuda`
(each marker only ever writes to its own row -- embarrassingly parallel, no
cross-thread interaction), accumulation kernels *scatter* every marker's
contribution into a shared grid array (:func:`~struphy.pic.accumulation.filler_kernels.fill_vec`'s
``vec[i1, i2, i3] += ...``): many markers whose (p+1)^3 local basis-function
support overlaps the same grid cell write to the same memory location. The
CPU kernel handles this by running the marker loop strictly sequentially (its
OpenMP ``reduction`` pragma is commented out in the source specifically
because of this race). The GPU port instead uses ``atomicAdd`` -- one thread
per marker, same as the pushers, but the grid write goes through an atomic
rather than a plain store. Double-precision ``atomicAdd`` is natively
supported on every CUDA compute capability this codebase targets (>= 6.0),
so no software fallback is needed.

Currently covered: :func:`~struphy.pic.accumulation.accum_kernels.charge_density_0form`,
used by :class:`~struphy.propagators.push_deterministic_diffusion.PushDeterministicDiffusion`
every step to build the (H^1) density field consumed by
:func:`~struphy.pic.pushing.pusher_kernels_cuda.push_deterministic_diffusion_stage_general_gpu`.
This one needs no domain-mapping Jacobian at all (the H^1 filling weight is
just the marker weight), so it reuses only the B-spline evaluation device
functions, not the geometry-mapping ones.
"""
from struphy.cuda import load_cuda_source

_CHARGE_DENSITY_0FORM_SRC = load_cuda_source(__file__, "accum_kernels_cuda/_charge_density_0form_src.cu")

_charge_density_0form_kernel = None


def _get_charge_density_0form_kernel():
    global _charge_density_0form_kernel
    if _charge_density_0form_kernel is None:
        import cupy as cp

        _charge_density_0form_kernel = cp.RawKernel(_CHARGE_DENSITY_0FORM_SRC, "charge_density_0form_cuda")
    return _charge_density_0form_kernel


def charge_density_0form_gpu(
    markers,
    weight_idx: int,
    pn: tuple[int, int, int],
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts: tuple[int, int, int],
    vec_dev,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.accumulation.accum_kernels.charge_density_0form`.

    ``markers`` is the host marker array, transferred to the device once per
    call (matching the pusher kernels' round-trip pattern). ``vec_dev`` is
    the target :class:`~feectools.linalg.stencil.StencilVector`'s ``._data``
    -- already device-resident under CuPy and already zeroed by the caller
    (:meth:`~struphy.pic.accumulation.particles_to_grid.AccumulatorVector._accumulate`
    always does ``dat[:] = 0.0`` before invoking the kernel), so this
    function only needs to add to it, not read markers back afterward: the
    caller reads ``vec_dev`` directly since it was written in place.
    """
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    dev_markers = markers
    threads = 256
    blocks = (n_markers + threads - 1) // threads
    _get_charge_density_0form_kernel()(
        (blocks,),
        (threads,),
        (
            dev_markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(weight_idx),
            np.int32(pn[0]),
            np.int32(pn[1]),
            np.int32(pn[2]),
            tn1_dev,
            np.int32(tn1_dev.shape[0]),
            tn2_dev,
            np.int32(tn2_dev.shape[0]),
            tn3_dev,
            np.int32(tn3_dev.shape[0]),
            np.int32(starts[0]),
            np.int32(starts[1]),
            np.int32(starts[2]),
            vec_dev,
            np.int32(vec_dev.shape[1]),
            np.int32(vec_dev.shape[2]),
        ),
    )


# ---------------------------------------------------------------------------
# linear_vlasov_ampere: accumulates into a symmetric V1 -> V1 block matrix
# (mat11, mat12, mat13, mat22, mat23, mat33) plus a V1 vector (vec1, vec2,
# vec3), using DF^-1(eta_p) @ v_p at each marker -- unlike
# charge_density_0form this needs the full domain-mapping Jacobian, so the
# kernel source below is prefixed with pusher_kernels_cuda's
# _GENERAL_GEOMETRY_SRC (df_dispatch_dev and friends) rather than
# duplicating it.
#
# The row/column basis combinations for the 6 matrix blocks and the fill
# formulas mirror struphy.pic.accumulation.particle_to_mat_kernels.m_v_fill_b_v1_symm
# exactly (which itself calls filler_kernels.fill_mat_vec/fill_mat) --
# fill_mat_vec_dev/fill_mat_dev below are direct ports of those two.
# ---------------------------------------------------------------------------

_LINEAR_VLASOV_AMPERE_EXTRA_SRC = load_cuda_source(__file__, "accum_kernels_cuda/_linear_vlasov_ampere_extra_src.cu")


def _linear_vlasov_ampere_source():
    from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

    return _GENERAL_GEOMETRY_SRC + _LINEAR_VLASOV_AMPERE_EXTRA_SRC


# ---------------------------------------------------------------------------
# vlasov_maxwell: same symmetric V1 -> V1 6-block-matrix-plus-vector fill as
# linear_vlasov_ampere (reuses fill_mat_vec_dev/fill_mat_dev from
# _LINEAR_VLASOV_AMPERE_EXTRA_SRC above), but with a different filling:
# A_p = w_p * G^-1(eta_p) (the metric inverse, not an outer product of
# velocity) and B_p = w_p * DF^-1(eta_p) v_p -- no f0_values/s0 involved, so
# unlike linear_vlasov_ampere this one can't hit the inf/nan-from-div-by-s0
# path. Also note: the CPU reference only skips markers[ip,0]==-1.0 (no
# markers[ip,-1]==-2.0 check), unlike linear_vlasov_ampere -- ported as-is.
# ---------------------------------------------------------------------------

_VLASOV_MAXWELL_EXTRA_SRC = load_cuda_source(__file__, "accum_kernels_cuda/_vlasov_maxwell_extra_src.cu")


def _vlasov_maxwell_source():
    from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

    return _GENERAL_GEOMETRY_SRC + _LINEAR_VLASOV_AMPERE_EXTRA_SRC + _VLASOV_MAXWELL_EXTRA_SRC


_vlasov_maxwell_kernel = None


def _get_vlasov_maxwell_kernel():
    global _vlasov_maxwell_kernel
    if _vlasov_maxwell_kernel is None:
        import cupy as cp

        _vlasov_maxwell_kernel = cp.RawKernel(_vlasov_maxwell_source(), "vlasov_maxwell_cuda")
    return _vlasov_maxwell_kernel


def vlasov_maxwell_gpu(
    markers,
    kind_map: int,
    params_dev,
    pn: tuple[int, int, int],
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts: tuple[int, int, int],
    mat11_dev,
    mat12_dev,
    mat13_dev,
    mat22_dev,
    mat23_dev,
    mat33_dev,
    vec1_dev,
    vec2_dev,
    vec3_dev,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.accumulation.accum_kernels.vlasov_maxwell`. Same
    calling convention as :func:`linear_vlasov_ampere_gpu`, minus
    ``f0_values`` (this kernel doesn't need a background distribution).
    """
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    dev_markers = markers
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    def dims(a):
        return (
            np.int32(a.shape[1]),
            np.int32(a.shape[2]),
            np.int32(a.shape[3]),
            np.int32(a.shape[4]),
            np.int32(a.shape[5]),
        )

    _get_vlasov_maxwell_kernel()(
        (blocks,),
        (threads,),
        (
            dev_markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(kind_map),
            params_dev,
            np.int32(pn[0]),
            np.int32(pn[1]),
            np.int32(pn[2]),
            tn1_dev,
            np.int32(tn1_dev.shape[0]),
            tn2_dev,
            np.int32(tn2_dev.shape[0]),
            tn3_dev,
            np.int32(tn3_dev.shape[0]),
            np.int32(starts[0]),
            np.int32(starts[1]),
            np.int32(starts[2]),
            mat11_dev,
            mat12_dev,
            mat13_dev,
            mat22_dev,
            mat23_dev,
            mat33_dev,
            vec1_dev,
            vec2_dev,
            vec3_dev,
            *dims(mat11_dev),
            *dims(mat12_dev),
            *dims(mat13_dev),
            *dims(mat22_dev),
            *dims(mat23_dev),
            *dims(mat33_dev),
            np.int32(vec1_dev.shape[1]),
            np.int32(vec1_dev.shape[2]),
            np.int32(vec2_dev.shape[1]),
            np.int32(vec2_dev.shape[2]),
            np.int32(vec3_dev.shape[1]),
            np.int32(vec3_dev.shape[2]),
        ),
    )


_linear_vlasov_ampere_kernel = None


def _get_linear_vlasov_ampere_kernel():
    global _linear_vlasov_ampere_kernel
    if _linear_vlasov_ampere_kernel is None:
        import cupy as cp

        _linear_vlasov_ampere_kernel = cp.RawKernel(_linear_vlasov_ampere_source(), "linear_vlasov_ampere_cuda")
    return _linear_vlasov_ampere_kernel


def linear_vlasov_ampere_gpu(
    markers,
    kind_map: int,
    params_dev,
    f0_values_dev,
    pn: tuple[int, int, int],
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts: tuple[int, int, int],
    mat11_dev,
    mat12_dev,
    mat13_dev,
    mat22_dev,
    mat23_dev,
    mat33_dev,
    vec1_dev,
    vec2_dev,
    vec3_dev,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.accumulation.accum_kernels.linear_vlasov_ampere`.

    ``markers`` is the host marker array, round-tripped through the device
    once per call (this kernel only reads markers, never writes them back).
    ``params_dev``/``f0_values_dev`` and all ``mat*_dev``/``vec*_dev`` arrays
    are expected to already be device-resident (cached once by the caller);
    the ``mat*_dev``/``vec*_dev`` arrays must already be zeroed, matching
    :meth:`~struphy.pic.accumulation.particles_to_grid.Accumulator._accumulate`'s
    ``dat[:] = 0.0`` reset before the kernel call.
    """
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    dev_markers = markers
    f0_values_dev = cp.ascontiguousarray(f0_values_dev)
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    def dims(a):
        return (
            np.int32(a.shape[1]),
            np.int32(a.shape[2]),
            np.int32(a.shape[3]),
            np.int32(a.shape[4]),
            np.int32(a.shape[5]),
        )

    _get_linear_vlasov_ampere_kernel()(
        (blocks,),
        (threads,),
        (
            dev_markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(kind_map),
            params_dev,
            f0_values_dev,
            np.int32(pn[0]),
            np.int32(pn[1]),
            np.int32(pn[2]),
            tn1_dev,
            np.int32(tn1_dev.shape[0]),
            tn2_dev,
            np.int32(tn2_dev.shape[0]),
            tn3_dev,
            np.int32(tn3_dev.shape[0]),
            np.int32(starts[0]),
            np.int32(starts[1]),
            np.int32(starts[2]),
            mat11_dev,
            mat12_dev,
            mat13_dev,
            mat22_dev,
            mat23_dev,
            mat33_dev,
            vec1_dev,
            vec2_dev,
            vec3_dev,
            *dims(mat11_dev),
            *dims(mat12_dev),
            *dims(mat13_dev),
            *dims(mat22_dev),
            *dims(mat23_dev),
            *dims(mat33_dev),
            np.int32(vec1_dev.shape[1]),
            np.int32(vec1_dev.shape[2]),
            np.int32(vec2_dev.shape[1]),
            np.int32(vec2_dev.shape[2]),
            np.int32(vec3_dev.shape[1]),
            np.int32(vec3_dev.shape[2]),
        ),
    )


# ---------------------------------------------------------------------------
# cc_lin_mhd_6d_1: accumulates into the 3 antisymmetric off-diagonal blocks
# (mat12, mat13, mat23) of a V_u -> V_u matrix, no vector, where V_u is
# whichever of H1vec/Hcurl/Hdiv the propagator's ``u_space`` option selects
# (runtime int ``basis_u`` in {0, 1, 2}). All 3 branches ultimately do a
# 3-block antisymmetric fill using fill_mat_dev (from
# _LINEAR_VLASOV_AMPERE_EXTRA_SRC above) with the row/col basis-degree
# combination matching struphy's mat_fill_v0vec_asym (basis_u=0, N-N-N both
# sides), mat_fill_v1_asym (basis_u=1, D-N-N/N-D-N/N-N-D -- same combination
# already used for linear_vlasov_ampere/vlasov_maxwell's off-diagonal
# blocks) and mat_fill_v2_asym (basis_u=2, Hdiv's N-D-D/D-N-D/D-D-N). Since
# basis_u is one value per kernel LAUNCH (not per marker), the branch is
# warp-coherent -- every thread takes the same path, no divergence cost.
# basis_u=0 needs no domain Jacobian at all (see the CPU reference: dfm is
# computed unconditionally there but only actually used by basis_u 1/2), so
# df_dispatch_dev is only called inside the basis_u==1/2 branches here.
# ---------------------------------------------------------------------------

_CC_LIN_MHD_6D_1_SRC = load_cuda_source(__file__, "accum_kernels_cuda/_cc_lin_mhd_6d_1_src.cu")


def _cc_lin_mhd_6d_1_source():
    from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

    return _GENERAL_GEOMETRY_SRC + _LINEAR_VLASOV_AMPERE_EXTRA_SRC + _CC_LIN_MHD_6D_1_SRC


_cc_lin_mhd_6d_1_kernel = None


def _get_cc_lin_mhd_6d_1_kernel():
    global _cc_lin_mhd_6d_1_kernel
    if _cc_lin_mhd_6d_1_kernel is None:
        import cupy as cp

        _cc_lin_mhd_6d_1_kernel = cp.RawKernel(_cc_lin_mhd_6d_1_source(), "cc_lin_mhd_6d_1_cuda")
    return _cc_lin_mhd_6d_1_kernel


def cc_lin_mhd_6d_1_gpu(
    markers,
    kind_map: int,
    params_dev,
    pn: tuple[int, int, int],
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts: tuple[int, int, int],
    b2_1_dev,
    b2_2_dev,
    b2_3_dev,
    basis_u: int,
    scale_mat: float,
    boundary_cut: float,
    mat12_dev,
    mat13_dev,
    mat23_dev,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.accumulation.accum_kernels.cc_lin_mhd_6d_1`.
    ``b2_*_dev`` are the Hdiv (2-form) magnetic field FE coefficients,
    already device-resident.
    """
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    dev_markers = markers
    b2_1_dev = cp.ascontiguousarray(b2_1_dev)
    b2_2_dev = cp.ascontiguousarray(b2_2_dev)
    b2_3_dev = cp.ascontiguousarray(b2_3_dev)
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    def dims(a):
        return (
            np.int32(a.shape[1]),
            np.int32(a.shape[2]),
            np.int32(a.shape[3]),
            np.int32(a.shape[4]),
            np.int32(a.shape[5]),
        )

    _get_cc_lin_mhd_6d_1_kernel()(
        (blocks,),
        (threads,),
        (
            dev_markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(kind_map),
            params_dev,
            np.int32(pn[0]),
            np.int32(pn[1]),
            np.int32(pn[2]),
            tn1_dev,
            np.int32(tn1_dev.shape[0]),
            tn2_dev,
            np.int32(tn2_dev.shape[0]),
            tn3_dev,
            np.int32(tn3_dev.shape[0]),
            np.int32(starts[0]),
            np.int32(starts[1]),
            np.int32(starts[2]),
            b2_1_dev,
            np.int32(b2_1_dev.shape[1]),
            np.int32(b2_1_dev.shape[2]),
            b2_2_dev,
            np.int32(b2_2_dev.shape[1]),
            np.int32(b2_2_dev.shape[2]),
            b2_3_dev,
            np.int32(b2_3_dev.shape[1]),
            np.int32(b2_3_dev.shape[2]),
            np.int32(basis_u),
            np.float64(scale_mat),
            np.float64(boundary_cut),
            mat12_dev,
            mat13_dev,
            mat23_dev,
            *dims(mat12_dev),
            *dims(mat13_dev),
            *dims(mat23_dev),
        ),
    )


# ---------------------------------------------------------------------------
# cc_lin_mhd_6d_2: like cc_lin_mhd_6d_1 (B2 field evaluation, bx() matrix,
# runtime basis_u in {0, 1, 2} selecting H1vec/Hcurl/Hdiv), but fills the
# full symmetric 6-block matrix plus a vector (like linear_vlasov_ampere /
# vlasov_maxwell), not just the 3 antisymmetric off-diagonal blocks. Basis
# combinations per branch (matching struphy's m_v_fill_v0vec_symm /
# m_v_fill_v1_symm / m_v_fill_v2_symm): basis_u=0 uses N-N-N everywhere
# (all 6 matrix blocks AND the vector); basis_u=1 is the same D-N-N/N-D-N/
# N-N-D combination already used for linear_vlasov_ampere/vlasov_maxwell;
# basis_u=2 is Hdiv's N-D-D/D-N-D/D-D-N (same as cc_lin_mhd_6d_1's
# basis_u=2). Per the CPU reference, basis_u=0 and 2 only ever need df_inv
# (g_inv is computed there but never actually used in those two branches --
# not replicated here); only basis_u=1 needs the full g_inv = DF^-1 DF^-T.
# ---------------------------------------------------------------------------

_CC_LIN_MHD_6D_2_SRC = load_cuda_source(__file__, "accum_kernels_cuda/_cc_lin_mhd_6d_2_src.cu")


def _cc_lin_mhd_6d_2_source():
    from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

    return _GENERAL_GEOMETRY_SRC + _LINEAR_VLASOV_AMPERE_EXTRA_SRC + _CC_LIN_MHD_6D_2_SRC


_cc_lin_mhd_6d_2_kernel = None


def _get_cc_lin_mhd_6d_2_kernel():
    global _cc_lin_mhd_6d_2_kernel
    if _cc_lin_mhd_6d_2_kernel is None:
        import cupy as cp

        _cc_lin_mhd_6d_2_kernel = cp.RawKernel(_cc_lin_mhd_6d_2_source(), "cc_lin_mhd_6d_2_cuda")
    return _cc_lin_mhd_6d_2_kernel


def cc_lin_mhd_6d_2_gpu(
    markers,
    kind_map: int,
    params_dev,
    pn: tuple[int, int, int],
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts: tuple[int, int, int],
    b2_1_dev,
    b2_2_dev,
    b2_3_dev,
    basis_u: int,
    scale_mat: float,
    scale_vec: float,
    boundary_cut: float,
    mat11_dev,
    mat12_dev,
    mat13_dev,
    mat22_dev,
    mat23_dev,
    mat33_dev,
    vec1_dev,
    vec2_dev,
    vec3_dev,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.accumulation.accum_kernels.cc_lin_mhd_6d_2`.
    ``b2_*_dev`` are the Hdiv (2-form) magnetic field FE coefficients,
    already device-resident.
    """
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    dev_markers = markers
    b2_1_dev = cp.ascontiguousarray(b2_1_dev)
    b2_2_dev = cp.ascontiguousarray(b2_2_dev)
    b2_3_dev = cp.ascontiguousarray(b2_3_dev)
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    def dims(a):
        return (
            np.int32(a.shape[1]),
            np.int32(a.shape[2]),
            np.int32(a.shape[3]),
            np.int32(a.shape[4]),
            np.int32(a.shape[5]),
        )

    _get_cc_lin_mhd_6d_2_kernel()(
        (blocks,),
        (threads,),
        (
            dev_markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(kind_map),
            params_dev,
            np.int32(pn[0]),
            np.int32(pn[1]),
            np.int32(pn[2]),
            tn1_dev,
            np.int32(tn1_dev.shape[0]),
            tn2_dev,
            np.int32(tn2_dev.shape[0]),
            tn3_dev,
            np.int32(tn3_dev.shape[0]),
            np.int32(starts[0]),
            np.int32(starts[1]),
            np.int32(starts[2]),
            b2_1_dev,
            np.int32(b2_1_dev.shape[1]),
            np.int32(b2_1_dev.shape[2]),
            b2_2_dev,
            np.int32(b2_2_dev.shape[1]),
            np.int32(b2_2_dev.shape[2]),
            b2_3_dev,
            np.int32(b2_3_dev.shape[1]),
            np.int32(b2_3_dev.shape[2]),
            np.int32(basis_u),
            np.float64(scale_mat),
            np.float64(scale_vec),
            np.float64(boundary_cut),
            mat11_dev,
            mat12_dev,
            mat13_dev,
            mat22_dev,
            mat23_dev,
            mat33_dev,
            vec1_dev,
            vec2_dev,
            vec3_dev,
            *dims(mat11_dev),
            *dims(mat12_dev),
            *dims(mat13_dev),
            *dims(mat22_dev),
            *dims(mat23_dev),
            *dims(mat33_dev),
            np.int32(vec1_dev.shape[1]),
            np.int32(vec1_dev.shape[2]),
            np.int32(vec2_dev.shape[1]),
            np.int32(vec2_dev.shape[2]),
            np.int32(vec3_dev.shape[1]),
            np.int32(vec3_dev.shape[2]),
        ),
    )


# ---------------------------------------------------------------------------
# pc_lin_mhd_6d_full / pc_lin_mhd_6d: accumulate a "pressure tensor" -- the
# same DF^-1(eta_p) DF^-T(eta_p) V1 -> V1 filling as vlasov_maxwell, but
# additionally scaled by every v_a*v_b product (a,b in x,y,z) of the marker
# velocity, giving one full symmetric 6-block matrix PER velocity-pair (6
# pairs: xx, xy, xz, yy, yz, zz -> 6*6=36 matrix arrays) plus one vector PER
# velocity-component (3*3=9 vector arrays) -- see
# particle_to_mat_kernels.m_v_fill_v1_pressure_full and
# filler_kernels.fill_mat_vec_pressure_full/fill_mat_pressure_full, which
# this is a direct port of (fill_mat_vec_pressure_full_dev/
# fill_mat_pressure_full_dev below are the CUDA equivalents, generalizing
# fill_mat_vec_dev/fill_mat_dev from a single mat/vec output to six/three).
#
# pc_lin_mhd_6d (no "_full") is the same accumulation restricted to the
# (x, y) "perpendicular" velocity plane only: 3 velocity-pairs (xx, xy, yy)
# and 2 velocity-components (x, y), i.e. 6*3=18 matrix arrays and 3*2=6
# vector arrays -- see m_v_fill_v1_pressure/fill_mat_vec_pressure/
# fill_mat_pressure. Both variants are called with the SAME 36+9=45 output
# arrays (the propagators share one call signature for _full and non-_full)
# but pc_lin_mhd_6d only ever writes the 24 "perp" ones -- the CPU reference
# leaves the other 21 untouched (at whatever the caller zeroed them to), and
# so does this port: pc_lin_mhd_6d_gpu accepts all 45 positionally (to match
# Accumulator._accumulate's ``*self._args_data`` unpacking) but only passes
# the 24 it needs into the CUDA launch.
#
# Both variants only differ from vlasov_maxwell's filling in the v_a*v_b
# scaling and in which marker column holds the weight: pc_lin_mhd_6d_full
# uses markers[ip, 8], pc_lin_mhd_6d uses markers[ip, 6] (matching the CPU
# reference exactly).
# ---------------------------------------------------------------------------

_SPATIAL_BLOCKS = ("11", "12", "13", "22", "23", "33")
_PC_PRESSURE_FILLERS_SRC = load_cuda_source(__file__, "accum_kernels_cuda/_pc_pressure_fillers_src.cu")
_PC_LIN_MHD_6D_FULL_SRC = load_cuda_source(__file__, "accum_kernels_cuda/pc_lin_mhd_6d_full.cu")
_PC_LIN_MHD_6D_SRC = load_cuda_source(__file__, "accum_kernels_cuda/pc_lin_mhd_6d.cu")


def _pc_lin_mhd_6d_full_source():
    from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

    return _GENERAL_GEOMETRY_SRC + _PC_PRESSURE_FILLERS_SRC + _PC_LIN_MHD_6D_FULL_SRC


def _pc_lin_mhd_6d_source():
    from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

    return _GENERAL_GEOMETRY_SRC + _PC_PRESSURE_FILLERS_SRC + _PC_LIN_MHD_6D_SRC


_pc_lin_mhd_6d_full_kernel = None
_pc_lin_mhd_6d_kernel = None


def _get_pc_lin_mhd_6d_full_kernel():
    global _pc_lin_mhd_6d_full_kernel
    if _pc_lin_mhd_6d_full_kernel is None:
        import cupy as cp

        _pc_lin_mhd_6d_full_kernel = cp.RawKernel(_pc_lin_mhd_6d_full_source(), "pc_lin_mhd_6d_full_cuda")
    return _pc_lin_mhd_6d_full_kernel


def _get_pc_lin_mhd_6d_kernel():
    global _pc_lin_mhd_6d_kernel
    if _pc_lin_mhd_6d_kernel is None:
        import cupy as cp

        _pc_lin_mhd_6d_kernel = cp.RawKernel(_pc_lin_mhd_6d_source(), "pc_lin_mhd_6d_cuda")
    return _pc_lin_mhd_6d_kernel


def _pc_lin_mhd_6d_launch(
    kernel,
    markers,
    kind_map: int,
    params_dev,
    pn,
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts,
    ep_scale: float,
    mat_args_45: dict,
    vec_args_45: dict,
    vel_pairs,
    vec_is,
):
    """Shared launch logic for pc_lin_mhd_6d_full_gpu/pc_lin_mhd_6d_gpu.
    ``mat_args_45``/``vec_args_45`` map every full-45-array name
    (``mat{sp}_{vel}`` / ``vec{mu}_{i}``) to its device array; only the
    subset named in ``vel_pairs``/``vec_is`` is actually passed to the
    kernel launch.
    """
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    dev_markers = markers
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    def dims(a):
        return (
            np.int32(a.shape[1]),
            np.int32(a.shape[2]),
            np.int32(a.shape[3]),
            np.int32(a.shape[4]),
            np.int32(a.shape[5]),
        )

    args = [
        dev_markers,
        np.int32(markers.shape[1]),
        np.int32(n_markers),
        np.int32(kind_map),
        params_dev,
        np.int32(pn[0]),
        np.int32(pn[1]),
        np.int32(pn[2]),
        tn1_dev,
        np.int32(tn1_dev.shape[0]),
        tn2_dev,
        np.int32(tn2_dev.shape[0]),
        tn3_dev,
        np.int32(tn3_dev.shape[0]),
        np.int32(starts[0]),
        np.int32(starts[1]),
        np.int32(starts[2]),
        np.float64(ep_scale),
    ]
    for vel in vel_pairs:
        for sp in _SPATIAL_BLOCKS:
            args.append(mat_args_45[f"mat{sp}_{vel}"])
    for i in vec_is:
        for mu in ("1", "2", "3"):
            args.append(vec_args_45[f"vec{mu}_{i}"])
    for sp in _SPATIAL_BLOCKS:
        args.extend(dims(mat_args_45[f"mat{sp}_11"]))
    for mu in ("1", "2", "3"):
        v = vec_args_45[f"vec{mu}_1"]
        args.extend((np.int32(v.shape[1]), np.int32(v.shape[2])))

    kernel((blocks,), (threads,), tuple(args))


def pc_lin_mhd_6d_full_gpu(
    markers,
    kind_map: int,
    params_dev,
    pn: tuple[int, int, int],
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts: tuple[int, int, int],
    ep_scale: float,
    *mat_and_vec_args,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.accumulation.accum_kernels.pc_lin_mhd_6d_full`.
    ``mat_and_vec_args`` are the 45 output arrays in the exact positional
    order of the CPU kernel's signature (36 matrix blocks: velocity-pair
    outer -- xx, xy, xz, yy, yz, zz -- spatial-block inner -- 11, 12, 13,
    22, 23, 33; then 9 vector blocks: velocity-component outer -- x, y, z
    -- spatial-component inner -- 1, 2, 3), matching
    ``Accumulator._args_data``'s construction for ``symmetry="pressure"``.
    """
    mat_args_45 = {
        f"mat{sp}_{vel}": mat_and_vec_args[k]
        for k, (vel, sp) in enumerate((vel, sp) for vel in _SPATIAL_BLOCKS for sp in _SPATIAL_BLOCKS)
    }
    vec_args_45 = {
        f"vec{mu}_{i}": mat_and_vec_args[36 + k]
        for k, (i, mu) in enumerate((i, mu) for i in ("1", "2", "3") for mu in ("1", "2", "3"))
    }
    _pc_lin_mhd_6d_launch(
        _get_pc_lin_mhd_6d_full_kernel(),
        markers,
        kind_map,
        params_dev,
        pn,
        tn1_dev,
        tn2_dev,
        tn3_dev,
        starts,
        ep_scale,
        mat_args_45,
        vec_args_45,
        _SPATIAL_BLOCKS,
        ("1", "2", "3"),
    )


def pc_lin_mhd_6d_gpu(
    markers,
    kind_map: int,
    params_dev,
    pn: tuple[int, int, int],
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts: tuple[int, int, int],
    ep_scale: float,
    *mat_and_vec_args,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.accumulation.accum_kernels.pc_lin_mhd_6d`. Same
    45-array positional convention as :func:`pc_lin_mhd_6d_full_gpu`
    (the propagator passes the identical 45-array signature for both), but
    -- matching the CPU reference exactly -- only the "perp" (x, y) subset
    (18 of the 36 matrix arrays, 6 of the 9 vector arrays) is ever written;
    the rest are left untouched (they stay at whatever
    ``Accumulator._accumulate``'s ``dat[:] = 0.0`` reset left them at).
    """
    mat_args_45 = {
        f"mat{sp}_{vel}": mat_and_vec_args[k]
        for k, (vel, sp) in enumerate((vel, sp) for vel in _SPATIAL_BLOCKS for sp in _SPATIAL_BLOCKS)
    }
    vec_args_45 = {
        f"vec{mu}_{i}": mat_and_vec_args[36 + k]
        for k, (i, mu) in enumerate((i, mu) for i in ("1", "2", "3") for mu in ("1", "2", "3"))
    }
    _pc_lin_mhd_6d_launch(
        _get_pc_lin_mhd_6d_kernel(),
        markers,
        kind_map,
        params_dev,
        pn,
        tn1_dev,
        tn2_dev,
        tn3_dev,
        starts,
        ep_scale,
        mat_args_45,
        vec_args_45,
        ("11", "12", "22"),
        ("1", "2"),
    )
