"""Gyro-, disc- and ring-averaged spline evaluations and depositions at a marker, for gyrokinetic kernels.

For a gyrocentre at the logical position :math:`\\boldsymbol \\eta` with gyroradius :math:`\\rho` (in units of the
physical length), the gyro-average and the disc average of a field :math:`Q` are (Remmerswaal, Hatzky,
Sonnendrücker, J. Plasma Phys. 91, E127 (2025), Eqs. (4.16) and (4.49))

.. math::

    \\langle Q \\rangle = \\frac{1}{2\\pi} \\int_0^{2\\pi} Q(\\mathbf R + \\boldsymbol \\rho(\\theta))\\, \\textnormal d\\theta\\,, \\qquad
    \\langle\\langle Q \\rangle\\rangle = \\frac{1}{\\pi} \\int_0^1 \\int_0^{2\\pi} \\varsigma\\, Q(\\mathbf R + \\varsigma \\boldsymbol \\rho(\\theta))\\, \\textnormal d\\theta\\, \\textnormal d\\varsigma\\,,

with :math:`\\boldsymbol \\rho(\\theta) = \\rho (\\cos\\theta\\, \\hat{\\mathbf e}_1 + \\sin\\theta\\, \\hat{\\mathbf e}_2)`
in the plane perpendicular to the background magnetic field. The gyro-points are mapped to logical coordinates
with the inverse Jacobian at the gyrocentre, :math:`\\boldsymbol \\eta_k = \\boldsymbol \\eta + \\varsigma \\rho\\, DF^{-1} (\\cos\\theta_k \\hat{\\mathbf e}_1 + \\sin\\theta_k \\hat{\\mathbf e}_2)`
(exact for affine mappings, :math:`\\mathcal O(\\varepsilon_B)` otherwise), and wrapped into the unit cube.

The angle is discretized with ``n_gyro`` equidistant points (``n_gyro = 0`` or ``rho = 0`` give the point
evaluation, i.e. the zero-Larmor-radius limit), the radius with the nodes ``disc_s`` and weights ``disc_w``
(``disc_s = [1.0]``, ``disc_w = [1.0]`` for the ring, i.e. the gyro-average).
"""

from typing import Final

from numpy import cos, empty, mod, pi, shape, sin, sqrt
from pyccel.decorators import pure, stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.accumulation.particle_to_mat_kernels as particle_to_mat_kernels
from struphy.bsplines.evaluation_kernels_3d import (
    eval_0form_spline_mpi,
    eval_1form_spline_mpi,
    eval_2form_spline_mpi,
    get_spans,
)
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments


@pure
def gyro_radius(epsilon: float, mu: float, abs_b0: float) -> float:
    r"""Gyroradius :math:`\rho = |\varepsilon| \sqrt{2 \mu / B_0}` in units of the length unit
    (:math:`v_\perp = \sqrt{2 \mu B_0}`, :math:`\rho = |\varepsilon| v_\perp / B_0`)."""
    if abs_b0 <= 0.0:
        return 0.0
    return abs(epsilon) * sqrt(2.0 * mu / abs_b0)


@stack_array("dfm", "df_inv", "df_inv_t", "b_form", "b_cart", "e1", "e2", "a")
def gyro_frame(
    eta1: float,
    eta2: float,
    eta3: float,
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    unit_b1_1: "float[:,:,:]",
    unit_b1_2: "float[:,:,:]",
    unit_b1_3: "float[:,:,:]",
    d1: "float[:]",
    d2: "float[:]",
):
    r"""Logical displacements ``d1``, ``d2`` per unit gyroradius along two orthonormal directions
    :math:`\hat{\mathbf e}_1, \hat{\mathbf e}_2` perpendicular to the background unit magnetic field at the
    gyrocentre: :math:`\mathbf d_i = DF^{-1} \hat{\mathbf e}_i`.

    The unit field :math:`\mathbf b_0 = DF^{-\top} \hat{\mathbf b}^1_0` is evaluated from its 1-form coefficients;
    :math:`\hat{\mathbf e}_1` is the normalized cross product of :math:`\mathbf b_0` with the Cartesian axis least
    aligned with it, :math:`\hat{\mathbf e}_2 = \mathbf b_0 \times \hat{\mathbf e}_1`.
    """
    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)
    b_form = empty(3, dtype=float)
    b_cart = empty(3, dtype=float)
    e1 = empty(3, dtype=float)
    e2 = empty(3, dtype=float)
    a = empty(3, dtype=float)

    evaluation_kernels.df(eta1, eta2, eta3, args_domain, dfm)
    linalg_kernels.matrix_inv(dfm, df_inv)
    linalg_kernels.transpose(df_inv, df_inv_t)

    span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)
    eval_1form_spline_mpi(span1, span2, span3, args_derham, unit_b1_1, unit_b1_2, unit_b1_3, b_form)
    linalg_kernels.matrix_vector(df_inv_t, b_form, b_cart)

    norm_b = sqrt(b_cart[0] ** 2 + b_cart[1] ** 2 + b_cart[2] ** 2)
    if norm_b == 0.0:
        d1[:] = 0.0
        d2[:] = 0.0
        return
    b_cart /= norm_b

    # the Cartesian axis least aligned with b_0
    a[:] = 0.0
    if abs(b_cart[0]) <= abs(b_cart[1]) and abs(b_cart[0]) <= abs(b_cart[2]):
        a[0] = 1.0
    elif abs(b_cart[1]) <= abs(b_cart[2]):
        a[1] = 1.0
    else:
        a[2] = 1.0

    linalg_kernels.cross(b_cart, a, e1)
    norm_e1 = sqrt(e1[0] ** 2 + e1[1] ** 2 + e1[2] ** 2)
    e1 /= norm_e1
    linalg_kernels.cross(b_cart, e1, e2)

    linalg_kernels.matrix_vector(df_inv, e1, d1)
    linalg_kernels.matrix_vector(df_inv, e2, d2)


def gyro_point(
    eta1: float,
    eta2: float,
    eta3: float,
    d1: "Final[float[:]]",
    d2: "Final[float[:]]",
    radius: float,
    theta: float,
    out: "float[:]",
):
    """The logical position of the gyro-point at angle ``theta`` and radius ``radius`` (wrapped into the unit cube)."""
    ct = cos(theta)
    st = sin(theta)
    out[0] = mod(eta1 + radius * (ct * d1[0] + st * d2[0]), 1.0)
    out[1] = mod(eta2 + radius * (ct * d1[1] + st * d2[1]), 1.0)
    out[2] = mod(eta3 + radius * (ct * d1[2] + st * d2[2]), 1.0)


@stack_array("pt", "tmp")
def eval_1form_gyro(
    eta1: float,
    eta2: float,
    eta3: float,
    d1: "float[:]",
    d2: "float[:]",
    rho: float,
    n_gyro: int,
    disc_s: "float[:]",
    disc_w: "float[:]",
    args_derham: "DerhamArguments",
    coeffs_1: "float[:,:,:]",
    coeffs_2: "float[:,:,:]",
    coeffs_3: "float[:,:,:]",
    out: "float[:]",
):
    """Ring/disc average of a 1-form spline field at a gyrocentre (point evaluation if ``n_gyro == 0`` or ``rho == 0``)."""
    pt = empty(3, dtype=float)
    tmp = empty(3, dtype=float)

    if n_gyro == 0 or rho == 0.0:
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)
        eval_1form_spline_mpi(span1, span2, span3, args_derham, coeffs_1, coeffs_2, coeffs_3, out)
        return

    out[:] = 0.0
    n_disc = shape(disc_s)[0]
    for j in range(n_disc):
        radius = rho * disc_s[j]
        for k in range(n_gyro):
            theta = 2.0 * pi * k / n_gyro
            gyro_point(eta1, eta2, eta3, d1, d2, radius, theta, pt)
            span1, span2, span3 = get_spans(pt[0], pt[1], pt[2], args_derham)
            eval_1form_spline_mpi(span1, span2, span3, args_derham, coeffs_1, coeffs_2, coeffs_3, tmp)
            out += disc_w[j] / n_gyro * tmp


@stack_array("pt", "tmp")
def eval_2form_gyro(
    eta1: float,
    eta2: float,
    eta3: float,
    d1: "float[:]",
    d2: "float[:]",
    rho: float,
    n_gyro: int,
    disc_s: "float[:]",
    disc_w: "float[:]",
    args_derham: "DerhamArguments",
    coeffs_1: "float[:,:,:]",
    coeffs_2: "float[:,:,:]",
    coeffs_3: "float[:,:,:]",
    out: "float[:]",
):
    """Ring/disc average of a 2-form spline field at a gyrocentre (point evaluation if ``n_gyro == 0`` or ``rho == 0``)."""
    pt = empty(3, dtype=float)
    tmp = empty(3, dtype=float)

    if n_gyro == 0 or rho == 0.0:
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)
        eval_2form_spline_mpi(span1, span2, span3, args_derham, coeffs_1, coeffs_2, coeffs_3, out)
        return

    out[:] = 0.0
    n_disc = shape(disc_s)[0]
    for j in range(n_disc):
        radius = rho * disc_s[j]
        for k in range(n_gyro):
            theta = 2.0 * pi * k / n_gyro
            gyro_point(eta1, eta2, eta3, d1, d2, radius, theta, pt)
            span1, span2, span3 = get_spans(pt[0], pt[1], pt[2], args_derham)
            eval_2form_spline_mpi(span1, span2, span3, args_derham, coeffs_1, coeffs_2, coeffs_3, tmp)
            out += disc_w[j] / n_gyro * tmp


@stack_array("pt")
def eval_0form_gyro(
    eta1: float,
    eta2: float,
    eta3: float,
    d1: "float[:]",
    d2: "float[:]",
    rho: float,
    n_gyro: int,
    disc_s: "float[:]",
    disc_w: "float[:]",
    args_derham: "DerhamArguments",
    coeffs: "float[:,:,:]",
) -> float:
    """Ring/disc average of a 0-form spline field at a gyrocentre (point evaluation if ``n_gyro == 0`` or ``rho == 0``)."""
    pt = empty(3, dtype=float)

    if n_gyro == 0 or rho == 0.0:
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)
        return eval_0form_spline_mpi(span1, span2, span3, args_derham, coeffs)

    out = 0.0
    n_disc = shape(disc_s)[0]
    for j in range(n_disc):
        radius = rho * disc_s[j]
        for k in range(n_gyro):
            theta = 2.0 * pi * k / n_gyro
            gyro_point(eta1, eta2, eta3, d1, d2, radius, theta, pt)
            span1, span2, span3 = get_spans(pt[0], pt[1], pt[2], args_derham)
            out += disc_w[j] / n_gyro * eval_0form_spline_mpi(span1, span2, span3, args_derham, coeffs)
    return out


@stack_array("pt")
def deposit_1form_gyro(
    eta1: float,
    eta2: float,
    eta3: float,
    d1: "float[:]",
    d2: "float[:]",
    rho: float,
    n_gyro: int,
    disc_s: "float[:]",
    disc_w: "float[:]",
    args_derham: "DerhamArguments",
    vec_1: "float[:,:,:]",
    vec_2: "float[:,:,:]",
    vec_3: "float[:,:,:]",
    fill_1: float,
    fill_2: float,
    fill_3: float,
):
    """Deposit a 1-form filling with the ring/disc average of the basis functions at a gyrocentre
    (the transpose of :func:`eval_1form_gyro`)."""
    pt = empty(3, dtype=float)

    if n_gyro == 0 or rho == 0.0:
        particle_to_mat_kernels.vec_fill_b_v1(
            args_derham, eta1, eta2, eta3, vec_1, vec_2, vec_3, fill_1, fill_2, fill_3
        )
        return

    n_disc = shape(disc_s)[0]
    for j in range(n_disc):
        radius = rho * disc_s[j]
        for k in range(n_gyro):
            theta = 2.0 * pi * k / n_gyro
            gyro_point(eta1, eta2, eta3, d1, d2, radius, theta, pt)
            fac = disc_w[j] / n_gyro
            particle_to_mat_kernels.vec_fill_b_v1(
                args_derham, pt[0], pt[1], pt[2], vec_1, vec_2, vec_3, fac * fill_1, fac * fill_2, fac * fill_3
            )


@stack_array("pt")
def deposit_0form_gyro(
    eta1: float,
    eta2: float,
    eta3: float,
    d1: "float[:]",
    d2: "float[:]",
    rho: float,
    n_gyro: int,
    disc_s: "float[:]",
    disc_w: "float[:]",
    args_derham: "DerhamArguments",
    vec: "float[:,:,:]",
    fill: float,
):
    """Deposit a 0-form filling with the ring/disc average of the basis functions at a gyrocentre
    (the transpose of :func:`eval_0form_gyro`)."""
    pt = empty(3, dtype=float)

    if n_gyro == 0 or rho == 0.0:
        particle_to_mat_kernels.vec_fill_b_v0(args_derham, eta1, eta2, eta3, vec, fill)
        return

    n_disc = shape(disc_s)[0]
    for j in range(n_disc):
        radius = rho * disc_s[j]
        for k in range(n_gyro):
            theta = 2.0 * pi * k / n_gyro
            gyro_point(eta1, eta2, eta3, d1, d2, radius, theta, pt)
            particle_to_mat_kernels.vec_fill_b_v0(args_derham, pt[0], pt[1], pt[2], vec, disc_w[j] / n_gyro * fill)
