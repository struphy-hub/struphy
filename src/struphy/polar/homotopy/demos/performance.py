"""Assembly cost of the polar mass matrices against the spline degree (counterpart of PolarFEEC.jl
``examples/08_assembly_performance.jl`` for the Struphy :class:`~struphy.feec.psydac_derham.Derham` with
``polar_splines=True``).

PolarFEEC.jl assembles 3D masses by sum factorization (Julia threads); Struphy assembles the tensor-product weighted
mass matrices with its own (pyccel) kernels, and the polar matrices are applied as ``E M E^T`` with the polar
extraction operators. The study measures the assembly time of ``M0 ... M3`` and of one polar matrix-vector product on
the circular torus ``HollowTorus(a1=0)`` for degrees ``p = 1 ... 5``, and checks the volume ``2 pi^2 R0 a^2``.
"""

import time

import numpy as np


def study(degrees=(1, 2, 3, 4, 5), cells=(12, 12, 12), regularity=1, repeats=3, verbose=True):
    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.psydac_derham import Derham
    from struphy.geometry.domains import HollowTorus
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    R0, a = 3.0, 1.0
    domain = HollowTorus(a1=0.0, a2=a, R0=R0, tor_period=1)
    results = {}
    for p in degrees:
        options = DerhamOptions(
            degree=(p,) * 3,
            bcs=(("free", "free"), None, None),
            polar_splines=True,
            polar_regularity=min(regularity, p),
        )
        derham = Derham(TensorProductGrid(num_elements=cells), options, domain=domain)
        timings, products = {}, {}
        for name in ("M0", "M1", "M2", "M3"):
            best = np.inf
            for _ in range(repeats):
                mass = WeightedMassOperators(derham, domain)
                start = time.perf_counter()
                M = getattr(mass, name)
                x = M.domain.zeros()
                M.dot(x)  # the matrices are assembled on first use
                best = min(best, time.perf_counter() - start)
            timings[name] = best
            start = time.perf_counter()
            M.dot(x)
            products[name] = time.perf_counter() - start
        # volume (1, 1)_{L2}: the constant 1 has the 3-form proxy det(DF), and M3 has the weight 1 / det(DF)
        one = derham.P3(lambda e1, e2, e3: domain.jacobian_det(e1, e2, e3))
        volume = float(WeightedMassOperators(derham, domain).M3.dot(one).inner(one))
        results[p] = dict(assembly=timings, product=products, volume=volume)
        if verbose:
            print(
                f"p = {p}: assembly "
                + ", ".join(f"{k} {v:.2f} s" for k, v in timings.items())
                + f"; volume {volume:.6f} (exact {2 * np.pi**2 * R0 * a**2:.6f})"
            )
    return results


def figure(results):
    import matplotlib.pyplot as plt

    degrees = list(results)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for name in ("M0", "M1", "M2", "M3"):
        ax.semilogy(degrees, [results[p]["assembly"][name] for p in degrees], "o-", label=name)
    ax.set_xlabel("degree $p$")
    ax.set_ylabel("assembly time and first product [s]")
    ax.set_title("polar mass matrices, $12^3$ cells")
    ax.legend()
    fig.tight_layout()
    return fig
