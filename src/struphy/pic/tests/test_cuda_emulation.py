"""CPU emulation of every CUDA kernel, compared with its pyccel kernel (runs without a GPU).

Runs the cases of ``cuda_parity_cases.PARITY_CASES``, like the GPU parity test.
"""

import cunumpy as xp
import numpy as np
import pytest
from cunumpy.kernel_testing import emulation_compiler

from struphy.geometry.tests import spline_mapping_cases
from struphy.pic.tests.cuda_emulation import emulate_struct_kernel
from struphy.pic.tests.cuda_parity_cases import PARITY_CASES, argument_arrays
from struphy.pic.tests.kernel_test_args import N_GEOMETRY_DOMAINS
from struphy.pic.tests.test_cuda_parity import CUDA_KERNELS

requires_compiler = pytest.mark.skipif(emulation_compiler() is None, reason="no C++ compiler")


@requires_compiler
@pytest.mark.parametrize("name", list(PARITY_CASES))
def test_emulated_parity(name):
    """The emulated CUDA kernel writes the declared outputs like the pyccel kernel, and nothing else."""
    kernel, spec = CUDA_KERNELS[name], PARITY_CASES[name]
    outputs = kernel.host_kernel.outputs
    with xp.use_backend("numpy"):
        for case in spec.cases:
            host_args = spec.build(case)
            emulated_args = spec.build(case)
            before = argument_arrays(emulated_args)
            kernel(*host_args)
            n_threads = spec.n_threads(emulated_args) if spec.n_threads else None
            emulate_struct_kernel(kernel.cuda_kernel, *emulated_args, n_threads=n_threads)
            host, emulated = argument_arrays(host_args, outputs), argument_arrays(emulated_args, outputs)
            for key, value in argument_arrays(emulated_args).items():
                if key not in emulated:
                    np.testing.assert_array_equal(value, before[key], err_msg=f"{name}{case}: {key} is not an output")
            for key in host:
                np.testing.assert_allclose(
                    emulated[key], host[key], rtol=spec.rtol, atol=spec.atol, err_msg=f"{name}{case}: {key}"
                )


@requires_compiler
@pytest.mark.parametrize("degree", range(1, 9))
def test_emulated_der_splines(degree):
    """The device b_splines_slim and b_der_splines_slim, emulated on the CPU, agree with pyccel."""
    from cunumpy.kernels import CudaKernel

    from struphy.geometry.tests.spline_mapping_cases import DER_SPLINES_SOURCE, der_splines_case
    from struphy.utils.cuda_arguments import CUDA_OPTIONS

    knots, pts, expected = der_splines_case(degree)
    out = np.zeros(expected.shape)
    kernel = CudaKernel(DER_SPLINES_SOURCE, "evaluate_der_splines", **CUDA_OPTIONS)
    emulate_struct_kernel(kernel, knots, len(knots), degree, pts, out, len(pts), n_threads=len(pts))
    np.testing.assert_allclose(out, expected, rtol=1e-13, atol=1e-13)


@requires_compiler
@pytest.mark.parametrize("name", list(spline_mapping_cases.DOMAINS))
def test_emulated_spline_mappings(name):
    """The device spline mappings (kind_map 0-2) and their Jacobians, emulated on the CPU, agree with pyccel."""
    args_domain = spline_mapping_cases.host_domain(name).args_domain
    etas = spline_mapping_cases.points()
    expected = spline_mapping_cases.expected(args_domain, etas)
    out = np.zeros(expected.size)
    emulate_struct_kernel(
        spline_mapping_cases.make_kernel(),
        *spline_mapping_cases.flat_inputs(etas),
        args_domain,
        out,
        out.size,
        n_threads=out.size,
    )
    np.testing.assert_allclose(out.reshape(expected.shape), expected, rtol=1e-13, atol=1e-13)


@requires_compiler
@pytest.mark.parametrize("domain_index", range(N_GEOMETRY_DOMAINS))
@pytest.mark.parametrize("avoid_round_off", [False, True])
def test_emulated_metric_helpers(domain_index, avoid_round_off):
    """test_device_helpers.test_metric_helpers by CPU emulation, for the analytic and the spline mappings."""
    from struphy.pic.tests.test_device_helpers import metric_helper_case

    kernel, inputs, args_domain, expected = metric_helper_case(domain_index, avoid_round_off)
    out = np.zeros(len(expected))
    emulate_struct_kernel(kernel, *inputs, args_domain, out, len(out), n_threads=len(out))
    np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-10)


@requires_compiler
@pytest.mark.parametrize("domain_index", range(N_GEOMETRY_DOMAINS))
def test_emulated_transform_helpers(domain_index):
    """test_device_helpers.test_transform_helpers by CPU emulation, for the analytic and the spline mappings."""
    from struphy.pic.tests.test_device_helpers import transform_helper_case

    kernel, inputs, args_domain, expected = transform_helper_case(domain_index)
    out = np.zeros(len(expected))
    emulate_struct_kernel(kernel, *inputs, args_domain, out, len(out), n_threads=len(out))
    np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-10)
