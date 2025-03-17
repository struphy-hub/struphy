from numpy import abs, cos, exp, pi, sin, sqrt


###########################################
# Uni-variate kernels for tensor products #
###########################################
def trigonometric_uni(
    x: "float",
    h: "float",
) -> float:
    """Uni-variate kernel S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    if abs(x / h) <= 1.0:
        return 0.785398163397448 / h * cos(x / h * pi / 2.0)
    else:
        return 0.0


def grad_trigonometric_uni(
    x: "float",
    h: "float",
) -> float:
    """Derivative of S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    if abs(x / h) <= 1.0:
        return -(1.2337005501361697 / h**2) * sin(x / h * pi / 2.0)
    else:
        return 0.0


def gaussian_uni(
    x: "float",
    h: "float",
) -> float:
    """Uni-variate S(x, h) = 1/(sqrt(pi)*h/3) * exp(-(x**2/(h/3)**2) if |x|<1, 0 else."""
    if abs(x / h) <= 1.0:
        return 1 / (sqrt(pi) * h / 3) * exp(-(x**2) / (h / 3) ** 2)
    else:
        return 0.0


def grad_gaussian_uni(
    x: "float",
    h: "float",
) -> float:
    """Derivative of S(x, h) = 1/(sqrt(pi)*h/3) * exp(-(x**2/(h/3)**2) if |x|<1, 0 else."""
    if abs(x / h) <= 1.0:
        return -54 * x / (h**3 * sqrt(pi)) * exp(-(x**2) / (h / 3) ** 2)
    else:
        return 0.0


def linear_uni(
    x: "float",
    h: "float",
) -> float:
    """Uni-variate S(x, h) = (1 - x)/h if |x|<1, 0 else."""
    if abs(x / h) <= 1.0:
        return (1.0 - abs(x / h)) / h
    else:
        return 0.0


def grad_linear_uni(
    x: "float",
    h: "float",
) -> float:
    """Derivative of S(x, h) = (1 - x)/h if |x|<1, 0 else."""
    if abs(x / h) <= 1.0:
        if x > 0.0:
            return -(1 / h**2)
        else:
            return 1 / h**2
    else:
        return 0.0


##############
# 1d kernels #
##############
def trigonometric_1d(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """1d kernel S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    s1 = trigonometric_uni(r1, h1)
    return s1


def grad_trigonometric_1d(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """Derivative of S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    ds1 = grad_trigonometric_uni(r1, h1)
    return ds1


def gaussian_1d(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """1d kernel S(x, h) = 1/(sqrt(pi)*h/3) * exp(-(x**2/(h/3)**2) if |x|<1, 0 else."""
    s1 = gaussian_uni(r1, h1)
    return s1


def grad_gaussian_1d(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """Derivative of S(x, h) = 1/(sqrt(pi)*h/3) * exp(-(x**2/(h/3)**2) if |x|<1, 0 else."""
    ds1 = grad_gaussian_uni(r1, h1)
    return ds1


def linear_1d(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """1d kernel S(x, h) = (1 - x)/h if |x|<1, 0 else."""
    s1 = linear_uni(r1, h1)
    return s1


def grad_linear_1d(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """Derivative of S(x, h) = (1 - x)/h if |x|<1, 0 else."""
    ds1 = grad_linear_uni(r1, h1)
    return ds1


##############
# 2d kernels #
##############
def trigonometric_2d(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    s1 = trigonometric_uni(r1, h1)
    s2 = trigonometric_uni(r2, h2)
    return s1 * s2


def grad_trigonometric_2d_1(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """1st component of gradient of Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    ds1 = grad_trigonometric_uni(r1, h1)
    s2 = trigonometric_uni(r2, h2)
    return ds1 * s2


def grad_trigonometric_2d_2(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """2nd component of gradient of Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    s1 = trigonometric_uni(r1, h1)
    ds2 = grad_trigonometric_uni(r2, h2)
    return s1 * ds2


def gaussian_2d(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """Tensor product of kernels S(x, h) = 1/(sqrt(pi)*h/3) * exp(-(x**2/(h/3)**2) if |x|<1, 0 else."""
    s1 = gaussian_uni(r1, h1)
    s2 = gaussian_uni(r2, h2)
    return s1 * s2


def grad_gaussian_2d_1(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """1st component of gradient of Tensor product of kernels S(x, h) = 1/(sqrt(pi)*h/3) * exp(-(x**2/(h/3)**2) if |x|<1, 0 else."""
    ds1 = grad_gaussian_uni(r1, h1)
    s2 = gaussian_uni(r2, h2)
    return ds1 * s2


def grad_gaussian_2d_2(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """2nd component of gradient of Tensor product of kernels S(x, h) = 1/(sqrt(pi)*h/3) * exp(-(x**2/(h/3)**2) if |x|<1, 0 else."""
    s1 = gaussian_uni(r1, h1)
    ds2 = grad_gaussian_uni(r2, h2)
    return s1 * ds2


def linear_2d(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    s1 = linear_uni(r1, h1)
    s2 = linear_uni(r2, h2)
    return s1 * s2


def grad_linear_2d_1(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """1st component of gradient of Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    ds1 = grad_linear_uni(r1, h1)
    s2 = linear_uni(r2, h2)
    return ds1 * s2


def grad_linear_2d_2(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """2nd component of gradient of Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    s1 = linear_uni(r1, h1)
    ds2 = grad_linear_uni(r2, h2)
    return s1 * ds2


##############
# 3d kernels #
##############
def trigonometric_3d(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    s1 = trigonometric_uni(r1, h1)
    s2 = trigonometric_uni(r2, h2)
    s3 = trigonometric_uni(r3, h3)
    return s1 * s2 * s3


def grad_trigonometric_3d_1(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """1st component of gradient of Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    ds1 = grad_trigonometric_uni(r1, h1)
    s2 = trigonometric_uni(r2, h2)
    s3 = trigonometric_uni(r3, h3)
    return ds1 * s2 * s3


def grad_trigonometric_3d_2(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """2nd component of gradient of Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    s1 = trigonometric_uni(r1, h1)
    ds2 = grad_trigonometric_uni(r2, h2)
    s3 = trigonometric_uni(r3, h3)
    return s1 * ds2 * s3


def grad_trigonometric_3d_3(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """3rd component of gradient of Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    s1 = trigonometric_uni(r1, h1)
    s2 = trigonometric_uni(r2, h2)
    ds3 = grad_trigonometric_uni(r3, h3)
    return s1 * s2 * ds3


def gaussian_3d(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """Tensor product of kernels S(x, h) = 1/(sqrt(pi)*h/3) * exp(-(x**2/(h/3)**2) if |x|<1, 0 else."""
    s1 = gaussian_uni(r1, h1)
    s2 = gaussian_uni(r2, h2)
    s3 = gaussian_uni(r3, h3)
    return s1 * s2 * s3


def grad_gaussian_3d_1(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """1st component of gradient of Tensor product of kernels S(x, h) = 1/(sqrt(pi)*h/3) * exp(-(x**2/(h/3)**2) if |x|<1, 0 else."""
    ds1 = grad_gaussian_uni(r1, h1)
    s2 = gaussian_uni(r2, h2)
    s3 = gaussian_uni(r3, h3)
    return ds1 * s2 * s3


def grad_gaussian_3d_2(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """2nd component of gradient of Tensor product of kernels S(x, h) = 1/(sqrt(pi)*h/3) * exp(-(x**2/(h/3)**2) if |x|<1, 0 else."""
    s1 = gaussian_uni(r1, h1)
    ds2 = grad_gaussian_uni(r2, h2)
    s3 = gaussian_uni(r3, h3)
    return s1 * ds2 * s3


def grad_gaussian_3d_3(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """3rd component of gradient of Tensor product of kernels S(x, h) = 1/(sqrt(pi)*h/3) * exp(-(x**2/(h/3)**2) if |x|<1, 0 else."""
    s1 = gaussian_uni(r1, h1)
    s2 = gaussian_uni(r2, h2)
    ds3 = grad_gaussian_uni(r3, h3)
    return s1 * s2 * ds3


def linear_isotropic_3d(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """
    Smoothing kernel S(r,h) = C(h)F(r/h) with F(x) = 1-x if x<1, 0 else,
    and C(h)=3/(pi*h^3) is a normalization coefficient so the the kernel has unit integral.
    """
    r = sqrt(r1**2 + r2**2 + r3**2)
    h = h1
    if r / h > 1.0:
        return 0.0
    else:
        return (1.0 - r / h) / (1.0471975512 * h**3)


def grad_linear_isotropic_3d_1(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """
    1st component of gradient of S(r,h) = C(h)F(r/h) with F(x) = 1-x if x<1, 0 else,
    and C(h)=3/(pi*h^3) is a normalization coefficient so the the kernel has unit integral.
    """
    r = sqrt(r1**2 + r2**2 + r3**2)
    h = h1
    if r / h > 1.0:
        return 0.0
    elif r == 0.0:
        return -1 / h / (1.0471975512 * h**3)
    else:
        return -r1 / (r * h) / (1.0471975512 * h**3)


def grad_linear_isotropic_3d_2(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """
    1st component of gradient of S(r,h) = C(h)F(r/h) with F(x) = 1-x if x<1, 0 else,
    and C(h)=3/(pi*h^3) is a normalization coefficient so the the kernel has unit integral.
    """
    r = sqrt(r1**2 + r2**2 + r3**2)
    h = h1
    if r / h > 1.0:
        return 0.0
    elif r == 0.0:
        return -1 / h / (1.0471975512 * h**3)
    else:
        return -r2 / (r * h) / (1.0471975512 * h**3)


def grad_linear_isotropic_3d_3(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """
    1st component of gradient of S(r,h) = C(h)F(r/h) with F(x) = 1-x if x<1, 0 else,
    and C(h)=3/(pi*h^3) is a normalization coefficient so the the kernel has unit integral.
    """
    r = sqrt(r1**2 + r2**2 + r3**2)
    h = h1
    if r / h > 1.0:
        return 0.0
    elif r == 0.0:
        return -1 / h / (1.0471975512 * h**3)
    else:
        return -r3 / (r * h) / (1.0471975512 * h**3)


def linear_3d(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    s1 = linear_uni(r1, h1)
    s2 = linear_uni(r2, h2)
    s3 = linear_uni(r3, h3)
    return s1 * s2 * s3


def grad_linear_3d_1(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """1st component of gradient of Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    ds1 = grad_linear_uni(r1, h1)
    s2 = linear_uni(r2, h2)
    s3 = linear_uni(r3, h3)
    return ds1 * s2 * s3


def grad_linear_3d_2(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """2nd component of gradient of Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    s1 = linear_uni(r1, h1)
    ds2 = grad_linear_uni(r2, h2)
    s3 = linear_uni(r3, h3)
    return s1 * ds2 * s3


def grad_linear_3d_3(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """3rd component of gradient of Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    s1 = linear_uni(r1, h1)
    s2 = linear_uni(r2, h2)
    ds3 = grad_linear_uni(r3, h3)
    return s1 * s2 * ds3


############
# selector #
############
def smoothing_kernel(
    kernel_type: "int",
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """Each smoothing kernel is normalized to 1.

    The kernel type numbers must have 3 digits, where the last digit is reserved for the gradient;
    if a kernel has the type number n, the i-th components of its gradient has the number n + i.
    This means we have space for 99 kernels (and its gradient components) in principle.

    - 1d kernels <= 330
    - 2d kernels <= 660
    - 3d kernels >= 670

    If you add a kernel, make sure it is also added to :meth:`~struphy.pic.base.Particles.ker_dct`."""

    # 1d kernels
    if kernel_type == 100:
        out = trigonometric_1d(r1, r2, r3, h1, h2, h3)
    elif kernel_type == 101:
        out = grad_trigonometric_1d(r1, r2, r3, h1, h2, h3)

    elif kernel_type == 110:
        out = gaussian_1d(r1, r2, r3, h1, h2, h3)
    elif kernel_type == 111:
        out = grad_gaussian_1d(r1, r2, r3, h1, h2, h3)

    elif kernel_type == 120:
        out = linear_1d(r1, r2, r3, h1, h2, h3)
    elif kernel_type == 121:
        out = grad_linear_1d(r1, r2, r3, h1, h2, h3)

    # 2d kernels
    elif kernel_type == 340:
        out = trigonometric_2d(r1, r2, r3, h1, h2, h3)
    elif kernel_type == 341:
        out = grad_trigonometric_2d_1(r1, r2, r3, h1, h2, h3)
    elif kernel_type == 342:
        out = grad_trigonometric_2d_2(r1, r2, r3, h1, h2, h3)

    elif kernel_type == 350:
        out = gaussian_2d(r1, r2, r3, h1, h2, h3)
    elif kernel_type == 351:
        out = grad_gaussian_2d_1(r1, r2, r3, h1, h2, h3)
    elif kernel_type == 352:
        out = grad_gaussian_2d_2(r1, r2, r3, h1, h2, h3)

    elif kernel_type == 360:
        out = linear_2d(r1, r2, r3, h1, h2, h3)
    elif kernel_type == 361:
        out = grad_linear_2d_1(r1, r2, r3, h1, h2, h3)
    elif kernel_type == 362:
        out = grad_linear_2d_2(r1, r2, r3, h1, h2, h3)

    # 3d kernels
    elif kernel_type == 670:
        out = trigonometric_3d(r1, r2, r3, h1, h2, h3)
    elif kernel_type == 671:
        out = grad_trigonometric_3d_1(r1, r2, r3, h1, h2, h3)
    elif kernel_type == 672:
        out = grad_trigonometric_3d_2(r1, r2, r3, h1, h2, h3)
    elif kernel_type == 673:
        out = grad_trigonometric_3d_3(r1, r2, r3, h1, h2, h3)

    elif kernel_type == 680:
        out = gaussian_3d(r1, r2, r3, h1, h2, h3)
    elif kernel_type == 681:
        out = grad_gaussian_3d_1(r1, r2, r3, h1, h2, h3)
    elif kernel_type == 682:
        out = grad_gaussian_3d_2(r1, r2, r3, h1, h2, h3)
    elif kernel_type == 683:
        out = grad_gaussian_3d_3(r1, r2, r3, h1, h2, h3)

    elif kernel_type == 690:
        out = linear_isotropic_3d(r1, r2, r3, h1, h2, h3)
    elif kernel_type == 691:
        out = grad_linear_isotropic_3d_1(r1, r2, r3, h1, h2, h3)
    elif kernel_type == 692:
        out = grad_linear_isotropic_3d_2(r1, r2, r3, h1, h2, h3)
    elif kernel_type == 693:
        out = grad_linear_isotropic_3d_3(r1, r2, r3, h1, h2, h3)

    elif kernel_type == 700:
        out = linear_3d(r1, r2, r3, h1, h2, h3)
    elif kernel_type == 701:
        out = grad_linear_3d_1(r1, r2, r3, h1, h2, h3)
    elif kernel_type == 702:
        out = grad_linear_3d_2(r1, r2, r3, h1, h2, h3)
    elif kernel_type == 703:
        out = grad_linear_3d_3(r1, r2, r3, h1, h2, h3)

    return out
