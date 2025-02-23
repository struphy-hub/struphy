from numpy import abs, cos, exp, pi, sin, sqrt


def linear_isotropic(r: "float", h: "float") -> float:
    """
    Smoothing kernel S(r,h) = C(h)F(r/h) with F(x) = 1-x if x<1, 0 else,
    and C(h)=3/(pi*h^3) is a normalization coefficient so the the kernel has unit integral.
    """
    if r / h > 1.0:
        return 0.0
    else:
        return (1.0 - r / h) / (1.0471975512 * h**3)  # normalization


def trigonometric(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    if abs(r1 / h1) <= 1.0:
        s1 = 0.785398163397448 / h1 * cos(r1 / h1 * pi / 2.0)
    else:
        return 0.0

    if abs(r2 / h2) <= 1.0:
        s2 = 0.785398163397448 / h2 * cos(r2 / h2 * pi / 2.0)
    else:
        return 0.0

    if abs(r3 / h3) <= 1.0:
        s3 = 0.785398163397448 / h3 * cos(r3 / h3 * pi / 2.0)
    else:
        return 0.0

    return s1 * s2 * s3


def gaussian(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """Tensor product of kernels S(x, h) = 1/(sqrt(pi)*h/3) * exp(-(x**2/(h/3)**2) if |x|<1, 0 else."""
    if abs(r1 / h1) <= 1.0:
        s1 = 1 / (sqrt(pi) * h1 / 3) * exp(-(r1**2) / (h1 / 3) ** 2)
    else:
        return 0.0

    if abs(r2 / h2) <= 1.0:
        s2 = 1 / (sqrt(pi) * h2 / 3) * exp(-(r2**2) / (h2 / 3) ** 2)
    else:
        return 0.0

    if abs(r3 / h3) <= 1.0:
        s3 = 1 / (sqrt(pi) * h3 / 3) * exp(-(r3**2) / (h3 / 3) ** 2)
    else:
        return 0.0

    return s1 * s2 * s3


def linear_tp(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    if abs(r1 / h1) <= 1.0:
        s1 = (1.0 - abs(r1 / h1)) / h1
    else:
        return 0.0

    if abs(r2 / h2) <= 1.0:
        s2 = (1.0 - abs(r2 / h2)) / h2
    else:
        return 0.0

    if abs(r3 / h3) <= 1.0:
        s3 = (1.0 - abs(r3 / h3)) / h3
    else:
        return 0.0

    return s1 * s2 * s3


def linear_tp_2d(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    if abs(r1 / h1) <= 1.0:
        s1 = (1.0 - abs(r1 / h1)) / h1
    else:
        return 0.0

    if abs(r2 / h2) <= 1.0:
        s2 = (1.0 - abs(r2 / h2)) / h2
    else:
        return 0.0

    return s1 * s2


def trigonometric_2d(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    if abs(r1 / h1) <= 1.0:
        s1 = 0.785398163397448 / h1 * cos(r1 / h1 * pi / 2.0)
    else:
        return 0.0

    if abs(r2 / h2) <= 1.0:
        s2 = 0.785398163397448 / h2 * cos(r2 / h2 * pi / 2.0)
    else:
        return 0.0

    return s1 * s2


def gaussian_2d(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """Tensor product of kernels S(x, h) = 1/(sqrt(pi)*h/3) * exp(-(x**2/(h/3)**2) if |x|<1, 0 else."""
    if abs(r1 / h1) <= 1.0:
        s1 = 1 / (sqrt(pi) * h1 / 3) * exp(-(r1**2) / (h1 / 3) ** 2)
    else:
        return 0.0

    if abs(r2 / h2) <= 1.0:
        s2 = 1 / (sqrt(pi) * h2 / 3) * exp(-(r2**2) / (h2 / 3) ** 2)
    else:
        return 0.0

    return s1 * s2


# gradient of isotropic kernel

# add later


# gradient of trigonometric kernel


def grad_trigonometric_1(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """1st component of gradient of Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    if abs(r1 / h1) <= 1.0:
        ds1 = -(1.2337005501361697 / h1**2) * sin(r1 / h1 * pi / 2.0)
    else:
        return 0.0

    if abs(r2 / h2) <= 1.0:
        s2 = 0.785398163397448 / h2 * cos(r2 / h2 * pi / 2.0)
    else:
        return 0.0

    if abs(r3 / h3) <= 1.0:
        s3 = 0.785398163397448 / h3 * cos(r3 / h3 * pi / 2.0)
    else:
        return 0.0

    return ds1 * s2 * s3


def grad_trigonometric_2(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """2nd component of gradient of Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    if abs(r1 / h1) <= 1.0:
        s1 = 0.785398163397448 / h2 * cos(r1 / h1 * pi / 2.0)
    else:
        return 0.0

    if abs(r2 / h2) <= 1.0:
        ds2 = -(1.2337005501361697 / h2**2) * sin(r2 / h2 * pi / 2.0)
    else:
        return 0.0

    if abs(r3 / h3) <= 1.0:
        s3 = 0.785398163397448 / h3 * cos(r3 / h3 * pi / 2.0)
    else:
        return 0.0

    return s1 * ds2 * s3


def grad_trigonometric_3(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """3rd component of gradient of Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    if abs(r1 / h1) <= 1.0:
        s1 = 0.785398163397448 / h1 * cos(r1 / h1 * pi / 2.0)
    else:
        return 0.0

    if abs(r2 / h2) <= 1.0:
        s2 = 0.785398163397448 / h2 * cos(r2 / h2 * pi / 2.0)
    else:
        return 0.0

    if abs(r3 / h3) <= 1.0:
        ds3 = -(1.2337005501361697 / h3**2) * sin(r3 / h3 * pi / 2.0)
    else:
        return 0.0

    return s1 * s2 * ds3


# gradient of gaussian kernel (components)


def grad_gaussian_1(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """1st component of gradient of Tensor product of kernels S(x, h) = 1/(sqrt(pi)*h/3) * exp(-(x**2/(h/3)**2) if |x|<1, 0 else."""
    if abs(r1 / h1) <= 1.0:
        ds1 = -54 * r1 / (h1**3 * sqrt(pi)) * exp(-(r1**2) / (h1 / 3) ** 2)
    else:
        return 0.0

    if abs(r2 / h2) <= 1.0:
        s2 = 1 / (sqrt(pi) * h2 / 3) * exp(-(r2**2) / (h2 / 3) ** 2)
    else:
        return 0.0

    if abs(r3 / h3) <= 1.0:
        s3 = 1 / (sqrt(pi) * h3 / 3) * exp(-(r3**2) / (h3 / 3) ** 2)
    else:
        return 0.0

    return ds1 * s2 * s3


def grad_gaussian_2(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """2nd component of gradient of Tensor product of kernels S(x, h) = 1/(sqrt(pi)*h/3) * exp(-(x**2/(h/3)**2) if |x|<1, 0 else."""
    if abs(r1 / h1) <= 1.0:
        s1 = 1 / (sqrt(pi) * h1 / 3) * exp(-(r1**2) / (h1 / 3) ** 2)
    else:
        return 0.0

    if abs(r2 / h2) <= 1.0:
        ds2 = -54 * r2 / (h2**3 * sqrt(pi)) * exp(-(r2**2) / (h2 / 3) ** 2)
    else:
        return 0.0

    if abs(r3 / h3) <= 1.0:
        s3 = 1 / (sqrt(pi) * h3 / 3) * exp(-(r3**2) / (h3 / 3) ** 2)
    else:
        return 0.0

    return s1 * ds2 * s3


def grad_gaussian_3(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """3rd component of gradient of Tensor product of kernels S(x, h) = 1/(sqrt(pi)*h/3) * exp(-(x**2/(h/3)**2) if |x|<1, 0 else."""
    if abs(r1 / h1) <= 1.0:
        s1 = 1 / (sqrt(pi) * h1 / 3) * exp(-(r1**2) / (h1 / 3) ** 2)
    else:
        return 0.0

    if abs(r2 / h2) <= 1.0:
        s2 = 1 / (sqrt(pi) * h2 / 3) * exp(-(r2**2) / (h2 / 3) ** 2)
    else:
        return 0.0

    if abs(r3 / h3) <= 1.0:
        ds3 = -54 * r3 / (h3**3 * sqrt(pi)) * exp(-(r3**2) / (h3 / 3) ** 2)
    else:
        return 0.0

    return s1 * s2 * ds3


# gradient of linear kernel (components)


def grad_linear_tp_1(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """1st component of gradient of Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    if abs(r1 / h1) <= 1.0:
        ds1 = -(1 / h1**2) * (r1 / abs(r1))
    else:
        return 0.0

    if abs(r2 / h2) <= 1.0:
        s2 = (1.0 - abs(r2 / h2)) / h2
    else:
        return 0.0

    if abs(r3 / h3) <= 1.0:
        s3 = (1.0 - abs(r3 / h3)) / h3
    else:
        return 0.0

    return ds1 * s2 * s3


def grad_linear_tp_2(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """2nd component of gradient of Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    if abs(r1 / h1) <= 1.0:
        s1 = (1.0 - abs(r1 / h1)) / h1
    else:
        return 0.0

    if abs(r2 / h2) <= 1.0:
        ds2 = -(1 / h2**2) * (r2 / abs(r2))
    else:
        return 0.0

    if abs(r3 / h3) <= 1.0:
        s3 = (1.0 - abs(r3 / h3)) / h3
    else:
        return 0.0

    return s1 * ds2 * s3


def grad_linear_tp_3(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """3rd component of gradient of Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    if abs(r1 / h1) <= 1.0:
        s1 = (1.0 - abs(r1 / h1)) / h1
    else:
        return 0.0

    if abs(r2 / h2) <= 1.0:
        s2 = (1.0 - abs(r2 / h2)) / h2
    else:
        return 0.0

    if abs(r3 / h3) <= 1.0:
        ds3 = -(1 / h3**2) * (r3 / abs(r3))
    else:
        return 0.0

    return s1 * s2 * ds3


# gradients in 2d

# 2d gradient of trigonometric kernel (components)


def grad_trigonometric_2d_1(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """1st component of gradient of Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    if abs(r1 / h1) <= 1.0:
        ds1 = -(1.2337005501361697 / h1**2) * sin(r1 / h1 * pi / 2.0)
    else:
        return 0.0

    if abs(r2 / h2) <= 1.0:
        s2 = 0.785398163397448 / h2 * cos(r2 / h2 * pi / 2.0)
    else:
        return 0.0

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
    if abs(r1 / h1) <= 1.0:
        s1 = 0.785398163397448 / h2 * cos(r1 / h1 * pi / 2.0)
    else:
        return 0.0

    if abs(r2 / h2) <= 1.0:
        ds2 = -(1.2337005501361697 / h2**2) * sin(r2 / h2 * pi / 2.0)
    else:
        return 0.0

    return s1 * ds2


# 2d gradient of gaussian kernel (components)


def grad_gaussian_2d_1(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """1st component of gradient of Tensor product of kernels S(x, h) = 1/(sqrt(pi)*h/3) * exp(-(x**2/(h/3)**2) if |x|<1, 0 else."""
    if abs(r1 / h1) <= 1.0:
        ds1 = -54 * r1 / (h1**3 * sqrt(pi)) * exp(-(r1**2) / (h1 / 3) ** 2)
    else:
        return 0.0

    if abs(r2 / h2) <= 1.0:
        s2 = 1 / (sqrt(pi) * h2 / 3) * exp(-(r2**2) / (h2 / 3) ** 2)
    else:
        return 0.0

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
    if abs(r1 / h1) <= 1.0:
        s1 = 1 / (sqrt(pi) * h1 / 3) * exp(-(r1**2) / (h1 / 3) ** 2)
    else:
        return 0.0

    if abs(r2 / h2) <= 1.0:
        ds2 = -54 * r2 / (h2**3 * sqrt(pi)) * exp(-(r2**2) / (h2 / 3) ** 2)
    else:
        return 0.0

    return s1 * ds2


# 2d gradient of linear kernel (components)


def grad_linear_tp_2d_1(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """1st component of gradient of Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    if abs(r1 / h1) <= 1.0:
        ds1 = -(1 / h1**2) * (r1 / abs(r1))
    else:
        return 0.0

    if abs(r2 / h2) <= 1.0:
        s2 = (1.0 - abs(r2 / h2)) / h2
    else:
        return 0.0

    return ds1 * s2


def grad_linear_tp_2d_2(
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """2nd component of gradient of Tensor product of kernels S(x, h) = pi/4/h * cos(x*pi/2) if |x|<1, 0 else."""
    if abs(r1 / h1) <= 1.0:
        s1 = (1.0 - abs(r1 / h1)) / h1
    else:
        return 0.0

    if abs(r2 / h2) <= 1.0:
        ds2 = -(1 / h2**2) * (r2 / abs(r2))
    else:
        return 0.0

    return s1 * ds2


def smoothing_kernel(
    kernel_type: "int",
    r1: "float",
    r2: "float",
    r3: "float",
    h1: "float",
    h2: "float",
    h3: "float",
) -> float:
    """Each smoothing kernel is normalized to 1."""

    if kernel_type == 0:
        r = sqrt(r1**2 + r2**2 + r3**2)
        h = h1
        out = linear_isotropic(r, h)
    elif kernel_type == 100:
        out = trigonometric(r1, r2, r3, h1, h2, h3)
    elif kernel_type == 200:
        out = gaussian(r1, r2, r3, h1, h2, h3)
    elif kernel_type == 300:
        out = linear_tp(r1, r2, r3, h1, h2, h3)
    elif kernel_type == 400:
        out = trigonometric_2d(r1, r2, r3, h1, h2, h3)
    elif kernel_type == 401:
        out = grad_trigonometric_2d_1(r1, r2, r3, h1, h2, h3)
    elif kernel_type == 402:
        out = grad_trigonometric_2d_2(r1, r2, r3, h1, h2, h3)
    elif kernel_type == 500:
        out = gaussian_2d(r1, r2, r3, h1, h2, h3)
    elif kernel_type == 501:
        out = grad_gaussian_2d_1(r1, r2, r3, h1, h2, h3)
    elif kernel_type == 502:
        out = grad_gaussian_2d_2(r1, r2, r3, h1, h2, h3)
    elif kernel_type == 600:
        out = linear_tp_2d(r1, r2, r3, h1, h2, h3)
    elif kernel_type == 601:
        out = grad_linear_tp_2d_1(r1, r2, r3, h1, h2, h3)
    elif kernel_type == 602:
        out = grad_linear_tp_2d_2(r1, r2, r3, h1, h2, h3)
    return out
