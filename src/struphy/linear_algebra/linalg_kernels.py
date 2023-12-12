from pyccel.decorators import pure, stack_array
from numpy import zeros


@pure
def matrix_vector(a: 'float[:,:]', b: 'float[:]', c: 'float[:]'):
    """
    Performs the matrix-vector product of a 3x3 matrix with a vector.

    Parameters
    ----------
        a : array[float]
            The input array (matrix) of shape (3,3).

        b : array[float]
            The input array (vector) of shape (3,).

        c : array[float]
            The output array (vector) of shape (3,) which is the result of the matrix-vector product a.dot(b).
    """

    c[:] = 0.

    for i in range(3):
        for j in range(3):
            c[i] += a[i, j] * b[j]


@pure
def matrix_matrix(a: 'float[:,:]', b: 'float[:,:]', c: 'float[:,:]'):
    """
    Performs the matrix-matrix product of a 3x3 matrix with another 3x3 matrix.

    Parameters
    ----------
        a : array[float]
            The first input array (matrix) of shape (3,3).

        b : array[float]
            The second input array (matrix) of shape (3,3).

        c : array[float]
            The output array (matrix) of shape (3,3) which is the result of the matrix-matrix product a.dot(b).
    """

    c[:, :] = 0.

    for i in range(3):
        for j in range(3):
            for k in range(3):
                c[i, j] += a[i, k] * b[k, j]


@pure
def transpose(a: 'float[:,:]', b: 'float[:,:]'):
    """
    Assembles the transposed of a 3x3 matrix.

    Parameters
    ----------
        a : array[float]
            The input array (matrix) of shape (3,3).

        b : array[float]
            The output array (matrix) of shape (3,3).
    """

    for i in range(3):
        for j in range(3):
            b[i, j] = a[j, i]


@pure
def scalar_dot(a: 'float[:]', b: 'float[:]') -> float:
    """
    Computes scalar (dot) product of two vectors of length 3.

    Parameters
    ----------
        a : array[float]
            The first input array (vector) of shape (3,).

        b : array[float]
            The second input array (vector) of shape (3,).

    Returns
    -------
        value : float
            The scalar poduct of the two input vectors a and b.
    """

    value = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

    return value


@pure
def det(a: 'float[:,:]') -> float:
    """
    Computes the determinant of a 3x3 matrix.

    Parameters
    ----------
        a : array[float]
            The input array (matrix) of shape (3,3) of which the determinant shall be computed.

    Returns
    -------
        det_a : float
            The determinant of the 3x3 matrix a.
    """

    plus = a[0, 0]*a[1, 1]*a[2, 2] + a[0, 1] * \
        a[1, 2]*a[2, 0] + a[0, 2]*a[1, 0]*a[2, 1]
    minus = a[2, 0]*a[1, 1]*a[0, 2] + a[2, 1] * \
        a[1, 2]*a[0, 0] + a[2, 2]*a[1, 0]*a[0, 1]

    det_a = plus - minus

    return det_a


@pure
def cross(a: 'float[:]', b: 'float[:]', c: 'float[:]'):
    """
    Computes the vector (cross) product of two vectors of length 3. 

    Parameters
    ----------
        a : array[float]
            The first input array (vector) of shape (3,).

        b : array[float]
            The second input array (vector) of shape (3,).

        c : array[float]
            The output array (vector) of shape (3,) which is the vector product a x b.
    """

    c[0] = a[1]*b[2] - a[2]*b[1]
    c[1] = a[2]*b[0] - a[0]*b[2]
    c[2] = a[0]*b[1] - a[1]*b[0]


@pure
def outer(a: 'float[:]', b: 'float[:]', c: 'float[:,:]'):
    """
    Computes the outer product of two vectors of length 3. 

    Parameters
    ----------
        a : array[float]
            The first input array (vector) of shape (3,).

        b : array[float]
            The second input array (vector) of shape (3,).

        c : array[float]
            The output array (matrix) of shape (3, 3) which is the outer product c_ij = a_i*b_j.
    """

    c[:, :] = 0.

    for i in range(3):
        for j in range(3):
            c[i, j] = a[i] * b[j]


@stack_array('det_a')
def matrix_inv(a: 'float[:,:]', b: 'float[:,:]'):
    """
    Computes the inverse of a 3x3 matrix.

    Parameters
    ----------
        a : array[float]
            The input array (matrix) of shape (3,3).

        b : array[float]
            The output array (matrix) of shape (3,3).
    """

    det_a = det(a)

    b[0, 0] = (a[1, 1]*a[2, 2] - a[2, 1]*a[1, 2]) / det_a
    b[0, 1] = (a[2, 1]*a[0, 2] - a[0, 1]*a[2, 2]) / det_a
    b[0, 2] = (a[0, 1]*a[1, 2] - a[1, 1]*a[0, 2]) / det_a

    b[1, 0] = (a[1, 2]*a[2, 0] - a[2, 2]*a[1, 0]) / det_a
    b[1, 1] = (a[2, 2]*a[0, 0] - a[0, 2]*a[2, 0]) / det_a
    b[1, 2] = (a[0, 2]*a[1, 0] - a[1, 2]*a[0, 0]) / det_a

    b[2, 0] = (a[1, 0]*a[2, 1] - a[2, 0]*a[1, 1]) / det_a
    b[2, 1] = (a[2, 0]*a[0, 1] - a[0, 0]*a[2, 1]) / det_a
    b[2, 2] = (a[0, 0]*a[1, 1] - a[1, 0]*a[0, 1]) / det_a


@pure
def matrix_inv_with_det(a: 'float[:,:]', det_a: float, b: 'float[:,:]'):
    """
    Computes the inverse of a 3x3 matrix for the case that the determinant is already known such that its extra compuation can be avoided.

    Parameters
    ----------
        a : array[float]
            The input array (matrix) of shape (3,3).

        det_a : float
            The determinant of the input array (matrix) a.

        b : array[float]
            The output array (matrix) of shape (3,3).
    """

    b[0, 0] = (a[1, 1]*a[2, 2] - a[2, 1]*a[1, 2]) / det_a
    b[0, 1] = (a[2, 1]*a[0, 2] - a[0, 1]*a[2, 2]) / det_a
    b[0, 2] = (a[0, 1]*a[1, 2] - a[1, 1]*a[0, 2]) / det_a

    b[1, 0] = (a[1, 2]*a[2, 0] - a[2, 2]*a[1, 0]) / det_a
    b[1, 1] = (a[2, 2]*a[0, 0] - a[0, 2]*a[2, 0]) / det_a
    b[1, 2] = (a[0, 2]*a[1, 0] - a[1, 2]*a[0, 0]) / det_a

    b[2, 0] = (a[1, 0]*a[2, 1] - a[2, 0]*a[1, 1]) / det_a
    b[2, 1] = (a[2, 0]*a[0, 1] - a[0, 0]*a[2, 1]) / det_a
    b[2, 2] = (a[0, 0]*a[1, 1] - a[1, 0]*a[0, 1]) / det_a


@pure
def matrix_vector4(a: 'float[:,:]', b: 'float[:]', c: 'float[:]'):
    """
    Performs the matrix-vector product of a 4x4 matrix with a vector.

    Parameters
    ----------
        a : array[float]
            The input array (matrix) of shape (4,4).

        b : array[float]
            The input array (vector) of shape (4,).

        c : array[float]
            The output array (vector) of shape (4,) which is the result of the matrix-vector product a.dot(b).
    """

    c[:] = 0.

    for i in range(4):
        for j in range(4):
            c[i] += a[i, j] * b[j]


@pure
def matrix_matrix4(a: 'float[:,:]', b: 'float[:,:]', c: 'float[:,:]'):
    """
    Performs the matrix-matrix product of a 4x4 matrix with another 4x4 matrix.

    Parameters
    ----------
        a : array[float]
            The first input array (matrix) of shape (4,4).

        b : array[float]
            The second input array (matrix) of shape (4,4).

        c : array[float]
            The output array (matrix) of shape (4,4) which is the result of the matrix-matrix product a.dot(b).
    """

    c[:, :] = 0.

    for i in range(4):
        for j in range(4):
            for k in range(4):
                c[i, j] += a[i, k] * b[k, j]


@stack_array('tmp1', 'tmp2')
def det4(a: 'float[:,:]') -> float:
    """
    Computes the determinant of a 4x4 matrix.

    Parameters
    ----------
        a : array[float]
            The input array (matrix) of shape (4,4) of which the determinant shall be computed.

    Returns
    -------
        det_a : float
            The determinant of the 3x3 matrix a.
    """

    tmp1 = zeros((3, 3), dtype=float)
    tmp2 = zeros((3, 3), dtype=float)

    tmp1[0] = a[0, 1:]
    tmp1[1] = a[1, 1:]
    tmp1[2] = a[3, 1:]

    tmp2[:] = a[1:, 1:]

    plus = a[0, 0]*det(tmp2) + a[2, 0]*det(tmp1)

    tmp1[2] = a[2, 1:]

    tmp2[:] = a[:3, 1:]

    minus = a[1, 0]*det(tmp1) + a[3, 0]*det(tmp2)

    det_a = plus - minus

    return det_a
