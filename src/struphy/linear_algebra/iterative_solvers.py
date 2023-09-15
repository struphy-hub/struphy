# coding: utf-8
"""
This module provides iterative solvers with and without preconditioners.

"""
from math import sqrt
import numpy as np

from psydac.linalg.basic import VectorSpace, Vector, LinearOperator, LinearSolver


__all__ = ['ConjugateGradient', 'PConjugateGradient',
           'BiConjugateGradientStab', 'PBiConjugateGradientStab']


class ConjugateGradient(LinearSolver):
    """
    Conjugate gradient algorithm for solving a linear system Ax = b. Implementation according to p. 160 in [1].

    Parameters
    ----------
    space : psydac.linalg.basic.VectorSpace
        The vector space associated to the linear solver.

    References
    ----------
    [1] A. Maister, Numerik linearer Gleichungssysteme, Springer ed. 2015.
    """

    def __init__(self, space):

        # vector space of the linear operator that will be inverted
        assert isinstance(space, VectorSpace)
        self._space = space

        # temporary vectors needed in interation loop
        self._tmps = {'v': space.zeros(), 'r': space.zeros(), 'p': space.zeros(),
                      'lp': space.zeros(), 'lv': space.zeros()}

    @property
    def space(self):
        return self._space

    def solve(self, A, b, x0=None, tol=1e-6, maxiter=1000, verbose=False, out=None):
        """
        Solves the linear system Ax = b for x.

        Parameters
        ----------
        A : psydac.linalg.basic.LinearOperator
            The linear operator to be inverted.

        b : psydac.linalg.basic.Vector
            The right-hand side vector of the liner system. From the same space as the domain/codomain of A.

        x0 : psydac.linalg.basic.Vector, optional
            Initial guess for the solution.

        tol : float, optional
            Stop tolerance of residual norm ||Ax - b||, where x is the current solution.

        maxiter : int, optional
            Maximum number of iterations.

        verbose : bool, optional
            Whether to print the residual norm in each iteration step.

        out : psydac.linalg.basic.Vector, optional
            If given, the converged solution will be written in-place into this vector.

        Returns
        -------
        x : psydac.linalg.basic.Vector
            The converged solution.

        info : dict
            Some information (number of iterations, residual norm, if convergence was reached).
        """

        assert isinstance(A, LinearOperator)
        assert A.shape[0] == A.shape[1]
        assert b.shape[0] == A.shape[0]

        # first guess of solution
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space == self._space
            out *= 0
            if x0 is None:
                x = out
            else:
                assert x0.shape == (A.shape[0],)
                out += x0
                x = out
        else:
            if x0 is None:
                x = b.copy()
                x *= 0.0
            else:
                assert x0.shape == (A.shape[0],)
                x = x0.copy()

        # extract temporary vectors
        v = self._tmps['v']
        r = self._tmps['r']
        p = self._tmps['p']

        lp = self._tmps['lp']
        lv = self._tmps['lv']

        # first values: r = p = b - A @ x, alpha = ||r||^2
        A.dot(x, out=v)
        b.copy(out=r)
        r -= v
        r.copy(out=p)
        alpha = r.dot(r)

        # squared residual norm and squared tolerance
        res_sqr = r.dot(r)
        tol_sqr = tol**2

        if verbose:
            print("CG solver:")
            print("+---------+---------------------+")
            print("+ Iter. # | L2-norm of residual |")
            print("+---------+---------------------+")
            template = "| {:7d} | {:19.2e} |"

        # iterate to convergence or maximum number of iterations
        niter = 0

        while res_sqr > tol_sqr and niter < maxiter:

            # v = A @ p, lam = alpha/(v.p)
            A.dot(p, out=v)
            lam = alpha / v.dot(p)

            # x = x + lam*p
            p.copy(out=lp)
            lp *= lam
            x += lp

            # r = r - lam*v
            v.copy(out=lv)
            lv *= lam
            r -= lv

            # alpha = ||r||^2, p = r + (alpha_new/alpha)*p
            alpha_new = r.dot(r)
            p *= (alpha_new/alpha)
            p += r
            alpha = 1*alpha_new

            # new residual norm
            res_sqr = r.dot(r)

            niter += 1

            if verbose:
                print(template.format(niter, sqrt(res_sqr)))

        if verbose:
            print("+---------+---------------------+")

        # convergence information
        info = {'niter': niter, 'success': res_sqr <
                tol_sqr, 'res_norm': sqrt(res_sqr)}

        return x, info


class PConjugateGradient(LinearSolver):
    """
    Pre-conditioned conjugate gradient algorithm for solving a linear system Ax = b. Implementation according to p. 248 in [1].

    Parameters
    ----------
    space : psydac.linalg.basic.VectorSpace
        The vector space associated to the linear solver.

    References
    ----------
    [1] A. Maister, Numerik linearer Gleichungssysteme, Springer ed. 2015.
    """

    def __init__(self, space):

        # vector space of the linear operator that will be inverted
        assert isinstance(space, VectorSpace)
        self._space = space

        # temporary vectors needed in interation loop
        self._tmps = {'v': space.zeros(), 'r': space.zeros(), 'p': space.zeros(),
                      'z': space.zeros(), 'lp': space.zeros(), 'lv': space.zeros()}

    @property
    def space(self):
        return self._space

    def solve(self, A, b, pc, x0=None, tol=1e-6, maxiter=1000, verbose=False, out=None):
        """
        Solves the linear system Ax = b for x.

        Parameters
        ----------
        A : psydac.linalg.basic.LinearOperator
            The linear operator to be inverted.

        b : psydac.linalg.basic.Vector
            The right-hand side vector of the liner system. From the same space as the domain/codomain of A.

        pc : psydac.linalg.basic.Vector.LinearSolver
            The preconditioner that approximates the inverse of A. Must have a "solve" method.

        x0 : psydac.linalg.basic.Vector, optional
            Initial guess for the solution.

        tol : float, optional
            Stop tolerance of residual norm ||Ax - b||, where x is the current solution.

        maxiter : int, optional
            Maximum number of iterations.

        verbose : bool, optional
            Whether to print the residual norm in each iteration step.

        out : psydac.linalg.basic.Vector, optional
            If given, the converged solution will be written in-place into this vector.

        Returns
        -------
        x : psydac.linalg.basic.Vector
            The converged solution.

        info : dict
            Some information (number of iterations, residual norm, if convergence was reached).
        """

        assert isinstance(A, LinearOperator)
        assert A.shape[0] == A.shape[1]
        assert b.shape[0] == A.shape[0]

        # first guess of solution
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space == self._space
            out *= 0
            if x0 is None:
                x = out
            else:
                assert x0.shape == (A.shape[0],)
                out += x0
                x = out
        else:
            if x0 is None:
                x = b.copy()
                x *= 0.0
            else:
                assert x0.shape == (A.shape[0],)
                x = x0.copy()

        # preconditioner (must have a .solve method)
        assert isinstance(pc, LinearSolver), 'If you do not use a preconditioner, please use the solver "ConjugateGradient" instead of "PConjugateGradient".'

        # extract temporary vectors
        v = self._tmps['v']
        r = self._tmps['r']
        p = self._tmps['p']
        z = self._tmps['z']

        lp = self._tmps['lp']
        lv = self._tmps['lv']

        # first values: r = b - A @ x, p = PC @ r, alphap = r.p
        A.dot(x, out=v)
        b.copy(out=r)
        r -= v
        pc.solve(r, out=p)
        alphap = r.dot(p)

        # squared residual norm and squared tolerance
        res_sqr = r.dot(r)
        tol_sqr = tol**2

        if verbose:
            print(" Pre-conditioned CG solver:")
            print("+---------+---------------------+")
            print("+ Iter. # | L2-norm of residual |")
            print("+---------+---------------------+")
            template = "| {:7d} | {:19.2e} |"

        # iterate to convergence or maximum number of iterations
        niter = 0

        while res_sqr > tol_sqr and niter < maxiter:

            # v = A @ p, lam = alphap/(v.p)
            A.dot(p, out=v)
            lamp = alphap / v.dot(p)

            # x = x + lam*p
            p.copy(out=lp)
            lp *= lamp
            x += lp

            # r = r - lam*v
            v.copy(out=lv)
            lv *= lamp
            r -= lv

            # z = PC @ r
            pc.solve(r, out=z)

            # alphap = r.z, p = z + (alphap_new/alphap)*p
            alphap_new = r.dot(z)
            p *= (alphap_new/alphap)
            p += z
            alphap = 1*alphap_new

            # new residual norm
            res_sqr = r.dot(r)

            niter += 1

            if verbose:
                print(template.format(niter, sqrt(res_sqr)))

        if verbose:
            print("+---------+---------------------+")

        # convergence information
        info = {'niter': niter, 'success': res_sqr <
                tol_sqr, 'res_norm': sqrt(res_sqr)}

        return x, info


class BiConjugateGradientStab(LinearSolver):
    """
    Biconjugate gradient stabilized algorithm for solving a linear system Ax = b. Implementation according to p. 208 in [1].

    Parameters
    ----------
    space : psydac.linalg.basic.VectorSpace
        The vector space associated to the linear solver.

    References
    ----------
    [1] A. Maister, Numerik linearer Gleichungssysteme, Springer ed. 2015.
    """

    def __init__(self, space):

        # vector space of the linear operator that will be inverted
        assert isinstance(space, VectorSpace)
        self._space = space

        # temporary vectors needed in interation loop
        self._tmps = {'v': space.zeros(), 'r': space.zeros(), 'p': space.zeros(),
                      's': space.zeros(), 't': space.zeros(), 'r0': space.zeros(),
                      'av': space.zeros(), 'ap': space.zeros(), 'os': space.zeros()}

    @property
    def space(self):
        return self._space

    def solve(self, A, b, x0=None, tol=1e-6, maxiter=1000, verbose=False, out=None):
        """
        Solves the linear system Ax = b for x.

        Parameters
        ----------
        A : psydac.linalg.basic.LinearOperator
            The linear operator to be inverted.

        b : psydac.linalg.basic.Vector
            The right-hand side vector of the liner system. From the same space as the domain/codomain of A.

        x0 : psydac.linalg.basic.Vector, optional
            Initial guess for the solution.

        tol : float, optional
            Stop tolerance of residual norm ||Ax - b||, where x is the current solution.

        maxiter : int, optional
            Maximum number of iterations.

        verbose : bool, optional
            Whether to print the residual norm in each iteration step.

        out : psydac.linalg.basic.Vector, optional
            If given, the converged solution will be written in-place into this vector.

        Returns
        -------
        x : psydac.linalg.basic.Vector
            The converged solution.

        info : dict
            Some information (number of iterations, residual norm, if convergence was reached).
        """

        assert isinstance(A, LinearOperator)
        assert A.shape[0] == A.shape[1]
        assert b.shape[0] == A.shape[0]

        # first guess of solution
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space == self._space
            out *= 0
            if x0 is None:
                x = out
            else:
                assert x0.shape == (A.shape[0],)
                out += x0
                x = out
        else:
            if x0 is None:
                x = b.copy()
                x *= 0.0
            else:
                assert x0.shape == (A.shape[0],)
                x = x0.copy()

        # extract temporary vectors
        v = self._tmps['v']
        r = self._tmps['r']
        p = self._tmps['p']
        s = self._tmps['s']
        t = self._tmps['t']

        av = self._tmps['av']
        ap = self._tmps['ap']
        os = self._tmps['os']

        # first values: r0 = p0 = b - A @ x0, rho0 = |r0|^2
        A.dot(x, out=v)
        b.copy(out=r)
        r -= v
        rho = r.dot(r)
        r.copy(out=p)

        # save initial residual vector r0
        r0 = self._tmps['r0']
        r.copy(out=r0)

        # squared residual norm and squared tolerance
        res_sqr = 1*rho
        tol_sqr = tol**2

        if verbose:
            print("BICGSTAB solver:")
            print("+---------+---------------------+")
            print("+ Iter. # | L2-norm of residual |")
            print("+---------+---------------------+")
            template = "| {:7d} | {:19.2e} |"

        # iterate to convergence or maximum number of iterations
        niter = 0

        while res_sqr > tol_sqr and niter < maxiter:

            # v = A @ p, alpha = rho/(v.r0)
            A.dot(p, out=v)
            alpha = rho / v.dot(r0)

            # s = r - alpha*v
            r.copy(out=s)
            v.copy(out=av)
            av *= alpha
            s -= av

            # t = A @ s, omega = (t.s)/(t.t)
            A.dot(s, out=t)
            omega = t.dot(s) / t.dot(t)

            # x = x + alpha*p + omega*s
            p.copy(out=ap)
            s.copy(out=os)
            ap *= alpha
            os *= omega
            x += ap
            x += os

            # r = s - omega*t
            s.copy(out=r)
            t *= omega
            r -= t

            # rho_new = r.r0, beta = (alpha*rho_new)/(omega*rho)
            rho_new = r.dot(r0)
            beta = (alpha*rho_new) / (omega*rho)
            rho = 1*rho_new

            # p = r + beta*(p - omega*v)
            v *= omega
            p -= v
            p *= beta
            p += r

            # new residual norm
            res_sqr = r.dot(r)

            niter += 1

            if verbose:
                print(template.format(niter, sqrt(res_sqr)))

        if verbose:
            print("+---------+---------------------+")

        # convergence information
        info = {'niter': niter, 'success': res_sqr <
                tol_sqr, 'res_norm': sqrt(res_sqr)}

        return x, info


class PBiConjugateGradientStab(LinearSolver):
    """
    Pre-conditioned biconjugate gradient stabilized algorithm for solving a linear system Ax = b. Implementation according to p. 251 in [1].

    Parameters
    ----------
    space : psydac.linalg.basic.VectorSpace
        The vector space associated to the linear solver.

    References
    ----------
    [1] A. Maister, Numerik linearer Gleichungssysteme, Springer ed. 2015.
    """

    def __init__(self, space):

        # vector space of the linear operator that will be inverted
        assert isinstance(space, VectorSpace)
        self._space = space

        # temporary vectors needed in interation loop
        self._tmps = {'v': space.zeros(), 'r': space.zeros(), 's': space.zeros(), 't': space.zeros(),
                      'vp': space.zeros(), 'rp': space.zeros(), 'pp': space.zeros(), 'sp': space.zeros(),
                      'tp': space.zeros(), 'av': space.zeros(), 'app': space.zeros(), 'osp': space.zeros(),
                      'rp0': space.zeros()}

    @property
    def space(self):
        return self._space

    def solve(self, A, b, pc, x0=None, tol=1e-6, maxiter=1000, verbose=False, out=None):
        """
        Solves the linear system Ax = b for x.

        Parameters
        ----------
        A : psydac.linalg.basic.LinearOperator
            The linear operator to be inverted.

        b : psydac.linalg.basic.Vector
            The right-hand side vector of the liner system. From the same space as the domain/codomain of A.

        pc : psydac.linalg.basic.Vector.LinearSolver
            The preconditioner that approximates the inverse of A. Must have a "solve" method.

        x0 : psydac.linalg.basic.Vector, optional
            Initial guess for the solution.

        tol : float, optional
            Stop tolerance of residual norm ||Ax - b||, where x is the current solution.

        maxiter : int, optional
            Maximum number of iterations.

        verbose : bool, optional
            Whether to print the residual norm in each iteration step.

        out : psydac.linalg.basic.Vector, optional
            If given, the converged solution will be written in-place into this vector.

        Returns
        -------
        x : psydac.linalg.basic.Vector
            The converged solution.

        info : dict
            Some information (number of iterations, residual norm, if convergence was reached).
        """

        assert isinstance(A, LinearOperator)
        assert A.shape[0] == A.shape[1]
        assert b.shape[0] == A.shape[0]

        # first guess of solution
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space == self._space
            out *= 0
            if x0 is None:
                x = out
            else:
                assert x0.shape == (A.shape[0],)
                out += x0
                x = out
        else:
            if x0 is None:
                x = b.copy()
                x *= 0.0
            else:
                assert x0.shape == (A.shape[0],)
                x = x0.copy()

        # preconditioner (must have a .solve method)
        assert isinstance(pc, LinearSolver)

        # extract temporary vectors
        v = self._tmps['v']
        r = self._tmps['r']
        s = self._tmps['s']
        t = self._tmps['t']

        vp = self._tmps['vp']
        rp = self._tmps['rp']
        pp = self._tmps['pp']
        sp = self._tmps['sp']
        tp = self._tmps['tp']

        av = self._tmps['av']

        app = self._tmps['app']
        osp = self._tmps['osp']

        # first values: r = b - A @ x, rp = pp = PC @ r, rhop = |rp|^2
        A.dot(x, out=v)
        b.copy(out=r)
        r -= v

        pc.solve(r, out=rp)
        rp.copy(out=pp)

        rhop = rp.dot(rp)

        # save initial residual vector rp0
        rp0 = self._tmps['rp0']
        rp.copy(out=rp0)

        # squared residual norm and squared tolerance
        res_sqr = r.dot(r)
        tol_sqr = tol**2

        if verbose:
            print("Pre-conditioned BICGSTAB solver:")
            print("+---------+---------------------+")
            print("+ Iter. # | L2-norm of residual |")
            print("+---------+---------------------+")
            template = "| {:7d} | {:19.2e} |"

        # iterate to convergence or maximum number of iterations
        niter = 0

        while res_sqr > tol_sqr and niter < maxiter:

            # v = A @ pp, vp = PC @ v, alphap = rhop/(vp.rp0)
            A.dot(pp, out=v)
            pc.solve(v, out=vp)
            alphap = rhop / vp.dot(rp0)

            # s = r - alphap*v, sp = PC @ s
            r.copy(out=s)
            v.copy(out=av)
            av *= alphap
            s -= av
            pc.solve(s, out=sp)

            # t = A @ sp, tp = PC @ t, omegap = (tp.sp)/(tp.tp)
            A.dot(sp, out=t)
            pc.solve(t, out=tp)
            omegap = tp.dot(sp) / tp.dot(tp)

            # x = x + alphap*pp + omegap*sp
            pp.copy(out=app)
            sp.copy(out=osp)
            app *= alphap
            osp *= omegap
            x += app
            x += osp

            # r = s - omegap*t, rp = sp - omegap*tp
            s.copy(out=r)
            t *= omegap
            r -= t

            sp.copy(out=rp)
            tp *= omegap
            rp -= tp

            # rhop_new = rp.rp0, betap = (alphap*rhop_new)/(omegap*rhop)
            rhop_new = rp.dot(rp0)
            betap = (alphap*rhop_new) / (omegap*rhop)
            rhop = 1*rhop_new

            # pp = rp + betap*(pp - omegap*vp)
            vp *= omegap
            pp -= vp
            pp *= betap
            pp += rp

            # new residual norm
            res_sqr = r.dot(r)

            niter += 1

            if verbose:
                print(template.format(niter, sqrt(res_sqr)))

        if verbose:
            print("+---------+---------------------+")

        # convergence information
        info = {'niter': niter, 'success': res_sqr <
                tol_sqr, 'res_norm': sqrt(res_sqr)}

        return x, info
