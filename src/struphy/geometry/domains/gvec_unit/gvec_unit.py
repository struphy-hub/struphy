import cunumpy as xp

from struphy.geometry.base import Spline, interp_mapping


class GVECunit(Spline):
    """The mapping from `pygvec <https://gvec.readthedocs.io/latest/index.html>`_, computed by the GVEC MHD equilibrium code.

    .. image:: ../../pics/mappings/gvec.png

    Parameters
    ----------
    gvec_equil : struphy.fields_background.equils.GVECequilibrium
        GVEC MHD equilibrium object.
    """

    def __init__(self, gvec_equil=None):
        import gvec

        from struphy.fields_background.equils import GVECequilibrium

        if gvec_equil is None:
            gvec_equil = GVECequilibrium()
        else:
            assert isinstance(gvec_equil, GVECequilibrium)

        # do not set params here because of a pickling error

        num_elements = gvec_equil.params["num_elements"]
        degree = gvec_equil.params["degree"]
        if gvec_equil.params["use_nfp"]:
            spl_kind = (False, True, False)
        else:
            spl_kind = (False, True, True)

        # project mapping to splines
        _rmin = gvec_equil.params["rmin"]

        def XYZ(e1, e2, e3):
            rho = _rmin + e1 * (1.0 - _rmin)
            theta = 2 * xp.pi * e2
            zeta = 2 * xp.pi * e3 / gvec_equil._nfp
            if gvec_equil.params["use_boozer"]:
                ev = gvec.EvaluationsBoozer(rho=rho, theta_B=theta, zeta_B=zeta, state=gvec_equil.state)
            else:
                ev = gvec.Evaluations(rho=rho, theta=theta, zeta=zeta, state=gvec_equil.state)
            gvec_equil.state.compute(ev, "pos")
            x = ev.pos.data[0]
            y = ev.pos.data[1]
            z = ev.pos.data[2]
            return x, y, z

        def X(e1, e2, e3):
            return XYZ(e1, e2, e3)[0]

        def Y(e1, e2, e3):
            return XYZ(e1, e2, e3)[1]

        def Z(e1, e2, e3):
            return XYZ(e1, e2, e3)[2]

        cx, cy, cz = interp_mapping(num_elements, degree, spl_kind, X, Y, Z)

        super().__init__(num_elements=num_elements, degree=degree, spl_kind=spl_kind, cx=cx, cy=cy, cz=cz)
