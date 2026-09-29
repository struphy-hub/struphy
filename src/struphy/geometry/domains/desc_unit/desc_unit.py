from struphy.geometry.base import Spline, interp_mapping


class DESCunit(Spline):
    r"""The mapping :math:`(\rho, \theta,\zeta) \mapsto (X, Y, Z)` to Cartesian coordinates computed by the `DESC MHD equilibrium code
    <https://desc-docs.readthedocs.io/en/latest/theory_general.html#flux-coordinates>`_.

    .. image:: ../../pics/mappings/desc.png

    Parameters
    ----------
    desc_equil : struphy.fields_background.equils.DESCequilibrium
        DESC MHD equilibrium object.
    """

    def __init__(self, desc_equil=None):
        from struphy.fields_background.equils import DESCequilibrium

        if desc_equil is None:
            desc_equil = DESCequilibrium()
        else:
            assert isinstance(desc_equil, DESCequilibrium)

        num_elements = desc_equil.params["num_elements"]
        degree = desc_equil.params["degree"]

        if desc_equil.eq.NFP > 1 and desc_equil.use_nfp:
            spl_kind = (False, True, False)
        else:
            spl_kind = (False, True, True)

        _rmin = desc_equil.params["rmin"]

        nfp = desc_equil.eq.NFP
        if not desc_equil.use_nfp:
            nfp = 1

        # project mapping to splines
        def X(e1, e2, e3):
            return desc_equil.desc_eval("X", e1, e2, e3, nfp=nfp)

        def Y(e1, e2, e3):
            return desc_equil.desc_eval("Y", e1, e2, e3, nfp=nfp)

        def Z(e1, e2, e3):
            return desc_equil.desc_eval("Z", e1, e2, e3, nfp=nfp)

        cx, cy, cz = interp_mapping(num_elements, degree, spl_kind, X, Y, Z)

        super().__init__(num_elements=num_elements, degree=degree, spl_kind=spl_kind, cx=cx, cy=cy, cz=cz)
