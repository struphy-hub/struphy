import numpy as np
from mpi4py import MPI

from struphy.models.base import StruphyModel
from struphy.pic.utilities import eval_field_at_particles


#############################
# Fluid models 
#############################
class LinearMHD( StruphyModel ):
    r'''Linear ideal MHD with zero-flow equilibrium (:math:`\mathbf U_0 = 0`). 

    Normalization: 

    .. math::

        \frac{\hat B}{\sqrt{\hat \rho \mu_0}} =: \hat v_\textnormal{A} = \frac{\hat \omega}{\hat k} = \hat U \,, \qquad \hat p = \hat \rho \, \hat v_\textnormal{A}^2\,,

    where :math:`\mu_0` the vacuum permeability. Implemented equations:

    .. math::

        &\frac{\partial \tilde \rho}{\partial t}+\nabla\cdot(\rho_0 \tilde{\mathbf{U}})=0\,, 

        \rho_0&\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde p
        =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_0 + \mathbf{J}_0\times \tilde{\mathbf{B}}
        \,, \qquad
        \mathbf{J}_0 = \nabla\times\mathbf{B}_0\,,

        &\frac{\partial \tilde p}{\partial t} + \nabla\cdot(p_0 \tilde{\mathbf{U}}) 
        + \frac{2}{3}\,p_0\nabla\cdot \tilde{\mathbf{U}}=0\,,

        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_0)
        = 0\,,

    where the equilibrium quantities must satisfy the MHD equilibrium condition :math:`\nabla p_0=\mathbf J_0\times\mathbf B_0`.

    Parameters
    ----------
        params : dict
            Simulation parameters, see from :ref:`params_yml`.
    '''

    def __init__(self, params, comm):

        from struphy.psydac_api.mass import WeightedMassOperators
        from struphy.psydac_api.basis_projection_ops import BasisProjectionOperators
        from struphy.fields_background.mhd_equil import analytical
        from struphy.propagators import propagators_fields

        self._u_space = params['fields']['mhd_u_space']

        if self._u_space == 'Hdiv':
            super().__init__(params, comm, n3='L2', u2=self._u_space, p3='L2', b2='Hdiv')
        elif self._u_space == 'H1vec':
            super().__init__(params, comm, n3='L2', uv=self._u_space, p3='L2', b2='Hdiv')
        else:
            raise ValueError(f'MHD velocity must be in Hdiv or in H1vec, but has been specified in {self._u_space}.')

        # extract necessary parameters
        equil_params = params['fields']['mhd_equilibrium']
        shearalfven_solver = params['solvers']['solver_1']
        magnetosonic_solver = params['solvers']['solver_2']

        # Load MHD equilibrium and project fields
        mhd_equil_class = getattr(analytical, equil_params['type'])
        mhd_equil = mhd_equil_class(
            equil_params[equil_params['type']], self.domain)

        self._b_eq = self.derham.P2(
            [mhd_equil.b2_1, mhd_equil.b2_2, mhd_equil.b2_3]).coeffs
        self._p_eq = self.derham.P3(mhd_equil.p3).coeffs

        self._ones = self.derham.V3.vector_space.zeros()
        self._ones[:] = 1.

        # Assemble necessary mass matrices
        self._mass_ops = WeightedMassOperators(self.derham, self.domain, eq_mhd=mhd_equil)

        # Assemble necessary linear basis projection operators
        self._basis_ops = BasisProjectionOperators(self.derham, self.domain, mhd_equil)

        # Pointers to Stencil-/Blockvectors
        self._n = self.fields[0].vector
        self._u = self.fields[1].vector
        self._p = self.fields[2].vector
        self._b = self.fields[3].vector

        # Initialize propagators/integrators used in splitting substeps
        self._propagators = []
        self._propagators += [propagators_fields.ShearAlfvén(self._u, self._b, self._u_space, self.derham,
                                          self._mass_ops, self._basis_ops, shearalfven_solver)]
        self._propagators += [propagators_fields.Magnetosonic(self._n, self._u, self._p, self._b, self._u_space,
                                           self.derham, self._mass_ops, self._basis_ops, magnetosonic_solver)]

        # Scalar variables to be saved during simulation

        # time
        self._scalar_quantities['time'] = np.empty(1, dtype=float)

        # energies
        self._scalar_quantities['en_U'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_p'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_p_eq'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B_eq'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B_tot'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_tot'] = np.empty(1, dtype=float)

    @property
    def propagators(self):
        return self._propagators

    def update_scalar_quantities(self, time):
        self._scalar_quantities['time'][0] = time

        if self._u_space == 'Hdiv':
            self._scalar_quantities['en_U'][0] = self._u.dot(
                self._mass_ops.M2n.dot(self._u))/2
        else:
            self._scalar_quantities['en_U'][0] = self._u.dot(
                self._mass_ops.Mvn.dot(self._u))/2

        self._scalar_quantities['en_p'][0] = self._p.dot(self._ones)/(5/3 - 1)
        self._scalar_quantities['en_B'][0] = self._b.dot(
            self._mass_ops.M2.dot(self._b))/2

        self._scalar_quantities['en_p_eq'][0] = self._p_eq.dot(
            self._ones)/(5/3 - 1)
        self._scalar_quantities['en_B_eq'][0] = self._b_eq.dot(
            self._mass_ops.M2.dot(self._b_eq))/2

        self._scalar_quantities['en_B_tot'][0] = (
            self._b_eq + self._b).dot(self._mass_ops.M2.dot(self._b_eq + self._b))/2

        self._scalar_quantities['en_tot'][0] = self._scalar_quantities['en_U'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_p'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_B'][0]


#############################
# Fluid-kinetic hybrid models 
#############################
class PC_LinearMHD_Vlasov_full( StruphyModel ):
    r'''Hybrid (Linear ideal MHD + Full-orbit Vlasov) equations with **full pressure coupling scheme**,
    including the parallel pressure tensor. 

    Normalization: 

    .. math::

        \frac{\hat B}{\sqrt{\hat \rho \mu_0}} =: \hat v_\textnormal{A} = \frac{\hat \omega}{\hat k} = \hat U \,, \qquad \hat p = \hat \rho \, \hat v_\textnormal{A}^2\,.

    where :math:`\mu_0` the vacuum permeability. Implemented equations:

    .. math::

        \begin{align}
        \textnormal{linear MHD} &\left\{
        \begin{aligned}
        &\frac{\partial \tilde \rho}{\partial t}+\nabla\cdot(\rho_0 \tilde{\mathbf{U}})=0\,, 
        \\
        &\rho_0\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde p \color{red}+ \nabla\cdot \tilde{\mathbb{P}}_h \color{black} 
        =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_0 + \mathbf{J}_0\times \tilde{\mathbf{B}}
        \,, \qquad
        \mathbf{J}_0 = \nabla\times\mathbf{B}_0\,, 
        \\
        &\frac{\partial \tilde p}{\partial t} + \nabla\cdot(p_0 \tilde{\mathbf{U}}) 
        + \frac{2}{3}\,p_0\nabla\cdot \tilde{\mathbf{U}}=0\,, 
        \\
        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_0)
        = 0\,,
        \end{aligned}
        \right.
        \\[2mm]
        \textnormal{Vlasov}\qquad& \frac{\partial f_h}{\partial t} + (\mathbf{v} \color{red} + \tilde{\mathbf{U}} \color{black})\cdot\frac{\partial f_h}{\partial \mathbf{x}}
        + \left[\frac{q_h}{m_h}\mathbf{v}\times(\mathbf{B}_0 + \tilde{\mathbf{B}}) \color{red}- \nabla \tilde{\mathbf{U}}\cdot \mathbf{v} \color{black} \right]\cdot\frac{\partial f_h}{\partial \mathbf{v}}
        = 0\,,
        \\
        &\color{red} \tilde{\mathbb{P}}_h = \int \mathbf{v}\mathbf{v}^\top f_h d\mathbf{v} \color{black}\,.
        \end{align}

    Parameters
    ----------
        params : dict
            Simulation parameters, see from :ref:`params_yml`.
    '''

    def __init__(self, params, comm):

        from struphy.psydac_api.mass import WeightedMassOperators
        from struphy.psydac_api.basis_projection_ops import BasisProjectionOperators
        from struphy.fields_background.mhd_equil import analytical
        from struphy.propagators import propagators_markers, propagators_fields, propagators_coupling

        self._u_space = params['fields']['mhd_u_space']
        super().__init__(params, comm, n3='L2', u1=self._u_space,
                         p3='L2', b2='Hdiv', energetic_ions='Particles6D')

        # extract necessary parameters
        equil_params = params['fields']['mhd_equilibrium']
        alfven_solver = params['solvers']['solver_1']
        magnetosonic_solver = params['solvers']['solver_2']
        pressurecoupling_solver = params['solvers']['solver_2']
        nuh = params['kinetic']['energetic_ions']['attributes']['nuh']
        self._nuh = nuh
        self._comm = self.derham.comm

        # Load MHD equilibrium and project fields
        mhd_equil_class = getattr(analytical, equil_params['type'])
        mhd_equil = mhd_equil_class(
            equil_params[equil_params['type']], self.domain)

        self._b_eq = self.derham.P2(
            [mhd_equil.b2_1, mhd_equil.b2_2, mhd_equil.b2_3]).coeffs

        # Assemble necessary mass matrices
        self._mass_ops = WeightedMassOperators(self.derham, self.domain, eq_mhd=mhd_equil)

        # Assemble necessary linear basis projection operators
        self._basis_ops = BasisProjectionOperators(self.derham, self.domain, mhd_equil)

        # Pointers to Stencil-/Blockvectors
        self._n = self.fields[0].vector
        self._u = self.fields[1].vector
        self._p = self.fields[2].vector
        self._b = self.fields[3].vector

        # Initialize propagators/integrators used in splitting substeps
        if self._u_space == 'Hcurl':
            Pressurecoupling = getattr(propagators_coupling, 'StepFullPressurecouplingHcurl')
            PushEta = getattr(propagators_markers, 'StepPushEtaFullPC')
            PushVel = getattr(propagators_markers, 'StepPushVxB')
        elif self._u_space == 'Hdiv':
            Pressurecoupling = getattr(propagators_coupling, 'StepFullPressurecouplingHdiv')
            PushEta = getattr(propagators_markers, 'StepPushEtaFullPC')
            PushVel = getattr(propagators_markers, 'StepPushVxB')
        elif self._u_space == 'H1vec':
            Pressurecoupling = getattr(propagators_coupling, 'StepFullPressurecouplingH1vec')
            PushEta = getattr(propagators_markers, 'StepPushEtaFullPC')
            PushVel = getattr(propagators_markers, 'StepPushVxB')

        self._propagators = []
        self._propagators += [propagators_fields.ShearAlfvén(self._u, self._b, self._u_space, self.derham,
                                     self._mass_ops, self._basis_ops, alfven_solver)]
        # self._propagators += [propagators_fields.Magnetosonic(self._n, self._u, self._p, self._b, self._u_space, self.derham, self._mass_ops, self._basis_ops, magnetosonic_solver)]
        for particles in self._kinetic_species:
            self._propagators += [PushEta(self._u, particles,
                                          self.derham, self.domain, self._u_space)]
            self._propagators += [Pressurecoupling(self._u, particles, self.derham,
                                                   self.domain, self._mass_ops, self._basis_ops, pressurecoupling_solver)]
            self._propagators += [PushVel(particles,
                                          self.derham, self._b, self._b_eq)]

        # Scalar variables to be saved during simulation
        self._scalar_quantities['time'] = np.empty(1, dtype=float)

        self._scalar_quantities['en_U'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_p'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B'] = np.empty(1, dtype=float)
        self._en_f_loc = np.empty(1, dtype=float)
        self._scalar_quantities['en_f'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_tot'] = np.empty(1, dtype=float)

    @property
    def propagators(self):
        return self._propagators

    def update_scalar_quantities(self, time):
        self._scalar_quantities['time'][0] = time

        if self._u_space == 'Hcurl':
            self._en_U_loc = self._u.dot(self._mass_ops.M1n.dot(self._u))/2
            self._scalar_quantities['en_U'][0] = self._u.dot(
                self._mass_ops.M1n.dot(self._u))/2
            self._scalar_quantities['en_p'][0] = self._p.toarray(
            ).sum()/(5/3 - 1)
        elif self._u_space == 'Hdiv':
            self._scalar_quantities['en_U'][0] = self._u.dot(
                self._mass_ops.M2n.dot(self._u))/2
            self._scalar_quantities['en_p'][0] = self._p.toarray(
            ).sum()/(5/3 - 1)
        else:
            self._scalar_quantities['en_U'][0] = self._u.dot(
                self._mass_ops.Mvn.dot(self._u))/2
            self._scalar_quantities['en_p'][0] = self._p.toarray(
            ).sum()/(5/3 - 1)

        self._scalar_quantities['en_B'][0] = self._b.dot(
            self._mass_ops.M2.dot(self._b))/2

        for particles in self._kinetic_species:
            self._en_f_loc = particles.markers[~particles.holes, 8].dot(particles.markers[~particles.holes, 3]**2
                                                                        + particles.markers[~particles.holes, 4]**2
                                                                        + particles.markers[~particles.holes, 5]**2
                                                                        )/(2. * particles.n_mks)

            self._comm.Reduce(
                self._en_f_loc, self._scalar_quantities['en_f'], op=MPI.SUM, root=0)

        self._scalar_quantities['en_tot'][0] = self._scalar_quantities['en_U'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_p'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_B'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_f'][0]


class PC_LinearMHD_Vlasov( StruphyModel ):
    r'''Hybrid (Linear ideal MHD + Full-orbit Vlasov) equations with **pressure coupling scheme**. 

    Normalization: 

    .. math::

        \frac{\hat B^2}{\hat \rho \mu_0} =: \hat v_\textnormal{A} = \frac{\hat \omega}{\hat k} = \hat U \,, \qquad \hat p = \hat \rho \, \hat v_\textnormal{A}^2\,.

    Implemented equations:

    .. math::

        \begin{align}
        \textnormal{linear MHD} &\left\{
        \begin{aligned}
        &\frac{\partial \tilde \rho}{\partial t}+\nabla\cdot(\rho_0 \tilde{\mathbf{U}})=0\,, 
        \\
        &\rho_0\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde p \color{red}+ \nabla\cdot \tilde{\mathbb{P}}_{h,\perp} \color{black} 
        =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_0 + \mathbf{J}_0\times \tilde{\mathbf{B}}
        \,, \qquad
        \mathbf{J}_0 = \nabla\times\mathbf{B}_0\,, 
        \\
        &\frac{\partial \tilde p}{\partial t} + \nabla\cdot(p_0 \tilde{\mathbf{U}}) 
        + \frac{2}{3}\,p_0\nabla\cdot \tilde{\mathbf{U}}=0\,, 
        \\
        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_0)
        = 0\,,
        \end{aligned}
        \right.
        \\[2mm]
        \textnormal{Vlasov}\qquad& \frac{\partial f_h}{\partial t} + (\mathbf{v} \color{red} + \tilde{\mathbf{U}}_\perp \color{black})\cdot\frac{\partial f_h}{\partial \mathbf{x}}
        + \left[\frac{q_h}{m_h}\mathbf{v}\times(\mathbf{B}_0 + \tilde{\mathbf{B}}) \color{red}- \nabla \tilde{\mathbf{U}}_\perp\cdot \mathbf{v} \color{black} \right]\cdot\frac{\partial f_h}{\partial \mathbf{v}}
        = 0\,,
        \\
        &\color{red} \tilde{\mathbb{P}}_{h,\perp} = \int \mathbf{v}_\perp\mathbf{v}^\top_\perp f_h d\mathbf{v} \color{black}\,.
        \end{align}

    Parameters
    ----------
        params : dict
            Simulation parameters, see from :ref:`params_yml`.
    '''

    def __init__(self, params, comm):

        from struphy.psydac_api.mass import WeightedMassOperators
        from struphy.psydac_api.basis_projection_ops import BasisProjectionOperators
        from struphy.fields_background.mhd_equil import analytical
        from struphy.propagators import propagators_fields, propagators_markers, propagators_coupling

        self._u_space = params['fields']['mhd_u_space']
        super().__init__(params, comm, n3='L2', u1=self._u_space,
                         p3='L2', b2='Hdiv', energetic_ions='Particles6D')

        # extract necessary parameters
        equil_params = params['fields']['mhd_equilibrium']
        alfven_solver = params['solvers']['solver_1']
        magnetosonic_solver = params['solvers']['solver_2']
        pressurecoupling_solver = params['solvers']['solver_2']
        nuh = params['kinetic']['energetic_ions']['attributes']['nuh']
        self._nuh = nuh
        self._comm = self.derham.comm

        # Load MHD equilibrium and project fields
        mhd_equil_class = getattr(analytical, equil_params['type'])
        mhd_equil = mhd_equil_class(
            equil_params[equil_params['type']], self.domain)

        self._b_eq = self.derham.P2(
            [mhd_equil.b2_1, mhd_equil.b2_2, mhd_equil.b2_3]).coeffs

        # Assemble necessary mass matrices
        self._mass_ops = WeightedMassOperators(self.derham, self.domain, eq_mhd=mhd_equil)

        # Assemble necessary linear basis projection operators
        self._basis_ops = BasisProjectionOperators(self.derham, self.domain, mhd_equil)

        # Pointers to Stencil-/Blockvectors
        self._n = self.fields[0].vector
        self._u = self.fields[1].vector
        self._p = self.fields[2].vector
        self._b = self.fields[3].vector

        # Initialize propagators/integrators used in splitting substeps
        if self._u_space == 'Hcurl':
            Pressurecoupling = getattr(propagators_coupling, 'StepPressurecouplingHcurl')
            PushEta = getattr(propagators_markers, 'StepPushEtaPC')
            PushVel = getattr(propagators_markers, 'StepPushVxB')
        elif self._u_space == 'Hdiv':
            Pressurecoupling = getattr(propagators_coupling, 'StepPressurecouplingHdiv')
            PushEta = getattr(propagators_markers, 'StepPushEtaPC')
            PushVel = getattr(propagators_markers, 'StepPushVxB')
        elif self._u_space == 'H1vec':
            Pressurecoupling = getattr(propagators_coupling, 'StepPressurecouplingH1vec')
            PushEta = getattr(propagators_markers, 'StepPushEtaPC')
            PushVel = getattr(propagators_markers, 'StepPushVxB')

        self._propagators = []
        self._propagators += [propagators_fields.ShearAlfvén(self._u, self._b, self._u_space, self.derham,
                                     self._mass_ops, self._basis_ops, alfven_solver)]
        # self._propagators += [propagators_fields.Magnetosonic(self._n, self._u, self._p, self._b, self._u_space, self.derham, self._mass_ops, self._basis_ops, magnetosonic_solver)]
        for particles in self._kinetic_species:
            self._propagators += [PushEta(self._u, particles,
                                          self.derham, self.domain, self._u_space)]
            self._propagators += [Pressurecoupling(self._u, particles, self.derham,
                                                   self.domain, self._mass_ops, self._basis_ops, pressurecoupling_solver)]
            self._propagators += [PushVel(particles,
                                          self.derham, self._b, self._b_eq)]

        # Scalar variables to be saved during simulation
        self._scalar_quantities['time'] = np.empty(1, dtype=float)

        self._scalar_quantities['en_U'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_p'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B'] = np.empty(1, dtype=float)
        self._en_f_loc = np.empty(1, dtype=float)
        self._scalar_quantities['en_f'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_tot'] = np.empty(1, dtype=float)

    @property
    def propagators(self):
        return self._propagators

    def update_scalar_quantities(self, time):
        self._scalar_quantities['time'][0] = time

        if self._u_space == 'Hcurl':
            self._en_U_loc = self._u.dot(self._mass_ops.M1n.dot(self._u))/2
            self._scalar_quantities['en_U'][0] = self._u.dot(
                self._mass_ops.M1n.dot(self._u))/2
            self._scalar_quantities['en_p'][0] = self._p.toarray(
            ).sum()/(5/3 - 1)
        elif self._u_space == 'Hdiv':
            self._scalar_quantities['en_U'][0] = self._u.dot(
                self._mass_ops.M2n.dot(self._u))/2
            self._scalar_quantities['en_p'][0] = self._p.toarray(
            ).sum()/(5/3 - 1)
        else:
            self._scalar_quantities['en_U'][0] = self._u.dot(
                self._mass_ops.Mvn.dot(self._u))/2
            self._scalar_quantities['en_p'][0] = self._p.toarray(
            ).sum()/(5/3 - 1)

        self._scalar_quantities['en_B'][0] = self._b.dot(
            self._mass_ops.M2.dot(self._b))/2

        for particles in self._kinetic_species:
            self._en_f_loc = particles.markers[~particles.holes, 8].dot(particles.markers[~particles.holes, 3]**2
                                                                        + particles.markers[~particles.holes, 4]**2
                                                                        + particles.markers[~particles.holes, 5]**2
                                                                        )/(2. * particles.n_mks)

            self._comm.Reduce(
                self._en_f_loc, self._scalar_quantities['en_f'], op=MPI.SUM, root=0)

        self._scalar_quantities['en_tot'][0] = self._scalar_quantities['en_U'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_p'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_B'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_f'][0]


#############################
# Kinetic models 
#############################
class LinearVlasovMaxwell( StruphyModel ):
    r'''The linearized Vlasov Maxwell system with a Maxwellian background distribution function
    is described by the following equations:

    .. math::

        \frac{\partial \mathbf{E}}{\partial t} & = \nabla \times \mathbf{B} -
        \sum_p w_p \sqrt{f_{0,p}(\mathbf{x}_p, \mathbf{v}_p)} \mathbf{v}_p \,,

        \frac{\partial \mathbf{B}}{\partial t} & = - \nabla \times \mathbf{E} \,,

        \frac{\text{d} \mathbf{x}_p}{\text{d} t} & = \mathbf{v}_p \,,

        \frac{\text{d} \mathbf{v}_p}{\text{d} t} & = \mathbf{E}_0 + \mathbf{v}_p \times \mathbf{B}_0 \,,

        \frac{\text{d} w_p}{\text{d} t} & = \frac{1}{v_{\text{th},p}^2} \,
        \sqrt{f_{0,p}(\mathbf{x}_p, \mathbf{v}_p)} \, \mathbf{E} \cdot \mathbf{v}_p

    which form a Hamiltonian system with the energies:

    .. math::

        H_0(t) & = \sum_p \left( \frac{\mathbf{v}_p^2}{2} + \phi_0(\mathbf{x}_p) \right) \,,

        H_h(t) & = \sum_p \frac{v_{\text{th},p}^2 w_p^2}{2}
        + \frac{1}{2} \int_\Omega |\mathbf{E}|^2 \, \text{d}^3 \mathbf{x}
        + \frac{1}{2} \int_\Omega |\mathbf{B}|^2 \, \text{d}^3 \mathbf{x} \,.

    All natural constants are set equal to 1 and all particles are normalized
    to have unit mass and unit charge.

    Parameters
    ----------
        params : dict
            Simulation parameters, see from :ref:`params_yml`.
    '''

    def __init__(self, params, comm):

        from struphy.psydac_api.mass import WeightedMassOperators
        from struphy.propagators import propagators_fields, propagators_markers, propagators_coupling
        from struphy.psydac_api.fields import Field
        from struphy.fields_background.mhd_equil import analytical

        super().__init__(params, comm, e_field='Hcurl',
                         b_field='Hdiv', electrons='Particles6D')

        # extract necessary parameters
        solver_params = params['solvers']['solver_1']

        # Assemble necessary mass matrices
        self._mass_ops = WeightedMassOperators(self.derham, self.domain)

        # Pointers to Stencil-/Blockvectors
        self._e = self.fields[0].vector
        self._b = self.fields[1].vector

        self._electrons = self.kinetic_species[0]

        # ====================================================================================
        # Instantiate background electric field and potential
        self._background_fields = []
        self._background_fields += [Field('e_background',
                                          'Hcurl', self.derham)]
        self._background_fields += [Field('phi_background', 'H1', self.derham)]

        self._background_fields[1].set_initial_conditions(
            self.domain, [True], params['fields']['init'])

        self._e_background = self._background_fields[0].vector
        self._phi_background = self._background_fields[1].vector

        self._e_background = self.derham.grad.dot(self._phi_background)

        # Initialize background magnetic field from MHD equilibrium
        self._background_fields += [Field('b_background', 'Hdiv', self.derham)]
        self._b_background = self._background_fields[2].vector

        # Create MHD equilibrium
        equil_params = params['fields']['mhd_equilibrium']
        mhd_equil_class = getattr(analytical, equil_params['type'])
        mhd_equil = mhd_equil_class(
            equil_params[equil_params['type']], self.domain)

        # self._b_background[0] =
        self._b_background = self.derham.P2(
            [mhd_equil.b_x, mhd_equil.b_y, mhd_equil.b_z]).coeffs
        # ====================================================================================

        # Initialize propagators/integrators used in splitting substeps
        self._propagators = []
        self._propagators += [propagators_markers.StepStaticEfield(
            self.domain, self.derham, self._electrons, self._e_background)]
        self._propagators += [propagators_markers.StepStaticBfield(
            self.domain, self.derham, self._electrons, self._b_background)]
        self._propagators += [propagators_coupling.StepEfieldWeights(self.domain, self.derham, self._e, self._electrons, self._mass_ops,
                                                params['kinetic']['electrons']['background'], params['solvers']['solver_1'])]
        self._propagators += [propagators_fields.Maxwell(self._e, self._b,
                                          self.derham, self._mass_ops, solver_params)]

        # Scalar variables to be saved during simulation
        self._scalar_quantities['time'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_E'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_weights'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_all'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_el_pot'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_kin'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_sing'] = np.empty(1, dtype=float)

    @property
    def propagators(self):
        return self._propagators

    def _compute_electric_potential(self):
        ''' Compute the sum of the electric potential at all particle positions '''

        res = eval_field_at_particles(
            self._phi_background, self._derham, 'H1', self._electrons)

        return res

    def update_scalar_quantities(self, time):
        self._scalar_quantities['time'][0] = time
        self._scalar_quantities['en_E'][0] = self._e.dot(
            self._mass_ops.M1.dot(self._e)) / 2.
        self._scalar_quantities['en_B'][0] = self._b.dot(
            self._mass_ops.M2.dot(self._b)) / 2.
        self._scalar_quantities['en_weights'][0] = np.sum(
            self._electrons.markers[:, 8])**2
        self._scalar_quantities['en_all'][0] = self._scalar_quantities['en_weights'][0] + \
            self._scalar_quantities['en_E'][0] + \
            self._scalar_quantities['en_B'][0]
        self._scalar_quantities['en_el_pot'][0] = self._compute_electric_potential(
        )
        self._scalar_quantities['en_kin'][0] = np.sum(
            np.sum(self._electrons.markers[:, 3:6], axis=1)**2)
        self._scalar_quantities['en_sing'][0] = self._scalar_quantities['en_el_pot'][0] + \
            self._scalar_quantities['en_kin'][0]


#############################
# Toy models 
#############################
class Maxwell( StruphyModel ):
    r'''Maxwell's equations in vacuum. 

    Normalization:

    .. math::

        c = \frac{\hat \omega}{\hat k} = \frac{\hat E}{\hat B}\,,

    where :math:`c` is the vacuum speed of light. Implemented equations:

    .. math::

        &\frac{\partial \mathbf E}{\partial t} - \nabla\times\mathbf B = 0\,, 

        &\frac{\partial \mathbf B}{\partial t} + \nabla\times\mathbf E = 0\,.

    Parameters
    ----------
        params : dict
            Simulation parameters, see from :ref:`params_yml`.
    '''

    def __init__(self, params, comm):

        from struphy.psydac_api.mass import WeightedMassOperators
        from struphy.propagators import propagators_fields

        super().__init__(params, comm, e_field='Hcurl', b_field='Hdiv')

        # extract necessary parameters
        solver_params = params['solvers']['solver_1']

        # Assemble necessary mass matrices
        self._mass_ops = WeightedMassOperators(self.derham, self.domain)

        # Pointers to Stencil-/Blockvectors
        self._e = self.fields[0].vector
        self._b = self.fields[1].vector

        # Initialize propagators/integrators used in splitting substeps
        self._propagators = []
        self._propagators += [propagators_fields.Maxwell(self._e, self._b, self.derham, self._mass_ops, solver_params)]

        # Scalar variables to be saved during simulation
        self._scalar_quantities['time'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_E'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_tot'] = np.empty(1, dtype=float)

    @property
    def propagators(self):
        return self._propagators

    def update_scalar_quantities(self, time):
        self._scalar_quantities['time'][0] = time
        self._scalar_quantities['en_E'][0] = .5 * \
            self._e.dot(self._mass_ops.M1.dot(self._e))
        self._scalar_quantities['en_B'][0] = .5 * \
            self._b.dot(self._mass_ops.M2.dot(self._b))
        self._scalar_quantities['en_tot'][0] = self._scalar_quantities['en_E'][0] + \
            self._scalar_quantities['en_B'][0]


class Vlasov( StruphyModel ):
    r'''Vlasov equation in static background magnetic field. 

    Normalization:

    .. math::

        &\hat \omega = \frac{q \hat B}{m} = \hat \Omega_{c} \,,
        
        &\hat v = \frac{\hat \omega}{\hat k} = \frac{\hat \Omega_{c}}{\hat k} \,.

    where :math:`\Omega_{c}` is cyclotron frequency. Implemented equations:

    .. math::

        \frac{\partial f}{\partial t} + \mathbf{v} \cdot \frac{\partial f}{\partial \mathbf{x}} + \left[\frac{q_h}{m_h}\mathbf{v}\times\mathbf{B}_0 \right] \cdot \frac{\partial f}{\partial \mathbf{v}} = 0\,.

    Parameters
    ----------
        params : dict
            Simulation parameters, see from :ref:`params_yml`.
    '''

    def __init__(self, params, comm):

        from struphy.propagators import propagators_markers
        from struphy.fields_background.mhd_equil import analytical

        super().__init__(params, comm, ions='Particles6D')

        print(
            f'Total number of markers : {self.kinetic_species[0].n_mks}, shape of markers array on rank {self.derham.comm.Get_rank()} : {self.kinetic_species[0].markers.shape}')

        # Load and project magnetic field
        equil_params = params['fields']['mhd_equilibrium']
        mhd_equil_class = getattr(analytical, equil_params['type'])
        self._mhd_equil = mhd_equil_class(
            equil_params[equil_params['type']], self.domain)

        if self.derham.comm.Get_rank() == 0:
            print('Start of background magnetic field projection ...')
        self._b = self.derham.P2([self._mhd_equil.b2_1,
                                  self._mhd_equil.b2_2,
                                  self._mhd_equil.b2_3]).coeffs
        if self.derham.comm.Get_rank() == 0:
            print('Background magnetic field projection done ...')

        # Pointer to ions
        self._ions = self.kinetic_species[0]

        # Initialize propagators/integrators used in splitting substeps
        self._propagators = []
        self._propagators += [propagators_markers.StepPushVxB(self._ions, self.derham, 
                                          self.kinetic_params[0]['push_algos']['vxb'], self._b)]
        self._propagators += [propagators_markers.StepPushEta(self._ions, self.derham, 
                                          self.kinetic_params[0]['push_algos']['eta'], 
                                          self.kinetic_params[0]['markers']['bc_type'])]

        # Scalar variables to be saved during simulation
        self._scalar_quantities['time'] = np.empty(1, dtype=float)

    @property
    def propagators(self):
        return self._propagators

    def update_scalar_quantities(self, time):
        self._scalar_quantities['time'][0] = time


class DriftKinetic( StruphyModel ):
    r'''DriftKinetic equation in static background magnetic field. 

    Normalization:

    .. math::

        &\hat \omega = \hat v_{th} * \hat k = \hat \Omega_{th} = \epsilon \hat \Omega_c \,,
        
        &\hat v = \hat v_{th} = \frac{\hat \Omega_{th}}{\hat k} = \epsilon \frac{\hat \Omega_c}{\hat k} \,.

    where :math:`\Omega_{c}` is cyclotron frequency and :math:`\epsilon = \frac{\hat \Omega_{th}}{\hat \Omega_c} = \hat k \rho_L \ll 1` is a drift time scale. Implemented equations:
    
    .. math::

        \frac{\partial f}{\partial t} + \left[ v_\parallel \frac{\mathbf{B}^*}{B^*_\parallel} + \epsilon \frac{\mathbf{E}^* \times \mathbf{b}_0}{B^*_\parallel}\right] \cdot \frac{\partial f}{\partial \mathbf{x}} + \left[ \frac{\mathbf{B}^*}{B^*_\parallel} \cdot \mathbf{E}^*\right] \cdot \frac{\partial f}{\partial v_\parallel} = 0\,.

    where :math:`f(\mathbf{X}, v_\parallel, t)` is the guiding center distribution and 

    .. math::

        &\mathbf{E}^* = - \mu \nabla |B_0| \,,  
        
        &\mathbf{B}^* = \mathbf{B}_0 + \epsilon v_\parallel \nabla \times \mathbf{b}_0  \,.
    
    Parameters
    ----------
        params : dict
            Simulation parameters, see from :ref:`params_yml`.
    '''

    def __init__(self, params, comm):

        from struphy.propagators import propagators_markers
        from struphy.fields_background.mhd_equil import analytical

        super().__init__(params, comm, ions='Particles5D') #TODO:particles.Particles5D

        print(
            f'Total number of markers : {self.kinetic_species[0].n_mks}, shape of markers array on rank {self.derham.comm.Get_rank()} : {self.kinetic_species[0].markers.shape}')

        # Load and project magnetic field
        equil_params = params['fields']['mhd_equilibrium']
        mhd_equil_class = getattr(analytical, equil_params['type'])
        self._mhd_equil = mhd_equil_class(
            equil_params[equil_params['type']], self.domain)

        if self.derham.comm.Get_rank() == 0:
            print('Start of background magnetic field projection ...')
        self._b = self.derham.P2([self._mhd_equil.b2_1,
                                  self._mhd_equil.b2_2,
                                  self._mhd_equil.b2_3]).coeffs

        # print(self.derham.comm.Get_rank(), self._b._data)

        self._abs_b = self.derham.P0(self._mhd_equil.b0)._coeffs

        # print(self.derham.comm.Get_rank(), self._abs_b._data)

        self._norm_b1 = self.derham.P1([self._mhd_equil.norm_b1_1,
                                        self._mhd_equil.norm_b1_2,
                                        self._mhd_equil.norm_b1_3]).coeffs
                                        
        # print(self.derham.comm.Get_rank(), self._norm_b1[0]._data)

        self._norm_b2 = self.derham.P2([self._mhd_equil.norm_b2_1,
                                        self._mhd_equil.norm_b2_2,
                                        self._mhd_equil.norm_b2_3]).coeffs

        # print(self.derham.comm.Get_rank(), self._norm_b2[0]._data)

        if self.derham.comm.Get_rank() == 0:
            print('Background magnetic field projection done ...')

        # Pointer to ions
        self._ions = self.kinetic_species[0]

        # guiding center scale factor
        self._epsilon = self.kinetic_params[0]['attributes']['epsilon']

        # Initialize propagators/integrators used in splitting substeps
        self._propagators = []
        self._propagators += [propagators_markers.StepPushGuidingCenter1(self._ions, self._epsilon,
                                                                         self._b, self._norm_b1, self._norm_b2, self._abs_b, 
                                                                         self.derham, 
                                                                         self.kinetic_params[0]['push_algos']['method'], 
                                                                         self.kinetic_params[0]['push_algos']['integrator'],
                                                                         self.kinetic_params[0]['markers']['bc_type'],
                                                                         self.kinetic_params[0]['push_algos']['maxiter'],
                                                                         self.kinetic_params[0]['push_algos']['tol'])]
        self._propagators += [propagators_markers.StepPushGuidingCenter2(self._ions, self._epsilon,
                                                                         self._b, self._norm_b1, self._norm_b2, self._abs_b, 
                                                                         self.derham, 
                                                                         self.kinetic_params[0]['push_algos']['method'], 
                                                                         self.kinetic_params[0]['push_algos']['integrator'],
                                                                         self.kinetic_params[0]['markers']['bc_type'],
                                                                         self.kinetic_params[0]['push_algos']['maxiter'],
                                                                         self.kinetic_params[0]['push_algos']['tol'])]
        # self._propagators += [propagators_markers.StepPushGuidingCenter(self._ions, self._epsilon,
        #                                                                 self._b, self._norm_b1, self._norm_b2, self._abs_b, 
        #                                                                 self.derham, 
        #                                                                 self.kinetic_params[0]['push_algos']['method'], 
        #                                                                  self.kinetic_params[0]['push_algos']['integrator'],
        #                                                                  self.kinetic_params[0]['markers']['bc_type'],
        #                                                                  self.kinetic_params[0]['push_algos']['maxiter'],
        #                                                                  self.kinetic_params[0]['push_algos']['tol'])]

        # Scalar variables to be saved during simulation
        self._scalar_quantities['time'] = np.empty(1, dtype=float)

    @property
    def propagators(self):
        return self._propagators

    def update_scalar_quantities(self, time):
        self._scalar_quantities['time'][0] = time