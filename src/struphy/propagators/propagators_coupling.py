'Particle and FEEC variables are updated.'


import numpy as np

from psydac.linalg.solvers import inverse
from psydac.linalg.block import BlockVector

from struphy.propagators.base import Propagator
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.pic.accumulation.particles_to_grid import Accumulator, AccumulatorVector
from struphy.pic.pushing.pusher import Pusher
import struphy.pic.utilities_kernels as utilities
from struphy.polar.basic import PolarVector
from struphy.kinetic_background.base import Maxwellian
from struphy.kinetic_background.maxwellians import Maxwellian6DUniform, Maxwellian5DUniform
from struphy.fields_background.mhd_equil.equils import set_defaults

from struphy.feec import preconditioner
from struphy.feec.linear_operators import LinOpWithTransp
from struphy.feec.mass import WeightedMassOperator


class VlasovMaxwell(Propagator):
    r'''Solve the following Crank-Nicolson step

    .. math::

        \begin{bmatrix}
            \mathbb{M}_1 \left( \mathbf{e}^{n+1} - \mathbf{e}^n \right) \\
            \mathbf{V}^{n+1} - \mathbf{V}^n
        \end{bmatrix}
        =
        \frac{\Delta t}{2}
        \begin{bmatrix}
            0 & - c_1 \left(\mathbb{\Lambda}^1\right)^\top \overline{DF^{-1}} \mathbb{W} \\
            c_2 \overline{DF^{-\top}} \mathbb{\Lambda}^1 & 0
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf{e}^{n+1} + \mathbf{e}^n \\
            \mathbf{V}^{n+1} + \mathbf{V}^n
        \end{bmatrix}

    based on the :ref:`Schur complement <schur_solver>` where

    .. math::
        \begin{align}
        \mathbb{W} & = \text{diag}(w_p) \,,
        \end{align}

    and the accumulation matrix writes

    .. math::
        \mathbb{A} = -\frac{{\Delta t}^2}{4} c_1 c_2 \, \left( \mathbb{\Lambda}^1 \right)^\top \overline{G^{-1}} \mathbb{W} \mathbb{\Lambda}^1 \,.

    Parameters
    ---------- 
    e : psydac.linalg.block.BlockVector
        FE coefficients of a 1-form.

    particles : struphy.pic.particles.Particles6D
        Particles object.

    **params : dict
        Solver- and/or other parameters for this splitting step.

    Note
    ----------
    For VlasovMaxwell :math:`c_1 = \alpha^2/\varepsilon \,, \, c_2 = 1/\varepsilon`
    '''

    def __init__(self, e, particles, **params):

        super().__init__(e, particles)

        # parameters
        params_default = {'c1': 1.,
                          'c2': 1.,
                          'type': ('pcg', 'MassMatrixPreconditioner'),
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False, }

        params = set_defaults(params, params_default)

        self._c1 = params['c1']
        self._c2 = params['c2']
        self._info = params['info']

        # Initialize Accumulator object
        self._accum = Accumulator(self.derham, self.domain, 'Hcurl', 'vlasov_maxwell',
                                  add_vector=True, symmetry='symm')

        # Create buffers to store temporarily _e and its sum with old e
        self._e_temp = e.space.zeros()
        self._e_sum = e.space.zeros()

        # store old weights to compute difference
        self._old_v_sq = np.empty(particles.markers.shape[0], dtype=float)
        self._new_v_sq = np.empty(particles.markers.shape[0], dtype=float)

        # ================================
        # ========= Schur Solver =========
        # ================================

        # Preconditioner
        if params['type'][1] == None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            pc = pc_class(self.mass_ops.M1)

        # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = self.mass_ops.M1
        _BC = - self._accum.operators[0].matrix / 4.

        # Instantiate Schur solver
        self._schur_solver = SchurSolver(_A, _BC,
                                         params['type'][0],
                                         pc=pc,
                                         tol=params['tol'],
                                         maxiter=params['maxiter'],
                                         verbose=params['verbose'])

        # Instantiate particle pusher
        self._pusher = Pusher(self.derham, self.domain,
                              'push_v_with_efield')

    def __call__(self, dt):
        # current e-field
        en = self.feec_vars[0]

        # accumulate
        self._accum.accumulate(self.particles[0])

        # Update Schur solver
        self._schur_solver.BC = - self._c1 * self._c2 / \
            4. * self._accum.operators[0].matrix

        # new e coeffs
        en1, info = self._schur_solver(
            en, self._c1 / 2. * self._accum.vectors[0], dt, out=self._e_temp)

        # Store old velocity magnitudes
        self._old_v_sq[~self.particles[0].holes] = np.sqrt(self.particles[0].markers[~self.particles[0].holes, 3]**2 +
                                                           self.particles[0].markers[~self.particles[0].holes, 4]**2 +
                                                           self.particles[0].markers[~self.particles[0].holes, 5]**2)

        # mid-point e-field (no tmps created here)
        _e = en.copy(out=self._e_sum)
        _e += en1
        _e *= 0.5

        # Update velocities
        self._pusher(self.particles[0], dt,
                     _e.blocks[0]._data,
                     _e.blocks[1]._data,
                     _e.blocks[2]._data,
                     self._c2)

        # Store new velocity magnitudes
        self._new_v_sq[~self.particles[0].holes] = np.sqrt(self.particles[0].markers[~self.particles[0].holes, 3]**2 +
                                                           self.particles[0].markers[~self.particles[0].holes, 4]**2 +
                                                           self.particles[0].markers[~self.particles[0].holes, 5]**2)

        # update_weights
        if self.particles[0].control_variate:
            self.particles[0].update_weights()

        # write new coeffs into self.variables
        max_de, = self.feec_vars_update(en1)

        # Print out max differences for weights and e-field
        if self._info:
            print('Status      for VlasovMaxwell:', info['success'])
            print('Iterations  for VlasovMaxwell:', info['niter'])
            print('Maxdiff e1  for VlasovMaxwell:', max_de)
            max_diff = np.max(np.abs(self._old_v_sq[~self.particles[0].holes]
                                     - self._new_v_sq[~self.particles[0].holes]))
            print('Maxdiff |v| for VlasovMaxwell:', max_diff)
            print()

    @classmethod
    def options(cls):
        dct = {}
        dct['solver'] = {'type': [('pcg', 'MassMatrixPreconditioner'),
                                  ('cg', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False}
        return dct


class EfieldWeightsImplicit(Propagator):
    r""" Solve the following Semi-Crank-Nicolson step

    .. math::

        \begin{bmatrix}
            \mathbb{M}_1 \left( \mathbf{e}^{n+1} - \mathbf{e}^n \right) \\
            \mathbf{W}^{n+1} - \mathbf{W}^n
        \end{bmatrix}
        =
        \frac{\Delta t}{2}
        \begin{bmatrix}
            0 & - \mathbb{E} \\
            \mathbb{W} & 0
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf{e}^{n+1} + \mathbf{e}^n \\
            \mathbf{W}^{n+1} + \mathbf{W}^n
        \end{bmatrix}

    based on the :ref:`Schur complement <schur_solver>` where for the linearized Vlasov-Maxwell the matrices are

    .. math::
        \begin{align}
        (\mathbb{W})_p & = \frac{1}{N \, s_{0, p}} \frac{1}{v_{\text{th}}^2} \sqrt{f_0}
            \left( DF^{-1} \mathbf{v}_p \right) \cdot \left( \mathbb{\Lambda}^1 \right)^T \,,
            \\
        (\mathbb{E})_p & = \alpha^2 \sqrt{f_0} \, \mathbb{\Lambda}^1 \cdot \left( DF^{-1} \mathbf{v}_p \right) \,.
        \end{align}

    and for the Vlasov-Maxwell with delta-f method the matrices are

    .. math::
        \begin{align}
        (\mathbb{W})_p & = \frac{1}{N \, s_{0, p}} \frac{1}{v_{\text{th}}^2} \sqrt{f_0}
            \left( DF^{-1} \mathbf{v}_p \right) \cdot \left( \mathbb{\Lambda}^1 \right)^T \,,
            \\
        (\mathbb{E})_p & = \alpha^2 \sqrt{f_0} \, \mathbb{\Lambda}^1 \cdot \left( DF^{-1} \mathbf{v}_p \right) \,.
        \end{align}

    which make up the accumulation matrix :math:`\mathbb{E} \mathbb{W}` .

    Parameters
    ---------- 
    e : psydac.linalg.block.BlockVector
        FE coefficients of a 1-form.

    particles : struphy.pic.particles.Particles6D
        Particles object.

    **params : dict
        Solver- and/or other parameters for this splitting step.
    """

    def __init__(self, e, particles, **params):

        from struphy.kinetic_background.maxwellians import Maxwellian6DUniform

        super().__init__(e, particles)

        # parameters
        params_default = {'alpha': 1.,
                          'kappa': 1.,
                          'f0': Maxwellian6DUniform(),
                          'model': 'linear_vlasov_maxwell',
                          'type': ('pcg', 'MassMatrixPreconditioner'),
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False}

        params = set_defaults(params, params_default)

        assert isinstance(params['f0'], Maxwellian6DUniform)
        assert params['model'] in (
            'linear_vlasov_maxwell', 'delta_f_vlasov_maxwell')

        self._alpha = params['alpha']
        self._kappa = params['kappa']
        self._f0 = params['f0']
        self._f0_params = np.array([self._f0.params['n'],
                                    self._f0.params['u1'],
                                    self._f0.params['u2'],
                                    self._f0.params['u3'],
                                    self._f0.params['vth1'],
                                    self._f0.params['vth2'],
                                    self._f0.params['vth3']])
        self._model = params['model']

        self._info = params['info']

        # Initialize Accumulator object
        if params['model'] == 'linear_vlasov_maxwell':
            self._accum = Accumulator(self.derham, self.domain, 'Hcurl', 'linear_vlasov_maxwell',
                                      add_vector=True, symmetry='symm')
        elif params['model'] == 'delta_f_vlasov_maxwell':
            self._accum = Accumulator(self.derham, self.domain, 'Hcurl', 'delta_f_vlasov_maxwell_scn',
                                      add_vector=True, symmetry='symm')
        else:
            raise NotImplementedError(f"Unknown model : {params['model']}")

        # Create buffers to store temporarily _e and its sum with old e
        self._e_temp = e.space.zeros()
        self._e_sum = e.space.zeros()

        # store old weights to compute difference
        self._old_weights = np.empty(particles.markers.shape[0], dtype=float)

        # ================================
        # ========= Schur Solver =========
        # ================================

        # Preconditioner
        if params['type'][1] == None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            pc = pc_class(self.mass_ops.M1)

        # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = self.mass_ops.M1
        _BC = - self._accum.operators[0].matrix / 4.

        # Instantiate Schur solver
        self._schur_solver = SchurSolver(_A, _BC,
                                         params['type'][0],
                                         pc=pc,
                                         tol=params['tol'],
                                         maxiter=params['maxiter'],
                                         verbose=params['verbose'])

        # Instantiate particle pusher
        if params['model'] == 'linear_vlasov_maxwell':
            self._pusher = Pusher(self.derham, self.domain,
                                  'push_weights_with_efield_lin_vm')
        elif params['model'] == 'delta_f_vlasov_maxwell':
            self._pusher = Pusher(self.derham, self.domain,
                                  'push_weights_with_efield_delta_f_vm')
        else:
            raise NotImplementedError(f"Unknown model : {params['model']}")

    def __call__(self, dt):
        # current e-field
        en = self.feec_vars[0]

        # evaluate f0 and accumulate
        f0_values = self._f0(self.particles[0].markers[:, 0],
                             self.particles[0].markers[:, 1],
                             self.particles[0].markers[:, 2],
                             self.particles[0].markers[:, 3],
                             self.particles[0].markers[:, 4],
                             self.particles[0].markers[:, 5])

        self._accum.accumulate(self.particles[0],
                               f0_values, float(self._f0_params[4]),
                               self._alpha, self._kappa)

        # Update Schur solver
        self._schur_solver.BC = - self._accum.operators[0].matrix / 4

        # new e-field (no tmps created here)
        en1, info = self._schur_solver(
            en, self._accum.vectors[0] / 2., dt,
            out=self._e_temp)

        # Store old weights
        self._old_weights[~self.particles[0].holes] = self.particles[0].markers[~self.particles[0].holes, 6]

        # Compute (e^{n+1} + e^n) / 2 (no tmps created here)
        _e = en.copy(out=self._e_sum)
        _e += en1
        _e *= 0.5

        # Update weights
        if self._model == 'linear_vlasov_maxwell':
            self._pusher(self.particles[0], dt,
                         self._e_sum.blocks[0]._data,
                         self._e_sum.blocks[1]._data,
                         self._e_sum.blocks[2]._data,
                         f0_values,
                         self._f0_params,
                         int(self.particles[0].n_mks),
                         self._kappa)
        elif self._model == 'delta_f_vlasov_maxwell':
            self._pusher(self.particles[0], dt,
                         self._e_sum.blocks[0]._data,
                         self._e_sum.blocks[1]._data,
                         self._e_sum.blocks[2]._data,
                         f0_values,
                         float(self._f0_params[4]),
                         self._kappa,
                         int(1)  # since we want to use the implicit substep
                         )

        # write new coeffs into self.variables
        max_de, = self.feec_vars_update(en1)

        # Print out max differences for weights and e-field
        if self._info:
            print('Status          for StepEfieldWeights:', info['success'])
            print('Iterations      for StepEfieldWeights:', info['niter'])
            print('Maxdiff    e1   for StepEfieldWeights:', max_de)
            max_diff = np.max(np.abs(self._old_weights[~self.particles[0].holes]
                                     - self.particles[0].markers[~self.particles[0].holes, 6]))
            print('Maxdiff weights for StepEfieldWeights:', max_diff)
            print()

    @classmethod
    def options(cls):
        dct = {}
        dct['solver'] = {'type': [('pcg', 'MassMatrixPreconditioner'),
                                  ('cg', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False}
        return dct


class EfieldWeightsDiscreteGradient(Propagator):
    r""" Solve the following system analytically

    .. math::

        \begin{align}
            \frac{\text{d}}{\text{d} t} w_p & = \frac{1}{N \, s_{0, p}} \frac{\kappa}{v_{\text{th}}^2} \left[ DF^{-T} (\mathbb{\Lambda}^1)^T \mathbf{e} \right]
            \cdot \mathbf{v}_p \left( \frac{f_0}{\ln(f_0)} - f_0 \right) \\[2mm]
            \frac{\text{d}}{\text{d} t} \mathbb{M}_1 \mathbf{e} & = - \alpha^2 \kappa \sum_p \mathbb{\Lambda}^1 \cdot \left( DF^{-1} \mathbf{v}_p \right)
            \frac{1}{N \, s_{0, p}} \left( \frac{f_0}{\ln(f_0)} - f_0 \right)
        \end{align}

    using the symplectic Euler method.

    Parameters
    ---------- 
    e : psydac.linalg.block.BlockVector
        FE coefficients of a 1-form.

    particles : struphy.pic.particles.Particles6D
        Particles object.

    **params : dict
        Solver- and/or other parameters for this splitting step.
    """

    def __init__(self, e, particles, **params):

        from struphy.kinetic_background.maxwellians import Maxwellian6DUniform

        super().__init__(e, particles)

        # parameters
        params_default = {'alpha': 1e2,
                          'kappa': 1.,
                          'f0': Maxwellian6DUniform(),
                          'type': 'pcg',
                          'pc': 'MassMatrixPreconditioner',
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False}

        params = set_defaults(params, params_default)

        self._params = params

        assert isinstance(params['f0'], Maxwellian6DUniform)

        self._alpha = params['alpha']
        self._kappa = params['kappa']
        self._f0 = params['f0']
        self._f0_params = np.array([self._f0.params['n'],
                                    self._f0.params['u1'],
                                    self._f0.params['u2'],
                                    self._f0.params['u3'],
                                    self._f0.params['vth1'],
                                    self._f0.params['vth2'],
                                    self._f0.params['vth3']])

        self._info = params['info']

        # Initialize Accumulator object
        self._accum = AccumulatorVector(
            self.derham, self.domain, 'Hcurl', 'delta_f_vlasov_maxwell')

        # Create buffers to temporarily store _e and its sum with old e
        self._m1_acc_vec = e.space.zeros()
        self._e_sum = e.space.zeros()

        # store old weights to compute difference
        self._old_weights = np.empty(particles.markers.shape[0], dtype=float)

        # Preconditioner
        if params['type'][1] == None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            pc = pc_class(self.mass_ops.M1)

        # solver
        self.solver = inverse(self.mass_ops.M1, 
                              params['type'][0], 
                              pc=pc,
                              x0=self.feec_vars[0],
                              tol=self._params['tol'],
                              maxiter=self._params['maxiter'],
                              verbose=self._params['verbose'])

        self._pusher = Pusher(self.derham, self.domain,
                              'push_weights_with_efield_delta_f_vm')

    def __call__(self, dt):
        # current e-field
        en = self.feec_vars[0]
        
        # evaluate f0 and accumulate
        f0_values = self._f0(self.particles[0].markers[:, 0],
                             self.particles[0].markers[:, 1],
                             self.particles[0].markers[:, 2],
                             self.particles[0].markers[:, 3],
                             self.particles[0].markers[:, 4],
                             self.particles[0].markers[:, 5])

        self._accum.accumulate(
            self.particles[0], f0_values, self._alpha, self._kappa, int(1))

        en1 = self.solver.solve(self._accum.vectors[0], out=self._m1_acc_vec)
        info = self.solver._info

        if self._info:
            # Store old weights
            self._old_weights[~self.particles[0].holes] = self.particles[0].markers[~self.particles[0].holes, 6]

        # Compute (e^{n+1} + e^n) / 2 = e^n + dt / 2 * accum_vec
        _e = en.copy(out=self._e_sum)
        en1 *= dt*0.5
        _e += en1

        # Update weights
        self._pusher(self.particles[0], dt,
                     _e.blocks[0]._data,
                     _e.blocks[1]._data,
                     _e.blocks[2]._data,
                     f0_values,
                     float(self._f0_params[4]),
                     self._kappa,
                     int(1)  # since we want to use the last substep
                     )

        # Update e-field and compute max difference
        en1 *= 2.
        max_de = np.max(np.abs(en1.toarray()))
        self.feec_vars[0] += en1

        # Print out max differences for weights and e-field
        if self._info:
            print('Status          for StepEfieldWeights:', info['success'])
            print('Iterations      for StepEfieldWeights:', info['niter'])
            print('Maxdiff    e1   for StepEfieldWeights:', max_de)
            max_diff = np.max(np.abs(self._old_weights[~self.particles[0].holes]
                                     - self.particles[0].markers[~self.particles[0].holes, 6]))
            print('Maxdiff weights for StepEfieldWeights:', max_diff)
            print()


class EfieldWeightsAnalytic(Propagator):
    r""" Solve the following system analytically

    .. math::

        \begin{align}
            \frac{\text{d}}{\text{d} t} w_p & = \frac{1}{N \, s_{0, p}} \frac{\kappa}{v_{\text{th}}^2} \left[ DF^{-T} (\mathbb{\Lambda}^1)^T \mathbf{e} \right]
            \cdot \mathbf{v}_p \left( \frac{f_0}{\ln(f_0)} - f_0 \right) \\[2mm]
            \frac{\text{d}}{\text{d} t} \mathbb{M}_1 \mathbf{e} & = - \alpha^2 \kappa \sum_p \mathbb{\Lambda}^1 \cdot \left( DF^{-1} \mathbf{v}_p \right)
            \frac{1}{N \, s_{0, p}} \left( \frac{f_0}{\ln(f_0)} - f_0 \right)
        \end{align}

    Parameters
    ---------- 
    e : psydac.linalg.block.BlockVector
        FE coefficients of a 1-form.

    particles : struphy.pic.particles.Particles6D
        Particles object.

    **params : dict
        Solver- and/or other parameters for this splitting step.
    """

    def __init__(self, e, particles, **params):

        from struphy.kinetic_background.maxwellians import Maxwellian6DUniform

        super().__init__(e, particles)

        # parameters
        params_default = {'alpha': 1e2,
                          'kappa': 1.,
                          'f0': Maxwellian6DUniform(),
                          'type': ('pcg', 'MassMatrixPreconditioner'),
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False}

        params = set_defaults(params, params_default)

        self._params = params

        assert isinstance(params['f0'], Maxwellian6DUniform)

        self._alpha = params['alpha']
        self._kappa = params['kappa']
        self._f0 = params['f0']
        self._f0_params = np.array([self._f0.params['n'],
                                    self._f0.params['u1'],
                                    self._f0.params['u2'],
                                    self._f0.params['u3'],
                                    self._f0.params['vth1'],
                                    self._f0.params['vth2'],
                                    self._f0.params['vth3']])

        self._info = params['info']

        # Initialize Accumulator object
        self._accum = AccumulatorVector(
            self.derham, self.domain, 'Hcurl', 'delta_f_vlasov_maxwell')

        # Create buffers to temporarily store _e and its sum with old e
        self._m1_acc_vec = e.space.zeros()
        self._e_dt2 = e.space.zeros()

        # store old weights to compute difference
        self._old_weights = np.empty(particles.markers.shape[0], dtype=float)

        # Preconditioner
        if params['type'][1] == None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            self._pc = pc_class(self.mass_ops.M1)

        # solver
        self.solver = inverse(self.mass_ops.M1, 
                              params['type'][0], 
                              pc=self._pc,
                              x0=self.feec_vars[0],
                              tol=self._params['tol'],
                              maxiter=self._params['maxiter'],
                              verbose=self._params['verbose'])

        self._pusher = Pusher(self.derham, self.domain,
                              'push_weights_with_efield_delta_f_vm')

    def __call__(self, dt):
        # evaluate f0 and accumulate
        f0_values = self._f0(self.particles[0].markers[:, 0],
                             self.particles[0].markers[:, 1],
                             self.particles[0].markers[:, 2],
                             self.particles[0].markers[:, 3],
                             self.particles[0].markers[:, 4],
                             self.particles[0].markers[:, 5])

        self._accum.accumulate(
            self.particles[0], f0_values, self._alpha, self._kappa, int(0))

        en1 = self.solver.solve(self._accum.vectors[0], out=self._m1_acc_vec)
        info = self.solver._info

        # Store old weights
        self._old_weights[~self.particles[0].holes] = self.particles[0].markers[~self.particles[0].holes, 6]

        # Compute vector for particle pushing
        self._e_dt2 *= 0.
        self._e_dt2 -= en1
        self._e_dt2 *= dt / 2
        self._e_dt2 += self.feec_vars[0]

        # Update weights
        self._pusher(self.particles[0], dt,
                     self._e_dt2.blocks[0]._data,
                     self._e_dt2.blocks[1]._data,
                     self._e_dt2.blocks[2]._data,
                     f0_values,
                     float(self._f0_params[4]),
                     self._kappa,
                     int(0)  # since we want to use the explicit substep
                     )

        # Update e-field and compute max difference
        en1 *= dt
        max_de = np.max(np.abs(en1.toarray()))
        self.feec_vars[0] -= en1

        # Print out max differences for weights and e-field
        if self._info:
            print('Status          for StepEfieldWeights:', info['success'])
            print('Iterations      for StepEfieldWeights:', info['niter'])
            print('Maxdiff    e1   for StepEfieldWeights:', max_de)
            max_diff = np.max(np.abs(self._old_weights[~self.particles[0].holes]
                                     - self.particles[0].markers[~self.particles[0].holes, 6]))
            print('Maxdiff weights for StepEfieldWeights:', max_diff)
            print()

    @classmethod
    def options(cls):
        dct = {}
        dct['solver'] = {'type': [('pcg', 'MassMatrixPreconditioner'),
                                  ('cg', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False}
        return dct


class PressureCoupling6D(Propagator):
    r'''Crank-Nicolson step for pressure coupling term in MHD equations and velocity update with the force term :math:`\nabla \mathbf U \cdot \mathbf v`.

    .. math::

        \begin{bmatrix} u^{n+1} - u^n \\ V^{n+1} - V^n \end{bmatrix} 
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^n)^{-1} V^\top (\bar {\mathcal X})^\top \mathbb G^\top (\bar {\mathbf \Lambda}^1)^\top \bar {DF}^{-1} \\ - {DF}^{-\top} \bar {\mathbf \Lambda}^1 \mathbb G \bar {\mathcal X} V (\mathbb M^n)^{-1} & 0 \end{bmatrix} 
        \begin{bmatrix} {\mathbb M^n}(u^{n+1} + u^n) \\ \bar W (V^{n+1} + V^{n} \end{bmatrix} ,

    based on the :ref:`Schur complement <schur_solver>`.

    Parameters
    ---------- 
        u : psydac.linalg.block.BlockVector
            FE coefficients of a discrete 1-form.

        u_space : dict
                  params['fields']['mhd_u_space']

        coupling : dict
                   params['coupling']['scheme']

        particles : struphy.pic.particles.Particles6D

        domain : struphy.geometry.base.Domain
                 Infos regarding mapping.

        mass_ops : struphy.feec.mass.WeightedMassOperators
                   Weighted mass matrices from struphy.feec.mass.

        mhd_ops : struphy.feec.basis_projection_ops.MHDOperators
                  Linear MHD operators from struphy.feec.basis_projection_ops.

        coupling_solver: dict
                         Solver parameters for this splitting step.
    '''

    def __init__(self, particles, u, **params):

        super().__init__(particles, u)

        # parameters
        params_default = {'u_space': 'Hdiv',
                          'use_perp_model': True,
                          'f0': Maxwellian6DUniform(),
                          'type': ('pcg', 'MassMatrixPreconditioner'),
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False,
                          'nuh': 5.,
                          'Ab': 1,
                          'Ah': 1,
                          'Zh': 1,
                          'kappa': 1.}

        params = set_defaults(params, params_default)

        self._G = self.derham.grad
        self._GT = self.derham.grad.transpose()

        self._info = params['info']
        self._type = params['type'][0]
        self._tol = params['tol']
        self._maxiter = params['maxiter']
        self._verbose = params['verbose']

        self._rank = self.derham.comm.Get_rank()

        assert params['u_space'] in {'Hcurl', 'Hdiv', 'H1vec'}

        if params['u_space'] == 'Hcurl':
            id_Mn = 'M1n'
            id_X = 'X1'
        elif params['u_space'] == 'Hdiv':
            id_Mn = 'M2n'
            id_X = 'X2'
        elif params['u_space'] == 'H1vec':
            id_Mn = 'Mvn'
            id_X = 'Xv'

        if params['u_space'] == 'H1vec':
            self._space_key_int = 0
        else:
            self._space_key_int = int(
                self.derham.space_to_form[params['u_space']])

        # Preconditioner
        if params['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            pc = pc_class(getattr(self.mass_ops, id_Mn))

        # Call the accumulation and Pusher class
        accum_ker = 'pc_lin_mhd_6d'
        pusher_ker = 'push_pc_GXu'
        if not params['use_perp_model']:
            accum_ker += '_full'
            pusher_ker += '_full'

        self._info = params['info']

        # ============= TODO ========
        self._coupling_mat = 1.
        self._coupling_vec = 1.
        self._scale_push = 1.
        # ===========================

        self._ACC = Accumulator(self.derham, self.domain, 'Hcurl',
                                accum_ker, add_vector=True,
                                symmetry='pressure')
        self._pusher = Pusher(self.derham, self.domain, pusher_ker)

        # Define operators
        self._A = getattr(self.mass_ops, id_Mn)
        self._X = getattr(self.basis_ops, id_X)
        self._XT = self._X.transpose()

        # Instantiate schur solver with dummy BC
        self._schur_solver = SchurSolver(self._A, self._XT @ self._X,
                                         self._type,
                                         pc=pc,
                                         tol=self._tol,
                                         maxiter=self._maxiter,
                                         verbose=self._verbose)

        self.u_temp = u.space.zeros()
        self.u_temp2 = u.space.zeros()
        self._tmp = self._X.codomain.zeros()
        self._tmp_g1 = self._G.codomain.zeros()
        self._tmp_g2 = self._G.codomain.zeros()
        self._tmp_g3 = self._G.codomain.zeros()
        self._BV = u.space.zeros()

    def __call__(self, dt):

        # current u
        un = self.feec_vars[0]
        un.update_ghost_regions()

        # acuumulate MAT and VEC
        self._ACC.accumulate(
            self.particles[0], self._coupling_mat, self._coupling_vec)

        MAT = [[self._ACC.operators[0].matrix, self._ACC.operators[1].matrix, self._ACC.operators[2].matrix],
               [self._ACC.operators[1].matrix, self._ACC.operators[3].matrix,
                   self._ACC.operators[4].matrix],
               [self._ACC.operators[2].matrix, self._ACC.operators[4].matrix, self._ACC.operators[5].matrix]]
        VEC = [self._ACC.vectors[0], self._ACC.vectors[1], self._ACC.vectors[2]]

        GT_VEC = BlockVector(self.derham.Vh['v'],
                             blocks=[self._GT.dot(VEC[0]),
                                     self._GT.dot(VEC[1]),
                                     self._GT.dot(VEC[2])])

        # define BC and B dot V of the Schur block matrix [[A, B], [C, I]]
        self._schur_solver.BC = -1/4 * \
            self._XT @ self.GT_MAT_G(self.derham, MAT) @ self._X

        self._BV = self._XT.dot(GT_VEC)*(-1/2)

        # update u (no tmps created here)
        un1, info = self._schur_solver(un, self._BV, dt, out=self.u_temp)

        _u = un.copy(out=self.u_temp2)
        _u += un1

        # calculate GXu
        Xu = self._X.dot(_u, out=self._tmp)

        GXu_1 = self._G.dot(Xu[0], out=self._tmp_g1)
        GXu_2 = self._G.dot(Xu[1], out=self._tmp_g2)
        GXu_3 = self._G.dot(Xu[2], out=self._tmp_g3)

        GXu_1.update_ghost_regions()
        GXu_2.update_ghost_regions()
        GXu_3.update_ghost_regions()

        # push particles
        self._pusher(self.particles[0], dt,
                     GXu_1[0]._data, GXu_1[1]._data, GXu_1[2]._data,
                     GXu_2[0]._data, GXu_2[1]._data, GXu_2[2]._data,
                     GXu_3[0]._data, GXu_3[1]._data, GXu_3[2]._data)

        # write new coeffs into Propagator.variables
        max_du, = self.feec_vars_update(un1)

        if self._info and self._rank == 0:
            print('Status     for StepPressurecoupling:', info['success'])
            print('Iterations for StepPressurecoupling:', info['niter'])
            print('Maxdiff u1 for StepPressurecoupling:', max_du)
            print()

    @classmethod
    def options(cls):
        dct = {}
        dct['u_space'] = ['Hcurl', 'Hdiv', 'H1vec']
        dct['solver'] = {'type': [('pcg', 'MassMatrixPreconditioner'),
                                  ('cg', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False}
        return dct

    class GT_MAT_G(LinOpWithTransp):
        r'''
        Class for defining LinearOperator corresponding to :math:`G^\top (\text{MAT}) G \in \mathbb{R}^{3N^0 \times 3N^0}` 
        where :math:`\text{MAT} = V^\top (\bar {\mathbf \Lambda}^1)^\top \bar{DF}^{-1} \bar{W} \bar{DF}^{-\top} \bar{\mathbf \Lambda}^1 V \in \mathbb{R}^{3N^1 \times 3N^1}`.

        Parameters
        ----------
            derham : struphy.feec.psydac_derham.Derham
                Discrete de Rham sequence on the logical unit cube.

            MAT : List of StencilMatrices
                List with six of accumulated pressure terms
        '''

        def __init__(self, derham, MAT, transposed=False):

            self._derham = derham
            self._G = derham.grad
            self._GT = derham.grad.transpose()

            self._domain = derham.Vh['v']
            self._codomain = derham.Vh['v']
            self._MAT = MAT

            self._vector = BlockVector(derham.Vh['v'])
            self._temp = BlockVector(derham.Vh['1'])

        @property
        def domain(self):
            return self._domain

        @property
        def codomain(self):
            return self._codomain

        @property
        def dtype(self):
            return self._derham.Vh['v'].dtype

        @property
        def tosparse(self):
            raise NotImplementedError()

        @property
        def toarray(self):
            raise NotImplementedError()

        @property
        def transposed(self):
            return self._transposed

        def transpose(self):
            return self.GT_MAT_G(self._derham, self._MAT, True)

        def dot(self, v, out=None):
            '''dot product between GT_MAT_G and v.

            Parameters
            ----------
                v : StencilVector or BlockVector
                    Input FE coefficients from V.vector_space.

            Returns
            -------
                A StencilVector or BlockVector from W.vector_space.'''

            assert v.space == self.domain

            v.update_ghost_regions()

            for i in range(3):
                for j in range(3):
                    self._temp += self._MAT[i][j].dot(self._G.dot(v[j]))

                self._vector[i] = self._GT.dot(self._temp)
                self._temp *= 0.

            self._vector.update_ghost_regions()

            if out is not None:
                self._vector.copy(out=out)

            assert self._vector.space == self.codomain

            return self._vector


class CurrentCoupling6DCurrent(Propagator):
    """
    Parameters
    ----------
    particles : struphy.pic.particles.Particles6D
        Holdes the markers to push.

    u : psydac.linalg.block.BlockVector
        FE coefficients of MHD velocity.

    **params : dict
        Solver- and/or other parameters for this splitting step.
    """

    def __init__(self, particles, u, **params):

        super().__init__(particles, u)

        # parameters
        params_default = {'u_space': 'Hdiv',
                          'b_eq': None,
                          'b_tilde': None,
                          'type': ('pcg', 'MassMatrixPreconditioner'),
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False,
                          'Ab': 1,
                          'Ah': 1,
                          'kappa': 1.}

        params = set_defaults(params, params_default)

        # assert parameters and expose some quantities to self
        assert params['u_space'] in {'Hcurl', 'Hdiv', 'H1vec'}
        if params['u_space'] == 'H1vec':
            self._space_key_int = 0
        else:
            self._space_key_int = int(
                self.derham.space_to_form[params['u_space']])

        assert isinstance(params['b_eq'], (BlockVector, PolarVector))

        if params['b_tilde'] is not None:
            assert isinstance(params['b_tilde'], (BlockVector, PolarVector))

        self._b_eq = params['b_eq']
        self._b_tilde = params['b_tilde']

        if self.particles[0].control_variate:
            assert isinstance(self.f_backgr, Maxwellian)

            # evaluate and save nh0 (0-form) * uh0 (2-form if H1vec or vector if Hdiv) at quadrature points for control variate
            quad_pts = [quad_grid[nquad].points.flatten()
                        for quad_grid, nquad in zip(self.derham.Vh_fem['0']._quad_grids, self.derham.Vh_fem['0'].nquads)]

            uh0_cart = self.f_backgr.u

            if params['u_space'] == 'H1vec':
                self._nuh0_at_quad = self.domain.pull(
                    uh0_cart, *quad_pts, kind='2_form', squeeze_out=False, coordinates='logical')
            else:
                self._nuh0_at_quad = self.domain.pull(
                    uh0_cart, *quad_pts, kind='vector', squeeze_out=False, coordinates='logical')

            self._nuh0_at_quad[0] *= self.domain.pull(
                [self.f_backgr.n], *quad_pts, kind='0_form', squeeze_out=False, coordinates='logical')
            self._nuh0_at_quad[1] *= self.domain.pull(
                [self.f_backgr.n], *quad_pts, kind='0_form', squeeze_out=False, coordinates='logical')
            self._nuh0_at_quad[2] *= self.domain.pull(
                [self.f_backgr.n], *quad_pts, kind='0_form', squeeze_out=False, coordinates='logical')

            # memory allocation for magnetic field at quadrature points
            self._b_quad1 = np.zeros_like(self._nuh0_at_quad[0])
            self._b_quad2 = np.zeros_like(self._nuh0_at_quad[0])
            self._b_quad3 = np.zeros_like(self._nuh0_at_quad[0])

            # memory allocation for (self._b_quad x self._nuh0_at_quad) * self._coupling_vec
            self._vec1 = np.zeros_like(self._nuh0_at_quad[0])
            self._vec2 = np.zeros_like(self._nuh0_at_quad[0])
            self._vec3 = np.zeros_like(self._nuh0_at_quad[0])

        self._info = params['info']
        self._rank = self.derham.comm.Get_rank()

        self._coupling_mat = params['Ah'] / params['Ab'] * params['kappa']**2
        self._coupling_vec = params['Ah'] / params['Ab'] * params['kappa']
        self._scale_push = 1*params['kappa']

        # load accumulator
        self._accumulator = Accumulator(
            self.derham, self.domain, params['u_space'], 'cc_lin_mhd_6d_2', add_vector=True, symmetry='symm')

        # load particle pusher
        self._pusher = Pusher(self.derham, self.domain,
                              'push_bxu_' + params['u_space'])

        # FEM spaces and basis extraction operators for u and b
        u_id = self.derham.space_to_form[params['u_space']]
        self._EuT = self.derham.extraction_ops[u_id].transpose()
        self._EbT = self.derham.extraction_ops['2'].transpose()

        # define system [[A B], [C I]] [u_new, v_new] = [[A -B], [-C I]] [u_old, v_old] (without time step size dt)
        _A = getattr(self.mass_ops, 'M' + u_id + 'n')

        # preconditioner
        if params['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            pc = pc_class(_A)

        _BC = -1/4 * self._accumulator.operators[0]

        self._schur_solver = SchurSolver(_A, _BC,
                                         params['type'][0],
                                         pc=pc,
                                         tol=params['tol'],
                                         maxiter=params['maxiter'],
                                         verbose=params['verbose'])

        # temporary vectors to avoid memory allocation
        self._b_full1 = self._b_eq.space.zeros()
        self._b_full2 = self._EbT.codomain.zeros()

        self._u_new = u.space.zeros()

        self._u_avg1 = u.space.zeros()
        self._u_avg2 = self._EuT.codomain.zeros()

    def __call__(self, dt):

        # pointer to old coefficients
        un = self.feec_vars[0]

        # sum up total magnetic field b_full1 = b_eq + b_tilde (in-place)
        self._b_eq.copy(out=self._b_full1)

        if self._b_tilde is not None:
            self._b_full1 += self._b_tilde

        # extract coefficients to tensor product space (in-place)
        self._EbT.dot(self._b_full1, out=self._b_full2)

        # update ghost regions because of non-local access in accumulation kernel!
        self._b_full2.update_ghost_regions()

        # perform accumulation (either with or without control variate)
        if self.particles[0].control_variate:

            # evaluate magnetic field at quadrature points (in-place)
            WeightedMassOperator.eval_quad(self.derham.Vh_fem['2'], self._b_full2,
                                           out=[self._b_quad1, self._b_quad2, self._b_quad3])

            self._vec1[:, :, :] = self._coupling_vec * \
                (self._b_quad2 *
                 self._nuh0_at_quad[2] - self._b_quad3*self._nuh0_at_quad[1])
            self._vec2[:, :, :] = self._coupling_vec * \
                (self._b_quad3 *
                 self._nuh0_at_quad[0] - self._b_quad1*self._nuh0_at_quad[2])
            self._vec3[:, :, :] = self._coupling_vec * \
                (self._b_quad1 *
                 self._nuh0_at_quad[1] - self._b_quad2*self._nuh0_at_quad[0])

            self._accumulator.accumulate(self.particles[0],
                                         self._b_full2[0]._data, self._b_full2[1]._data, self._b_full2[2]._data,
                                         self._space_key_int, self._coupling_mat, self._coupling_vec,
                                         control_vec=[self._vec1, self._vec2, self._vec3])
        else:
            self._accumulator.accumulate(self.particles[0],
                                         self._b_full2[0]._data, self._b_full2[1]._data, self._b_full2[2]._data,
                                         self._space_key_int, self._coupling_mat, self._coupling_vec)

        # solve linear system for updated u coefficients (in-place)
        un1, info = self._schur_solver(
            un, -self._accumulator.vectors[0]/2, dt, out=self._u_new)

        # call pusher kernel with average field (u_new + u_old)/2 and update ghost regions because of non-local access in kernel
        _u = un.copy(out=self._u_avg1)
        _u += un1
        _u *= 0.5

        _Eu = self._EuT.dot(_u, out=self._u_avg2)

        _Eu.update_ghost_regions()

        # push particles
        self._pusher(self.particles[0], self._scale_push*dt,
                     self._b_full2[0]._data, self._b_full2[1]._data, self._b_full2[2]._data,
                     _Eu[0]._data, _Eu[1]._data, _Eu[2]._data)

        # write new coeffs into Propagator.variables
        max_du = self.feec_vars_update(un1)

        # update weights in case of control variate
        if self.particles[0].control_variate:
            self.particles[0].update_weights()

        if self._info and self._rank == 0:
            print('Status     for CurrentCoupling6DCurrent:', info['success'])
            print('Iterations for CurrentCoupling6DCurrent:', info['niter'])
            print('Maxdiff up for CurrentCoupling6DCurrent:', max_du)
            print()

    @classmethod
    def options(cls):
        dct = {}
        dct['u_space'] = ['Hcurl', 'Hdiv', 'H1vec']
        dct['solver'] = {'type': [('pcg', 'MassMatrixPreconditioner'),
                                  ('cg', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False}
        return dct


class CurrentCoupling5DCurlb(Propagator):
    r'''Crank-Nicolson step for the current coupling part (:math:`v_\parallel \nabla \times \mathbf b_0`) in `LinearMHDDriftkineticCC <https://struphy.pages.mpcdf.de/struphy/sections/models.html#struphy.models.hybrid.LinearMHDDriftkineticCC>`_ model,

    Equation:

    .. math::

        \left\{ 
            \begin{aligned} 
                n_0 &\frac{\partial \tilde{\mathbf U}}{\partial t} = - \frac{A_h}{A_b} \iint f_{\textnormal{h}} \frac{1}{B^*_\parallel} v_\parallel^2 \nabla \times \mathbf b_0 \textnormal{d} v_\parallel \textnormal{d} \mu \times \mathbf B \,,
                \\
                &\frac{\partial v_\parallel}{\partial t} = - \frac{1}{B^*_\parallel} v_\parallel (\nabla \times \mathbf b_0) \cdot (\mathbf B \times \tilde{\mathbf U}) \,.
            \end{aligned}
        \right.

    FE coefficients and marker update (:math:`\alpha = 2`):

    .. math::

        \begin{bmatrix} 
            \mathbf u^{n+1} - \mathbf u^n \\ V_\parallel^{n+1} - V_\parallel^n
        \end{bmatrix} 
        = \frac{\Delta t}{2} \,.
        \begin{bmatrix} 
            0 & (\mathbb M^\rho_\alpha)^{-1} {\mathbb{P}^2}^\top \mathbb B^{*,-1}_\parallel \mathbb{V}_\parallel \sqrt{\mathbb g}^{-1} \sqrt{\mathbb g}^{-1} \mathbb{B}^\times \mathbb b_0^{\nabla \times}
            \\  
            - {\mathbb b_0^{\nabla \times}}^\top \mathbb{B}^\times \sqrt{\mathbb g}^{-1} \sqrt{\mathbb g}^{-1} \mathbb{V}_\parallel \mathbb B^{*,-1}_\parallel \mathbb{P}^2 (\mathbb M^\rho_\alpha)^{-1} & 0 
        \end{bmatrix} 
        \begin{bmatrix}
            \mathbb M^\rho_\alpha (\mathbf u^{n+1} + \mathbf u^n)
            \\
            \mathbb W (\bar{V}_\parallel^{n+1} + \bar{V}_\parallel^n)
        \end{bmatrix} \,,

    where 
    :math:`\mathbb M^\rho_\alpha` is a :ref:`weighted_mass` being weighted with :math:`\rho_0`, the MHD equilibirum density. 
    :math:`\alpha \in \{1, 2, v\}` denotes the :math:`\alpha`-form space where the operators correspond to.
    Moreover, :math:`\mathbb B^\times, \, \mathbb b_0^{\nabla \times}, \, \mathbb P^2` and notations with over-bar are the block matrices which are diagonally stacked collocation vectors.    Note that following matrices are not assembled but only for representing the accumulation and pushing of particles compactly:

    .. math::

        \begin{alignat}{2}
            &\mathbb{V}_\parallel := \mathbb{I}_3 \otimes \text{diag}(V_\parallel) && 
            \bar{V}_\parallel := (V_\parallel, V_\parallel, V_\parallel) \qquad \qquad V_\parallel := (v_{\parallel,1}, \, \dots, v_{\parallel,N_p})^\top
            \\
            &\mathbb W_\parallel := \mathbb{I}_3 \otimes \text{diag}(W) && 
            W := (\omega_1, \, \dots, \omega_{N_p})^\top 
            \\
            &\mathbb B^{*,-1}_\parallel := \mathbb{I}_3 \otimes \mathbb{B}^{*, -1}_\parallel && 
            \bar B^{*,-1}_\parallel := \text{diag}(B^{*, -1}_{\parallel}(\boldsymbol{\eta}_1, v_{\parallel,1}), \, \dots, B^{*, -1}_{\parallel}(\boldsymbol{\eta}_{N_p}, v_{\parallel,N_p}))
            \\
            &\sqrt{\mathbb g}^{-1} := \mathbb{I}_3 \otimes  \bar{\sqrt{g}}^{-1} && 
            \bar{\sqrt{g}}^{-1} := \text{diag}(\sqrt{g(\boldsymbol{\eta}_{1})}^{-1}, \, \dots, \sqrt{g(\boldsymbol{\eta}_{N_p})}^{-1})
            \\
            &\mathbb{P}^n := \text{diag}(\mathbb{P}^n_1, \mathbb{P}^n_2, \mathbb{P}^n_3) && 
            \mathbb{P}^n_\mu := (\Lambda^n_{\mu,i}(\boldsymbol{\eta}_k))_{0\leq i \leq N^n_\mu, \,  1\leq k \leq N_p, \, n \in \{v,1,2\}, \, \mu \in \{ 1,2,3\}}
            \\
            &\mathbb{b}^{\nabla \times}_0 := (\mathbb{b}^{\nabla \times}_{0,1}, \mathbb{b}^{\nabla \times}_{0,2}, \mathbb{b}^{\nabla \times}_{0,3}) &&
            \mathbb{b}^{\nabla\times}_{0,\mu} := ((\widehat \nabla \times \widehat{\mathbf b}^1_{0})_\mu(\boldsymbol{\eta}_k))_{1 \leq k \leq N_p, \mu \in \{1,2,3\}}
            \\
            &\mathbb{B}^\times := 
            \begin{pmatrix}
                0 & - \mathbb{B}^2_3 & \mathbb{B}^2_2
                \\
                \mathbb{B}^2_3 & 0 & -\mathbb{B}^2_1
                \\
                - \mathbb{B}^2_2 & \mathbb{B}^2_1 & 0
            \end{pmatrix} \qquad && 
            \mathbb{B}^2_\mu = \mathbf b^\top \mathbb{P}^2_\mu + (\widehat{\mathbf B}^2_{0,\mu}(\boldsymbol{\eta}_k))_{1 \leq k \leq N_p, \mu \in \{1,2,3\}}
        \end{alignat}

    The solution of the above system is based on the :ref:`Schur complement <schur_solver>`.

    Parameters
    ---------- 
    particles : struphy.pic.particles.Particles6D
        Particles object.

    u : psydac.linalg.block.BlockVector
        FE coefficients of MHD velocity.

    **params : dict
        Solver- and/or other parameters for this splitting step.
    '''

    def __init__(self, particles, u, **params):

        super().__init__(particles, u)

        # parameters
        params_default = {'u_space': 'Hdiv',
                          'b': None,
                          'b_eq': None,
                          'unit_b1': None,
                          'curl_unit_b2': None,
                          'f0': Maxwellian5DUniform(),
                          'type': ('pcg', 'MassMatrixPreconditioner'),
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False,
                          'Ab': 1,
                          'Ah': 1,
                          'epsilon': 1.}

        params = set_defaults(params, params_default)

        assert params['u_space'] in {'Hcurl', 'Hdiv', 'H1vec'}

        if params['u_space'] == 'H1vec':
            self._space_key_int = 0
        else:
            self._space_key_int = int(
                self.derham.space_to_form[params['u_space']])

        self._epsilon = params['epsilon']
        self._f0 = params['f0']
        self._b = params['b']
        self._b_eq = params['b_eq']
        self._unit_b1 = params['unit_b1']
        self._curl_norm_b = params['curl_unit_b2']

        self._info = params['info']
        self._rank = self.derham.comm.Get_rank()

        self._coupling_mat = params['Ah'] / params['Ab']
        self._coupling_vec = params['Ah'] / params['Ab']
        self._scale_push = 1

        u_id = self.derham.space_to_form[params['u_space']]
        self._EuT = self.derham.extraction_ops[u_id].transpose()
        self._E2T = self.derham.extraction_ops['2'].transpose()
        self._E1T = self.derham.extraction_ops['1'].transpose()

        self._unit_b1 = self._E1T.dot(self._unit_b1)
        self._curl_norm_b = self._E2T.dot(self._curl_norm_b)
        self._curl_norm_b.update_ghost_regions()

        # define system [[A B], [C I]] [u_new, v_new] = [[A -B], [-C I]] [u_old, v_old] (without time step size dt)
        _A = getattr(self.mass_ops, 'M' + u_id + 'n')

        # preconditioner
        if params['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            pc = pc_class(_A)

        # Call the accumulation and Pusher class
        self._ACC = Accumulator(self.derham, self.domain, params['u_space'],
                                'cc_lin_mhd_5d_J1', add_vector=True, symmetry='symm')
        self._pusher = Pusher(self.derham, self.domain,
                              'push_gc_cc_J1_' + params['u_space'])

        # define BC and B dot V of the Schur block matrix [[A, B], [C, I]]
        _BC = -1/4 * self._ACC.operators[0]

        # call SchurSolver class
        self._schur_solver = SchurSolver(_A, _BC, 
                                         params['type'][0],
                                         pc=pc,
                                         tol=params['tol'], 
                                         maxiter=params['maxiter'],
                                         verbose=params['verbose'])

        # temporary vectors to avoid memory allocation
        self._b_full1 = self._b_eq.space.zeros()
        self._b_full2 = self._E2T.codomain.zeros()
        self._u_new = u.space.zeros()
        self._u_avg1 = u.space.zeros()
        self._u_avg2 = self._EuT.codomain.zeros()

    def __call__(self, dt):

        un = self.feec_vars[0]

        # sum up total magnetic field b_full1 = b_eq + b_tilde (in-place)
        b_full = self._b_eq.copy(out=self._b_full1)

        if self._b is not None:
            self._b_full1 += self._b

        # extract coefficients to tensor product space (in-place)
        Eb_full = self._E2T.dot(b_full, out=self._b_full2)

        # update ghost regions because of non-local access in accumulation kernel!
        Eb_full.update_ghost_regions()

        # acuumulate MAT and VEC
        self._ACC.accumulate(self.particles[0], self._epsilon,
                             Eb_full[0]._data, Eb_full[1]._data, Eb_full[2]._data,
                             self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
                             self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                             self._space_key_int, self._coupling_mat, self._coupling_vec)

        # update u coefficients
        un1, info = self._schur_solver(
            un, -self._ACC.vectors[0]/2, dt, out=self._u_new)

        # call pusher kernel with average field (u_new + u_old)/2 and update ghost regions because of non-local access in kernel
        _u = un.copy(out=self._u_avg1)
        _u += un1
        _u *= 0.5

        _Eu = self._EuT.dot(_u, out=self._u_avg2)

        _Eu.update_ghost_regions()

        self._pusher(self.particles[0], self._scale_push*dt,
                     self._epsilon,
                     Eb_full[0]._data, Eb_full[1]._data, Eb_full[2]._data,
                     self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
                     self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                     _Eu[0]._data, _Eu[1]._data, _Eu[2]._data)

        # write new coeffs into Propagator.variables
        max_du, = self.feec_vars_update(un1)

        if self._info and self._rank == 0:
            print('Status     for CurrentCoupling5DCurlb:', info['success'])
            print('Iterations for CurrentCoupling5DCurlb:', info['niter'])
            print('Maxdiff up for CurrentCoupling5DCurlb:', max_du)
            print()

    @classmethod
    def options(cls):
        dct = {}
        dct['u_space'] = ['Hcurl', 'Hdiv', 'H1vec']
        dct['solver'] = {'type': [('pcg', 'MassMatrixPreconditioner'),
                                  ('cg', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False}
        return dct


class CurrentCoupling5DGradBxB(Propagator):
    r'''Crank-Nicolson step for the current coupling part (:math:`\mu \nabla B_\parallel \times \mathbf b_0`) in `LinearMHDDriftkineticCC <https://struphy.pages.mpcdf.de/struphy/sections/models.html#struphy.models.hybrid.LinearMHDDriftkineticCC>`_ model,

    Equation:

    .. math::

        \left\{ 
            \begin{aligned} 
                n_0 &\frac{\partial \tilde{\mathbf U}}{\partial t} = \frac{A_h}{A_b} \iint f_{\textnormal{h}} \frac{1}{B^*_\parallel} \mathbf b_0 \times (\mu \nabla B_\parallel) \textnormal{d} v_\parallel \textnormal{d} \mu \times \mathbf B \,,
                \\
                &\frac{\partial \mathbf X}{\partial t} = \frac{1}{B^*_\parallel} \mathbf b_0 \times \tilde{\mathbf U} \times \mathbf B \,.
            \end{aligned}
        \right.

    FE coefficients and marker update (:math:`\alpha = 2`):

    .. math::

        \begin{bmatrix} 
            \mathbf u^{n+1} - \mathbf u^n \\ \mathbf H^{n+1} - \mathbf H^n
        \end{bmatrix} 
        = \frac{\Delta t}{2} \,.
        \begin{bmatrix} 
            0 & (\mathbb M^\rho_\alpha)^{-1} {\mathbb{P}^2}^\top \mathbb B^{*,-1}_\parallel \sqrt{\mathbb g}^{-1}\mathbb{B}^\times \bar G^{-1} \mathbb b_0^\times \bar G^{-1}
            \\  
            - \bar G^{-1} \mathbb b_0^\times \bar G^{-1} \mathbb{B}^\times \sqrt{\mathbb g}^{-1} \mathbb B^{*,-1}_\parallel \mathbb{P}^2 (\mathbb M^\rho_\alpha)^{-1} & 0 
        \end{bmatrix} 
        \begin{bmatrix}
            \mathbb M^\rho_\alpha (\mathbf u^{n+1} + \mathbf u^n)
            \\
            \mathbb{W} \bar \mu \mathbb P^2 \mathbb G \mathcal{P}_b (\mathbf b^{n+1} + \mathbf b^n)
        \end{bmatrix} \,,

    where 
    :math:`\mathbb M^\rho_\alpha` is a :ref:`weighted_mass` being weighted with :math:`\rho_0`, the MHD equilibirum density. 
    :math:`\alpha \in \{1, 2, v\}` denotes the :math:`\alpha`-form space where the operators correspond to.
    Moreover, :math:`\mathbb B^\times, \, \mathbb b_0^{\times}, \, \mathbb P^2` and notations with over-bar are the block matrices which are diagonally stacked collocation vectors.    Note that following matrices are not assembled but only for representing the accumulation and pushing of particles compactly:

    .. math::

        \begin{alignat}{2}
            \\
            &\mathbf H := (\eta_{1,1}, \dots, \eta_{N_p,1}, \eta_{1,2}, \dots, \eta_{N_p,2}, \eta_{1,3}, \dots, \eta_{N_p,3})^\top
            \\
            &\bar \mu := \mathbb{I}_3 \otimes \text{diag}((\mu_1, \, \dots, \mu_{N_p})^\top )
            \\
            &\mathbb W_\parallel := \mathbb{I}_3 \otimes \text{diag}(W) && 
            W := (\omega_1, \, \dots, \omega_{N_p})^\top 
            \\
            &\mathbb B^{*,-1}_\parallel := \mathbb{I}_3 \otimes \mathbb{B}^{*, -1}_\parallel && 
            \bar B^{*,-1}_\parallel := \text{diag}(B^{*, -1}_{\parallel}(\boldsymbol{\eta}_1, v_{\parallel,1}), \, \dots, B^{*, -1}_{\parallel}(\boldsymbol{\eta}_{N_p}, v_{\parallel,N_p}))
            \\
            &\bar{G}^{-1} := \text{diag}(G^{-1}(\boldsymbol{\eta}_{1}), \dots, G^{-1}(\boldsymbol{\eta}_{N_p}))
            \\
            &\sqrt{\mathbb g}^{-1} := \mathbb{I}_3 \otimes  \bar{\sqrt{g}}^{-1} && 
            \bar{\sqrt{g}}^{-1} := \text{diag}(\sqrt{g(\boldsymbol{\eta}_{1})}^{-1}, \, \dots, \sqrt{g(\boldsymbol{\eta}_{N_p})}^{-1})
            \\
            &\mathbb{P}^n := \text{diag}(\mathbb{P}^n_1, \mathbb{P}^n_2, \mathbb{P}^n_3) && 
            \mathbb{P}^n_\mu := (\Lambda^n_{\mu,i}(\boldsymbol{\eta}_k))_{0\leq i \leq N^n_\mu, \,  1\leq k \leq N_p, \, n \in \{v,1,2\}, \, \mu \in \{ 1,2,3\}}
            \\
            &\mathbb{b}_0^\times := 
            \begin{pmatrix}
                0 & - \mathbb{b}^2_{0,3} & \mathbb{b}^2_{0,2}
                \\
                \mathbb{b}^2_{0,3} & 0 & -\mathbb{b}^2_{0,1}
                \\
                - \mathbb{b}^2_{0,2} & \mathbb{b}^2_{0,1} & 0
            \end{pmatrix} \qquad && 
            \mathbb{b}^2_{0,\mu} = (\widehat{\mathbf b}^2_{0,\mu}(\boldsymbol{\eta}_k))_{1 \leq k \leq N_p, \mu \in \{1,2,3\}}
        \end{alignat}

    The solution of the above system is based on the :ref:`Schur complement <schur_solver>`.

    Parameters
    ---------- 
    particles : struphy.pic.particles.Particles6D
        Particles object.

    u : psydac.linalg.block.BlockVector
        FE coefficients of MHD velocity.

    **params : dict
        Solver- and/or other parameters for this splitting step.
    '''

    def __init__(self, particles, u, **params):

        from struphy.pic.pushing.pusher import ButcherTableau

        super().__init__(particles, u)

        # parameters
        params_default = {'u_space': 'Hdiv',
                          'b': None,
                          'b_eq': None,
                          'unit_b1': None,
                          'unit_b2': None,
                          'abs_b': None,
                          'gradB1': None,
                          'curl_unit_b2': None,
                          'f0': Maxwellian5DUniform(),
                          'type': ('pcg', 'MassMatrixPreconditioner'),
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False,
                          'Ab': 1,
                          'Ah': 1,
                          'epsilon': 1.,
                          'method': 'rk4'}

        params = set_defaults(params, params_default)

        assert params['u_space'] in {'Hcurl', 'Hdiv', 'H1vec'}

        if params['u_space'] == 'H1vec':
            self._space_key_int = 0
        else:
            self._space_key_int = int(
                self.derham.space_to_form[params['u_space']])

        self._epsilon = params['epsilon']
        self._f0 = params['f0']
        self._b = params['b']
        self._b_eq = params['b_eq']
        self._unit_b1 = params['unit_b1']
        self._unit_b2 = params['unit_b2']
        self._abs_b = params['abs_b']
        self._grad_abs_b = params['gradB1']
        self._curl_norm_b = params['curl_unit_b2']
        self._info = params['info']
        self._rank = self.derham.comm.Get_rank()

        self._coupling_mat = params['Ah'] / params['Ab']
        self._coupling_vec = params['Ah'] / params['Ab']
        self._scale_push = 1

        u_id = self.derham.space_to_form[params['u_space']]
        self._E0T = self.derham.extraction_ops['0'].transpose()
        self._EuT = self.derham.extraction_ops[u_id].transpose()
        self._E1T = self.derham.extraction_ops['1'].transpose()
        self._E2T = self.derham.extraction_ops['2'].transpose()

        self._PB = getattr(self.basis_ops, 'PB')

        self._unit_b1 = self._E1T.dot(self._unit_b1)
        self._unit_b2 = self._E2T.dot(self._unit_b2)
        self._curl_norm_b = self._E2T.dot(self._curl_norm_b)

        # define system [[A B], [C I]] [u_new, v_new] = [[A -B], [-C I]] [u_old, v_old] (without time step size dt)
        _A = getattr(self.mass_ops, 'M' + u_id + 'n')

        # preconditioner
        if params['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['type'][1])
            pc = pc_class(_A)

        # Call the accumulation and Pusher class
        self._ACC = Accumulator(
            self.derham, self.domain,  params['u_space'], 'cc_lin_mhd_5d_J2', add_vector=True, symmetry='symm')

        # choose algorithm
        if params['method'] == 'forward_euler':
            a = []
            b = [1.]
            c = [0.]
        elif params['method'] == 'heun2':
            a = [1.]
            b = [1/2, 1/2]
            c = [0., 1.]
        elif params['method'] == 'rk2':
            a = [1/2]
            b = [0., 1.]
            c = [0., 1/2]
        elif params['method'] == 'heun3':
            a = [1/3, 2/3]
            b = [1/4, 0., 3/4]
            c = [0., 1/3, 2/3]
        elif params['method'] == 'rk4':
            a = [1/2, 1/2, 1.]
            b = [1/6, 1/3, 1/3, 1/6]
            c = [0., 1/2, 1/2, 1.]
        else:
            raise NotImplementedError('Chosen algorithm is not implemented.')

        self._butcher = ButcherTableau(a, b, c)
        self._pusher = Pusher(self.derham, self.domain,
                              'push_gc_cc_J2_stage_' + params['u_space'], n_stages=self._butcher.n_stages)

        _BC = -1/4 * self._ACC.operators[0]

        self._schur_solver = SchurSolver(_A, _BC,
                                         params['type'][0],
                                         pc=pc,
                                         tol=params['tol'],
                                         maxiter=params['maxiter'],
                                         verbose=params['verbose'])

        # temporary vectors to avoid memory allocation
        self._b_full1 = self._b_eq.space.zeros()
        self._b_full2 = self._E2T.codomain.zeros()
        self._u_new = u.space.zeros()
        self._u_avg1 = u.space.zeros()
        self._u_avg2 = self._EuT.codomain.zeros()
        self._tmp1 = self._abs_b.space.zeros()
        self._tmp2 = self._grad_abs_b.space.zeros()
        self._tmp3 = self._E1T.codomain.zeros()

    def __call__(self, dt):

        un = self.feec_vars[0]

        # sum up total magnetic field b_full1 = b_eq + b_tilde (in-place)
        b_full = self._b_eq.copy(out=self._b_full1)

        if self._b is not None:
            self._b_full1 += self._b

        PBb = self._PB.dot(self._b, out=self._tmp1)
        grad_PBb = self.derham.grad.dot(PBb, out=self._tmp2)
        grad_PBb += self._grad_abs_b

        Eb_full = self._E2T.dot(b_full, out=self._b_full2)
        Eb_full.update_ghost_regions()

        Egrad_PBb = self._E1T.dot(grad_PBb, out=self._tmp3)
        Egrad_PBb.update_ghost_regions()

        # accumulate MAT and VEC
        self._ACC.accumulate(self.particles[0], self._epsilon,
                             Eb_full[0]._data, Eb_full[1]._data, Eb_full[2]._data,
                             self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
                             self._unit_b2[0]._data, self._unit_b2[1]._data, self._unit_b2[2]._data,
                             self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                             Egrad_PBb[0]._data, Egrad_PBb[1]._data, Egrad_PBb[2]._data,
                             self._space_key_int, self._coupling_mat, self._coupling_vec)

        # solve linear system for updated u coefficients
        un1, info = self._schur_solver(
            un, -self._ACC.vectors[0]/2, dt, out=self._u_new)

        # call pusher kernel with average field (u_new + u_old)/2 and update ghost regions because of non-local access in kernel
        _u = un.copy(out=self._u_avg1)
        _u += un1
        _u *= 0.5

        Eu = self._EuT.dot(_u, out=self._u_avg2)
        Eu.update_ghost_regions()

        self._pusher(self.particles[0], dt,
                     self._epsilon,
                     Eb_full[0]._data, Eb_full[1]._data, Eb_full[2]._data,
                     self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
                     self._unit_b2[0]._data, self._unit_b2[1]._data, self._unit_b2[2]._data,
                     self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                     Eu[0]._data, Eu[1]._data, Eu[2]._data,
                     self._butcher.a, self._butcher.b, self._butcher.c,
                     mpi_sort='each')

        # write new coeffs into Propagator.variables
        max_du, = self.feec_vars_update(un1)

        if self._info and self._rank == 0:
            print('Status     for CurrentCoupling5DGradBxB:', info['success'])
            print('Iterations for CurrentCoupling5DGradBxB:', info['niter'])
            print('Maxdiff up for CurrentCoupling5DGradBxB:', max_du)
            print()

    @classmethod
    def options(cls):
        dct = {}
        dct['u_space'] = ['Hcurl', 'Hdiv', 'H1vec']
        dct['solver'] = {'type': [('pcg', 'MassMatrixPreconditioner'),
                                  ('cg', None)],
                         'tol': 1.e-8,
                         'maxiter': 3000,
                         'info': False,
                         'verbose': False}
        dct['algo'] = ['rk4', 'forward_euler', 'heun2', 'rk2', 'heun3']
        return dct
