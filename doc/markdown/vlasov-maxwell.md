(disc_example)=
## Example: Vlasov-Maxwell-Poisson discretization

The Vlasov-Maxwell equations for one species in a static background provide a good example 
for PDE discretization in Struphy (see {class}`~struphy.models.kinetic.VlasovMaxwellOneSpecies`) for the full implementation). The model we are going to discretize reads as follows:

$$
\begin{align}
    &\partial_t f + \mathbf{v} \cdot  \nabla f - \frac em (\mathbf{E} + \mathbf{v} \times \mathbf{B}) 
    \cdot \frac{\partial f}{\partial \mathbf{v}} = 0 \,,
    \\[2mm]
    -\frac{1}{c^2}&\frac{\partial \mathbf{E}}{\partial t} + \nabla \times \mathbf{B} = -\mu_0 e \int_{\mathbb{R}^3} \mathbf{v} f \, \text{d} \mathbf{v} \,,
    \\[2mm]
    &\frac{\partial \mathbf{B}}{\partial t} + \nabla \times \mathbf{E} = 0 \,.
\end{align}
$$ (eq:model)

Here, $f(t, \mathbf x, \mathbf v)$ denotes the kinetic distribution function, $\mathbf E(t, \mathbf x)$ and $\mathbf B(t, \mathbf x)$ are the electric and magnetic field, respectively, $e/m$ is the charge-to-mass ratio of the electrons, $c$ denotes the speed of light and $\mu_0$ stands for the magnetic constant. In order to determine an initial electric field that is consistent with Gauss' law, one has to solve Poisson's equation once at the beginning of the simulation:

$$
\begin{equation}
 -\epsilon_0\Delta \phi = \rho_\textrm{i0} - e \int_{\mathbb{R}^3} f(t=0) \, \text{d} \mathbf{v}\,,\qquad \mathbf E(t=0) = -\nabla \phi\,.
\end{equation}
$$

Here, $\phi(\mathbf x)$ denotes the electrostatic potential, $\epsilon_0$ is the dielectric constant and $\rho_\textrm{i0}(\mathbf x)$ is a static ion background (the ion current is assumed  zero). Aside from field and particle pushing propagators, the model features also field-particle coupling propagators, and has two particle-to-grid accumulations, one charge deposition in Poisson's equation (to be solved only once at the beginning of the simulation), and one current deposition in Ampère's law.

Every model discretization for Struphy should proceed through the following steps:

1. {ref}`normalization_ex`
2. {ref}`def_spaces` (weak equations)
3. {ref}`pullback`
2. {ref}`semi_disc`
3. {ref}`time_disc`

Let us now go through these steps for the above model.

(normalization_ex)=
### Normalization

Prior to implementation, we have to find suitable units for the model quantities, a process called {ref}`normalization`. For this, let us write the Vlasov-Maxwell system in terms of units (with a hat) and dimensionless quantities (with a prime):

$$
\begin{align}
    & \frac{\hat f}{\hat t}\,\partial_{t'} f' + \frac{\hat v \hat f}{\hat x}\,\mathbf{v}' \cdot  \nabla' f' - \frac em \hat B \hat f\left(\frac{\hat E}{\hat v\hat B}\mathbf{E}'  + \mathbf{v}' \times \mathbf{B}' \right) 
    \cdot \frac{\partial f'}{\partial \mathbf{v}'} = 0 \,,
    \\[2mm]
    -\frac{1}{c^2} \frac{\hat E}{\hat t}&\frac{\partial \mathbf{E}'}{\partial t'} + \frac{\hat B}{\hat x}\nabla' \times \mathbf{B}' = -\mu_0 e\, \hat v \hat n \int_{\mathbb{R}^3} \mathbf{v}' f' \, \text{d} \mathbf{v}' \,,
    \\[3mm]
    &\frac{\hat B}{\hat t}\frac{\partial \mathbf{B}'}{\partial t'} + \frac{\hat E}{\hat x}\nabla' \times \mathbf{E}' = 0 \,,
    \\[3mm]
    &-\epsilon_0\,\frac{\hat \phi}{\hat x^2}\Delta' \phi' = e \hat n\left(\rho_\textrm{i0}' - \int_{\mathbb{R}^3} f'(t=0) \, \text{d} \mathbf{v}' \right)\,,\qquad \hat E\mathbf E'(t=0) = -\frac{\hat \phi}{\hat x}\nabla' \phi'\,.
\end{align}
$$ (eq:norm)

In Struphy, the three basic units $\hat x$, $\hat B$ and $\hat n$ are defined by the user through the {ref}`parameter file <units>`. Moreover, several other units are fixed, as described in {ref}`normalization`, namely:

$$
\hat t, \, \hat p,\, \hat \rho,\,\hat \jmath \quad \textrm{are fixed}\,.
$$

Therefore, in the present model, 

$$
\hat v,\,\hat f,\,\hat E,\,\hat \phi
$$

must be defined in order to complete the normalization process.
Let us introduce the unit of the electron cyclotron frequency and its product with the time unit,

$$
 \hat \Omega_\textrm{ce} := \frac em \hat B\qquad \varepsilon := \frac{1}{\hat \Omega_\textrm{ce} \hat t}\,.
$$

In our model, it makes sense to set the unit of the $E \times B$-velocity to $\hat v$,

$$
\frac{\hat E}{\hat B} = \hat v = \frac{\hat x}{\hat t}\,.
$$

This determines the unit $\hat E$ of the electric field and renders Faraday's law (7) scale invariant. It also sets the unit for the electric potential,

$$
 \hat \phi = \hat E \hat x = \hat v \hat B \hat x\,.
$$

In the Poisson equation, this brings into play the unit of the electron plasma frequency and its ration to the unit of the electron cyclotron frequency,

$$
 \hat \Omega_\textrm{pe} := \sqrt{\frac{e^2 \hat n}{\epsilon_0 m}}\qquad \alpha := \frac{\hat \Omega_\textrm{pe}}{\hat \Omega_\textrm{ce}}\,.
$$

Let us summarize what we have thus far, omitting the primes in {eq}`eq:norm` for clarity:

$$
\begin{align}
    & \partial_{t} f + \mathbf{v} \cdot \nabla f - \frac{1}{\varepsilon}\left(\mathbf{E}  + \mathbf{v} \times \mathbf{B} \right) 
    \cdot \frac{\partial f}{\partial \mathbf{v}} = 0 \,,
    \\[2mm]
    -\frac{\hat v^2}{c^2} &\frac{\partial \mathbf{E}}{\partial t} + \nabla \times \mathbf{B} = -\frac{\mu_0 e\, \hat v \hat n\hat x}{\hat B} \int_{\mathbb{R}^3} \mathbf{v} f \, \text{d} \mathbf{v} \,,
    \\[2mm]
    &\frac{\partial \mathbf{B}}{\partial t} + \nabla \times \mathbf{E} = 0 \,,
    \\[2mm]
    &-\Delta \phi = \frac{\alpha^2}{\varepsilon}\left(\rho_\textrm{i0} - \int_{\mathbb{R}^3} f(t=0) \, \text{d} \mathbf{v} \right)\,,\qquad \mathbf E(t=0) = -\nabla \phi\,.
\end{align}
$$

In Ampere's law we have

$$
 \frac{\mu_0 e\, \hat v \hat n\hat x}{\hat B} = \frac{\epsilon_0\mu_0 e^2\, \hat v^2 m\hat n\hat x}{\epsilon_0 me\hat B \hat v} = \frac{\hat v^2}{c^2} \frac{\hat \Omega_\textrm{pe}^2}{\hat \Omega_\textrm{ce}} \frac{\hat x}{\hat v} = \frac{\hat v^2}{c^2} \frac{\hat \Omega_\textrm{pe}^2}{\hat \Omega_\textrm{ce}^2} \frac{1}{\varepsilon}\,.
$$

Therefore, choosing the velocity unit as 

$$
 \hat v = c\,,
$$

leads to the final, Struphy-normalized equations

$$
\begin{align}
    & \partial_{t} f + \mathbf{v} \cdot \nabla f - \frac{1}{\varepsilon}\left(\mathbf{E}  + \mathbf{v} \times \mathbf{B} \right) 
    \cdot \frac{\partial f}{\partial \mathbf{v}} = 0 \,,
    \\[2mm]
    - &\frac{\partial \mathbf{E}}{\partial t} + \nabla \times \mathbf{B} = -\frac{\alpha^2}{\varepsilon} \int_{\mathbb{R}^3} \mathbf{v} f \, \text{d} \mathbf{v} \,,
    \\[2mm]
    &\frac{\partial \mathbf{B}}{\partial t} + \nabla \times \mathbf{E} = 0 \,,
    \\[2mm]
    &-\Delta \phi = \frac{\alpha^2}{\varepsilon}\left(\rho_\textrm{i0} - \int_{\mathbb{R}^3} f(t=0) \, \text{d} \mathbf{v} \right)\,,\qquad \mathbf E(t=0) = -\nabla \phi\,.
\end{align}
$$

(def_spaces)=
### Definition of solution spaces

In Struphy, kinetic equations are always solved by means of the PIC method due to the high dimensionality of the phase space. By contrast, field (and fluid) equations are solved with the FEEC method; thus one has to decide in which of the four De Rham spaces $H^1$, $H$(curl), $H$(div) or $L^2$ each of the unkowns lives. The choice is usually informed by the grad-, curl- and div-operators appearing in the model equations, so that all terms are well-defined. For instance in Faraday's law (15), the electric field $\mathbf E$ must be in the domain of the curl operator, whereas the magnetic field $\mathbf B$ must be in the co-domain (or image) of the curl operator, which implies the natural choice $\mathbf E \in H$(curl) and $\mathbf B \in H$(div). In Struphy, there is another very important guideline:

**Field (or fluid) equations containing kinetic coupling terms should be written in weak form!**

The above rule applied to the current Vlasov-Maxwell model means that Ampère's law and Poisson's equation should be written in weak form. The weak form enables a straightforward implementation of the kinetic coupling terms expressed with Dirac-delta functions. On top of that, it allows integration by parts so that grad-, curl- and div-operators can be shifted to the test functions. Therefore, we obtain the following "natural" formulation of the Vlasov-Maxwell model:

Find $(f, \mathbf E, \mathbf B, \phi) \in C^\infty \times H(\textrm{curl}) \times H(\textrm{div}) \times H^1$ such that

$$
\begin{align}
    & \partial_{t} f + \mathbf{v} \cdot  \nabla f - \frac{1}{\varepsilon}\left(\mathbf{E}  + \mathbf{v} \times \mathbf{B} \right) 
    \cdot \frac{\partial f}{\partial \mathbf{v}} = 0 \,,
    \\[2mm]
    - &\int \mathbf F \cdot \frac{\partial \mathbf{E}}{\partial t} \,\textrm d \mathbf x + \int \nabla \times \mathbf{F} \cdot \mathbf B \,\textrm d \mathbf x = -\frac{\alpha^2}{\varepsilon} \int\int_{\mathbb{R}^3}  \mathbf{v} \cdot \mathbf F f \,\, \text{d} \mathbf{v}\textrm d \mathbf x \,,\qquad \forall \ \mathbf F \in H(\textrm{curl})\,,
    \\[3mm]
    &\frac{\partial \mathbf{B}}{\partial t} + \nabla \times \mathbf{E} = 0 \,,
    \\[2mm]
    &\int \nabla \psi \cdot \nabla \phi\,\textrm d \mathbf x = \frac{\alpha^2}{\varepsilon}\left(\int \rho_\textrm{i0}\,\psi\,\textrm d \mathbf x - \int\int_{\mathbb{R}^3} f(t=0)\, \psi \, \text{d} \mathbf{v} \textrm d \mathbf x\right)\,, \qquad \forall \ \psi \in H^1\,,
    \\[4mm]
    &\mathbf E(t=0) = -\nabla \phi\,.
\end{align}
$$ (eq:spaces)

(pullback)=
### Pull-back to the logical domain

All PDE models in Struphy are discretized on the unit cube $(0,1)^3$, called the "logical domain". The mapping to the actual problem domain, a torus for instance, called "physical" or Cartesian domain and denoted by $\Omega$, is described in the [Domain base class](https://struphy.pages.mpcdf.de/struphy/sections/subsections/domains.html#struphy.geometry.base.Domain). Briefly, logical coordinates are curvi-linear and denoted by $\boldsymbol \eta \in (0, 1)^3$, whereas physical Cartesian coordinates are denoted by $\mathbf x \in \Omega$. The mapping $F: (0, 1)^3 \to \Omega, \boldsymbol \eta \mapsto \mathbf x$ is one-to-one and differentiable, with Jacobian matrix $DF: (0, 1)^3 \to \mathbb R^{3\times 3}$, metric tensor $G = DF^\top DF$ and determinant $\sqrt g = |\textrm{det} DF|$. In Struphy, only right-handed mappings with $\textrm{det} DF > 0$ are allowed.

Usually, any text book model equations are described in Cartesian (physical) coordinates $\mathbf x$, and this is also true for our model {eq}`eq:spaces`. Hence, as a next step we have to transform our model to logical, curvi-linear coordinates $\boldsymbol \eta$. This process is also called the "pull-back" to the logical domain. There are different possible "representations" of a pulled-back variable on the logical domain:

* $0$-forms and $3$-forms for scalar-valued functions,
* $1$-forms, $2$-forms or "vector-fields" for vector-valued functions.

These representations differ in their pullback formulas and have their root in differential geometry, due to coordinate-free notions of line-, surface- and volume integrals (see [Stefan's lecture notes](https://gitlab.mpcdf.mpg.de/spossann/variational-plasma/-/blob/main/lecture_notes.pdf?ref_type=heads), Chapter 5). In particular, these pullback formulas are as follows, given a scalar-valued function $\phi(\mathbf x)$ and a vector-valued function $\mathbf E(\mathbf x)$ on the physical domain:

* $0$-form pullback: $\qquad\hat \phi(\boldsymbol \eta) := \phi(F(\boldsymbol \eta))$,
* $3$-form pullback: $\qquad\hat \phi^3(\boldsymbol \eta) := \sqrt g(\boldsymbol \eta)\, \phi(F(\boldsymbol \eta))$,

for scalar-valued, and

* $1$-form pullback: $\qquad \hat{\mathbf E}^1(\boldsymbol \eta) := DF^\top(\boldsymbol \eta)\,\mathbf E(F(\boldsymbol \eta))$,
* $2$-form pullback: $\qquad \hat{\mathbf E}^2(\boldsymbol \eta) := \sqrt g(\boldsymbol \eta)\, DF^{-1}(\boldsymbol \eta)\,\mathbf E(F(\boldsymbol \eta))$,
* vector-field pullback: $\ \ \ \hat{\mathbf E}(\boldsymbol \eta) := DF^{-1}(\boldsymbol \eta)\,\mathbf E(F(\boldsymbol \eta))$,

for vector-valued functions. We see that pulled-back variables are denoted with a "hat", and with an upper index denoting the respective differential form, except for $0$-forms and vector fields, which are seen as the "standard" logical representations without an upper index. In the literature $\hat{\mathbf E}$ are called the **contra-variant** components of $\mathbf E$, moreover $\hat{\mathbf E}^1$ are called the **co-variant** components of $\mathbf E$ and $\hat{\mathbf E}^2$ is called a **pseudo-vector**.


The connection of differential $p$-forms to the {ref}`Struphy de Rham spaces <geomFE>` is as follows:

* Elements of $H^1$ are $0$-forms
* Elements of $H$(curl) are proxy functions of $1$-forms
* Elements of $H$(div) are proxy functions of $2$-forms
* Elements of $L^2$ are proxy functions of $3$-forms.

A proxy function is the representation of a differential form in a particular basis defined by a map $F$. This means nothing else than the above pullbacks have to be applied for the respective spaces. Note that there is no space for vector fields in the de Rham diagram, however [vector fields are implemented in Struphy](https://struphy.pages.mpcdf.de/struphy/sections/subsections/feec_classes.html#struphy.feec.psydac_derham.Derham.create_spline_function) as elements of $(H^1)^3$ with the correct vector field pullback.  

The grad-, curl- and div-operators transform as follows:

$$
 \nabla \phi = DF^{-\top} \hat \nabla \hat \phi\,,\qquad \nabla \times \mathbf E = \frac{DF}{\sqrt g} \hat \nabla \times \hat{\mathbf E}^1\,,\qquad \nabla \cdot \mathbf E = \frac{1}{\sqrt g} \hat \nabla \cdot \hat{\mathbf E}^2\,,
$$

where $\hat \nabla = (\partial_{\eta_1}, \partial_{\eta_2}, \partial_{\eta_3})$ denotes the usual nabla-operator on the logical domain. Looking at the above pullback formulas we deduce the following:

* the grad-operator acts on $0$-forms and transforms as a $1$-form
* the curl-operator acts on $1$-forms and transforms as a $2$-form  
* the div-operator acts on $2$-forms and transforms as a $3$-form

It is no coincidence that this exactly fits the correspondence between the de Rham diagram and differential forms given above: gradients are $1$-forms, curls are $2$-forms and divergences are $3$-forms.

Given the choice of spaces we made in writing down the model (17)-(21), we can now apply the appropriate pullback formulas to derive the model on the logical domain. For the kinetic distribution function $f$, the mapping $F:\boldsymbol \eta \mapsto \mathbf x$ only acts on the spatial coordinate; the velocity $\mathbf v$ is not tranformed. Hence we write $\hat f(t,\boldsymbol \eta, \mathbf v) := f(t, F(\boldsymbol \eta), \mathbf v)$ to obtain

$$
\begin{align}
    & \partial_{t} \hat f + \mathbf{v} \cdot DF^{-\top} \hat\nabla \hat f - \frac{1}{\varepsilon}\left(DF^{-\top}\hat{\mathbf{E}}^1  + \mathbf{v} \times \frac{DF}{\sqrt g}\hat{\mathbf{B}}^2 \right) 
    \cdot \frac{\partial \hat f}{\partial \mathbf{v}} = 0 \,,
    \\[2mm]
    - &\int \hat{\mathbf F}^1 G^{-1} \frac{\partial \hat{\mathbf{E}}^1}{\partial t} \sqrt g\,\textrm d \boldsymbol \eta + \int \hat\nabla \times \hat{\mathbf{F}}^1 G \,\hat{\mathbf B}^2 \frac{1}{\sqrt g}\,\textrm d \boldsymbol \eta = -\frac{\alpha^2}{\varepsilon} \int\int_{\mathbb{R}^3} \mathbf{v} \cdot DF^{-\top}\hat{\mathbf F}^1 \hat f \sqrt g \,\, \text{d} \mathbf{v}\textrm d \boldsymbol \eta \,,\qquad \forall \ \hat{\mathbf F}^1 \in H(\textrm{curl})\,,
    \\[3mm]
    &\frac{\partial \hat{\mathbf{B}}^2}{\partial t} + \hat \nabla \times \hat{\mathbf{E}}^1 = 0 \,,
    \\[2mm]
    &\int \hat \nabla \hat \psi \,G^{-1} \hat \nabla \hat \phi \sqrt g\,\textrm d \boldsymbol \eta = \frac{\alpha^2}{\varepsilon}\left(\int \hat \rho_\textrm{i0}\,\hat\psi \sqrt g\,\textrm d \boldsymbol \eta - \int\int_{\mathbb{R}^3} \hat f(t=0)\, \hat \psi \sqrt g \, \text{d} \mathbf{v} \textrm d \boldsymbol \eta\right)\,, \qquad \forall \ \hat \psi \in H^1\,,
    \\[4mm]
    &\hat{\mathbf E}^1(t=0) = -\hat \nabla \hat \phi\,.
\end{align} 
$$ (eq:pulledback)

Note here in particular that the third and fifth equation are independent of metric coefficients, which means they will have the same form for any mapping $F$ (they are indeed coordinate independent with the present choice of spaces). This immediately guarantees

$$
  \frac{\partial (\hat \nabla \cdot \hat{\mathbf{B}}^2)}{\partial t} = 0\,,\qquad \hat \nabla \times \hat{\mathbf E}^1(t=0) = 0\,,
$$

regardless of the mapping $F$ used in the simulation.

(semi_disc)=
### Semi-discretization in space

There are three types of terms that can appear in a Struphy discretization:

1. pure FEEC terms
2. particle equations of motion (Lagrangian scheme for kinetic equations)
3. particle-to-grid coupling terms (accumulation terms).

All of these three types are present in the above model. Let us start with the **pure FEEC terms**.
According to {ref}`geomFE` the FEEC variables are expanded with respect to the basis functions of the appropriate space,

$$
\begin{aligned}
 \hat{\mathbf{E}}^1(t, \boldsymbol\eta) \approx \hat{\mathbf{E}}^1_h(t, \boldsymbol\eta) &= \sum_{\mu=1}^3 \sum_{ijk} e_{\mu, ijk}(t)\,\vec{\Lambda}^1_{\mu, ijk}(\boldsymbol\eta) = \sum_{\mu=1}^3 \mathbf e_\mu^\top \vec{\mathbf \Lambda}^1_\mu \qquad \in V_h^1\,,
 \\[2mm]
 \hat{\mathbf{B}}^2(t, \boldsymbol\eta)  \approx \hat{\mathbf B}_h^2(t, \boldsymbol\eta) &= \sum_{\mu=1}^3 \sum_{ijk} b_{\mu, ijk}(t)\,\vec{\Lambda}^2_{\mu, ijk}(\boldsymbol\eta) = \sum_{\mu=1}^3 \mathbf b_\mu^\top \vec{\mathbf \Lambda}^2_\mu \qquad \in V_h^2\,,
 \\[3mm]
 \hat \phi(t, \boldsymbol\eta) \approx \hat \phi_h(t, \boldsymbol\eta) &= \sum_{ijk} p_{ijk}(t)\, \Lambda^0_{ijk}(\boldsymbol\eta) = \mathbf p^\top \mathbf \Lambda^0 \qquad \in V_h^0 \,,
\end{aligned}
$$

where $\mu \in \{1, 2, 3\}$ indicated the three components of a vector, hence $\vec{\Lambda}^1_{\mu, ijk}$ and $\vec{\Lambda}^2_{\mu, ijk}$ are vector-valued, and the discrete FE fields are represented by their time-dependent **FE coefficients** 
$\mathbf e = (\mathbf e_\mu)_{\mu=1}^3 \in \mathbb R^{N_1}$, $\mathbf b = (\mathbf b_\mu)_{\mu=1}^3 \in \mathbb R^{N_2}$ and $\boldsymbol \phi \in \mathbb R^{N_3}$. In Struphy's FEEC framework, it is easy to compute the grad, curl and div of a discrete FE field given in term of its coefficients:

$$
\begin{aligned}
    \hat \nabla \hat \phi_h &= \sum_{\mu=1}^3(\mathbb G_\mu \boldsymbol \phi)^\top \vec{\mathbf \Lambda}^1_\mu \quad \in \,V_h^1 \,, 
    \\
    \hat \nabla \times \hat{\mathbf E}_h^1 &= \sum_{\mu=1}^{3} \sum_{\alpha=1}^{3}(\mathbb C_{\mu, \alpha} \mathbf e_\alpha)^\top \vec{\mathbf \Lambda}^2_\mu \quad \in \,V_h^2 \,, 
    \\
    \hat \nabla \cdot \hat{\mathbf B}_h^2 &= \sum_{\mu=1}^3 (\mathbb D_\mu \mathbf b_\mu)^\top \mathbf \Lambda^3 \quad \in \,V_h^3\,.
\end{aligned}
$$

This is written in terms of the matrices $\mathbb G \in \mathbb R^{N_1 \times N_0}$,
$\mathbb C \in \mathbb R^{N_2 \times N_1}$ and $\mathbb D \in \mathbb R^{N_3 \times N_2}$,
which satisfy 

$$
    \mathbb C\, \mathbb G = 0\,,\qquad \mathbb D\, \mathbb C = 0\,.
$$

Let us substitute these discretizations in the model {eq}`eq:pulledback`:

$$
\begin{align}
    & \partial_{t} \hat f + \mathbf{v} \cdot DF^{-\top} \hat\nabla \hat f - \frac{1}{\varepsilon}\left(DF^{-\top}\hat{\mathbf{E}}^1_h  + \mathbf{v} \times \frac{DF}{\sqrt g}\hat{\mathbf{B}}^2_h \right) 
    \cdot \frac{\partial \hat f}{\partial \mathbf{v}} = 0 \,,
    \\[2mm]
    - &\sum_{\mu=1, \nu = 1}^{3,3} \mathbf f _\nu^\top \left( \int \vec{\mathbf \Lambda}^1_\mu  G^{-1} \left(\vec{\mathbf \Lambda}^1_\nu  \right)^\top\sqrt g\,\textrm d \boldsymbol \eta \right) \dot{\mathbf e}_\mu  + \sum_{\mu=1, \nu = 1}^{3,3} \sum_{\alpha = 1}^3(\mathbb C_{\nu, \alpha} \mathbf f_\alpha)^\top \left(\int \vec{\mathbf \Lambda}^2_\mu G \,\left(\vec{\mathbf \Lambda}^2_\nu\right)^\top \frac{1}{\sqrt g}\,\textrm d \boldsymbol \eta \right) \mathbf b_\mu
    \\[2mm]
    &\qquad\qquad\qquad\qquad\qquad\qquad = - \frac{\alpha^2}{\varepsilon} \sum_{\mu=1}^3 \mathbf f_\mu^\top\int\int_{\mathbb{R}^3} \mathbf{v} \cdot DF^{-\top}\vec{\mathbf \Lambda}^1_\mu \hat f \sqrt g \,\, \text{d} \mathbf{v}\textrm d \boldsymbol \eta \,,\qquad \forall \ \mathbf f = (\mathbf f_\mu)_{\mu=1}^3 \in \mathbb R^{N_1}\,, \nonumber
    \\[3mm]
    &\sum_{\mu=1}^3 \left( \dot{\mathbf b}_\mu + \sum_{\alpha=1}^{3} \mathbb C_{\mu, \alpha} \mathbf e_\alpha \right)^\top \vec{\mathbf \Lambda}^2_\mu = 0 \,,
    \\[2mm]
    &\sum_{\mu=1, \nu = 1}^{3,3} (\mathbb G_\mu \boldsymbol \psi)^\top \left(\int \vec{\mathbf \Lambda}^1_\mu \,G^{-1} \left( \vec{\mathbf \Lambda}^1_\nu\right)^\top \sqrt g\,\textrm d \boldsymbol \eta \right) \mathbb G_\nu \boldsymbol \phi = \frac{\alpha^2}{\varepsilon}\boldsymbol \psi^\top\left( \int \hat \rho_\textrm{i0}\,\mathbf \Lambda^0 \sqrt g\,\textrm d \boldsymbol \eta - \int\int_{\mathbb{R}^3} \hat f(t=0)\, \mathbf \Lambda^0 \sqrt g \, \text{d} \mathbf{v} \textrm d \boldsymbol \eta\right)\,, \qquad \forall \ \boldsymbol \psi \in \mathbb R^{N_0}\,,
    \\[4mm]
    &\sum_{\mu=1}^3 \mathbf (\mathbf e_\mu + \mathbb G_\mu \boldsymbol \phi)^\top \vec{\mathbf \Lambda}^1_\mu = 0\,.
\end{align}
$$

There appear a couple of mass matrices that are already [predefined in Struphy](https://struphy.pages.mpcdf.de/struphy/sections/subsections/feec_classes.html#struphy.feec.mass.WeightedMassOperators.M1), for any mapping:

$$
\begin{aligned}
 &\mathbb M^1= (\mathbb M^1_{\mu, \nu})_{\mu, \nu=1, 1}^{3, 3}\,,\qquad \mathbb M^1_{\mu, \nu} := \int \vec{\mathbf \Lambda}^1_\mu  G^{-1} \left(\vec{\mathbf \Lambda}^1_\nu  \right)^\top \sqrt g\,\textrm d \boldsymbol \eta\,,
 \\[2mm]
 &\mathbb M^2= (\mathbb M^2_{\mu, \nu})_{\mu, \nu=1, 1}^{3, 3}\,,\qquad \mathbb M^2_{\mu, \nu} := \int \vec{\mathbf \Lambda}^2_\mu G \,\left(\vec{\mathbf \Lambda}^2_\nu\right)^\top \frac{1}{\sqrt g}\,\textrm d \boldsymbol \eta\,.
\end{aligned}
$$

These are 3x3 block matrices (implemented as [BlockLinearOperators](https://struphy.pages.mpcdf.de/struphy/tutorials/tutorial_08_data_structures.html#FEEC-data-structures)), where the blocks are indexed by $(\mu, \nu)$. We remark that the weak equations must hold for any choice of $\mathbf f = (\mathbf f_\mu)_{\mu=1}^3 \in \mathbb R^{N_1}$ and $ \boldsymbol \psi \in \mathbb R^{N_0}$, respectively, which means that these can be factored out to lead to a system of equations. Besides, all basis functions are linearly independent such that the coefficients in the third and fifth equation must vanish separately. This leads to the much more compact notation

$$
\begin{align}
    & \partial_{t} \hat f + \mathbf{v} \cdot DF^{-\top} \hat\nabla \hat f - \frac{1}{\varepsilon}\left(DF^{-\top}\hat{\mathbf{E}}^1_h  + \mathbf{v} \times \frac{DF}{\sqrt g}\hat{\mathbf{B}}^2_h \right) 
    \cdot \frac{\partial \hat f}{\partial \mathbf{v}} = 0 \,,
    \\[2mm]
    - \mathbb M^1 &\dot{\mathbf e}  + \mathbb C^\top \mathbb M^2 \mathbf b = - \frac{\alpha^2}{\varepsilon} \int\int_{\mathbb{R}^3} \mathbf{v} \cdot DF^{-\top}\vec{\mathbf \Lambda}^1 \hat f \sqrt g \,\, \text{d} \mathbf{v}\textrm d \boldsymbol \eta \,,
    \\[3mm]
    &\dot{\mathbf b} + \mathbb C \mathbf e  = 0 \,,
    \\[2mm]
    &\mathbb G^\top \mathbb M^1\mathbb G \boldsymbol \phi = \frac{\alpha^2}{\varepsilon}\left( \int \hat \rho_\textrm{i0}\,\mathbf \Lambda^0 \sqrt g\,\textrm d \boldsymbol \eta - \int\int_{\mathbb{R}^3} \hat f(t=0)\, \mathbf \Lambda^0 \sqrt g \, \text{d} \mathbf{v} \textrm d \boldsymbol \eta\right)\,, 
    \\[4mm]
    &\mathbf e + \mathbb G\boldsymbol \phi = 0\,.
\end{align}
$$ (eq:compact)

The next step is to discretize the kinetic equation by means of {ref}`particle_discrete`, which leads us to **particle equations of motion**. For this, the volume form $\hat f^\textrm{vol} := \hat f \sqrt g$ in "logical" phase space is such that it includes the measure of the phase space, i.e. the Jacobian determinant arising from coordinate transformations in phase space. The volume form is then aproximated by a sum of Dirac delta functions,

$$
\begin{equation}
 \hat f^\textrm{vol}(t, \boldsymbol \eta, \mathbf v) \approx \hat f^\textrm{vol}_h(t, \boldsymbol \eta, \mathbf v) = \frac 1 N\sum_{p=0}^{N-1} w_p \,\delta(\boldsymbol \eta - \boldsymbol \eta_p(t))\,\delta(\mathbf v - \mathbf v_p(t))\,,
\end{equation}
$$ (eq:pic)

where $\boldsymbol \eta_p(t)$ and $\mathbf v_p(t)$ satisfy the characteristics of the kinetic transport equation in {eq}`eq:compact`, that is

$$
\begin{align}
 \dot{\boldsymbol \eta}_p &= DF^{-1}(\boldsymbol \eta_p) \mathbf v_p\,,
 \\[2mm]
 \dot{\mathbf v}_p &= -\frac{1}{\varepsilon}\left(DF^{-\top}(\boldsymbol \eta_p)\hat{\mathbf{E}}^1_h (\boldsymbol \eta_p)  + \mathbf{v} \times \frac{DF}{\sqrt g} (\boldsymbol \eta_p)\hat{\mathbf{B}}^2_h (\boldsymbol \eta_p) \right)\,.
\end{align}
$$

Since the number of particles in PIC simulations is usually very large (on the order of millions or even billions), an efficient solution loop over $p$ (sometimes also $k$ is used as the particle index) is absolutely mandatory here. Therefore, specific [pusher kernels](https://github.com/struphy-hub/struphy/blob/devel/src/struphy/pic/pushing/pusher_kernels.py) must be written for each particle pushing step, which are then accelerated (compiled) with Pyccel (see our [Tl:dr](https://struphy-hub.github.io/struphy/sections/abstract.html)) to enable C- or Fortran execution speed. In Struphy models, the pusher kernels are integrated via the [Pusher class](https://github.com/struphy-hub/struphy/blob/devel/src/struphy/pic/pushing/pusher.py) that provides some syntactic sugar for calling the kernels. 
See {ref}`prop_kernels` for more details.

Now that we know how to discretize the kinetic equation by means of a Lagrangian particle method, it remains to tackle the right-hand sides of Ampère's law and of Poisson's equation in {eq}`eq:compact`. In the latter, there is the source term

$$
 \boldsymbol \rho_\textrm{i0} := \frac{\alpha^2}{\varepsilon}\int \hat \rho_\textrm{i0}\,\mathbf \Lambda^0 \sqrt g\,\textrm d \boldsymbol \eta = \left(\frac{\alpha^2}{\varepsilon} \rho_\textrm{i0}\,, \, \boldsymbol\Lambda^0\right)_{L^2} \qquad \in \mathbb R^{N_0}\,,
$$

coming from the static ion charge density $\rho_\textrm{i0}$. This term can be viewed as the right-hand side of an [L2Projection](https://struphy.pages.mpcdf.de/struphy/sections/subsections/feec_classes.html#struphy.feec.projectors.L2Projector) into $V_h^0$ and computed with the method [L2Projector.get_dofs](https://struphy.pages.mpcdf.de/struphy/sections/subsections/feec_classes.html#struphy.feec.projectors.L2Projector.get_dofs). 

Finally, let us consider the **particle-to-grid coupling terms**. In the Poisson equation we have the coupling term

$$
 \begin{aligned}
  S_{ijk} := -\frac{\alpha^2}{\varepsilon}\int\int_{\mathbb{R}^3} \hat f\, \Lambda^0_{ijk} \sqrt g \, \text{d} \mathbf{v} \textrm d \boldsymbol \eta &= -\frac{\alpha^2}{\varepsilon}\int\int_{\mathbb{R}^3} \hat f^\textrm{vol}\, \Lambda^0_{ijk}  \, \text{d} \mathbf{v} \textrm d \boldsymbol \eta
  \\[2mm]
  & \approx -\frac{\alpha^2}{\varepsilon} \frac 1 N\sum_{p=0}^{N-1} w_p \Lambda^0_{ijk}(\boldsymbol \eta_p)\,,
 \end{aligned}
$$

where we inserted the PIC ansatz {eq}`eq:pic` in the last line. The result is a vector $\mathbf S = (S_{ijk}) \in \mathbb R^{N_0}$, that can be stored as a distributed [StencilVector](https://struphy.pages.mpcdf.de/struphy/tutorials/tutorial_08_data_structures.html#FEEC-data-structures) of the space $V_h^0$. We can write this in more compact notation by introducing the rectangular matrix

$$
 \mathbb L^0 \in \mathbb R^{N_0 \times N}\,,\qquad \mathbb L^0_{ijk,p} = \Lambda^0_{ijk}(\boldsymbol \eta_p) \in \mathbb R\,,
$$

as well as the "weight vector" $\mathbf w = \left( w_p/N \right)_{p=0}^{N-1} \in \mathbb R^N$, which leads to

$$
 \mathbf S = - \frac{\alpha^2}{\varepsilon}\mathbb L^0\mathbf w\,.
$$

Here again, because of the large number of particles in a PIC simulation, the sum over $p$ must be very efficient and accelerated with Pyccel, see our [Tl:dr](https://struphy.pages.mpcdf.de/struphy/sections/abstract.html), to enable C- or Fortran execution speed. The integration of these [accumulation kernels](https://struphy.pages.mpcdf.de/struphy/sections/subsections/accumulators.html#module-struphy.pic.accumulation.accum_kernels) in Struphy is done through the [Accumulator base classes](https://struphy.pages.mpcdf.de/struphy/sections/subsections/accumulators.html#module-struphy.pic.accumulation.particles_to_grid), which provide some syntactic sugar for calling the kernels.
The above expression can be assembled with [AccumulatorVector](https://struphy.pages.mpcdf.de/struphy/sections/subsections/accumulators.html#struphy.pic.accumulation.particles_to_grid.AccumulatorVector), where in the kernel the weight is 

$$
B_p = -\frac{\alpha^2}{\varepsilon} \frac 1 N w_p\,.
$$

When writing an accumulation kernel, it is mandatory to follow the instructions in [a_documentation](https://struphy.pages.mpcdf.de/struphy/sections/subsections/accumulators.html#struphy.pic.accumulation.accum_kernels.a_documentation).
As an example, we can look at the kernel [poisson](https://struphy.pages.mpcdf.de/struphy/sections/subsections/accumulators.html#struphy.pic.accumulation.accum_kernels.poisson) (see [source code](https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/src/struphy/pic/accumulation/accum_kernels.py#L86)), which is needed in the present example and implements the above weights $B_p$. Inside of the particle loop, an accumulation kernel consists of three steps:

1. Extract the marker position $\boldsymbol \eta_p$ and other relevant marker quantities.
2. Compute the "filling function", denoted $A^{\mu,\nu}_p$ or $B^\mu_p$ in the [Accumulator docstring](https://struphy.pages.mpcdf.de/struphy/sections/subsections/accumulators.html#struphy.pic.accumulation.particles_to_grid.Accumulator).
3. Accumulate the contributions of the particle into the array by calling the correct kernel from the module:

        import struphy.pic.accumulation.particle_to_mat_kernels as particle_to_mat_kernels

which is imported at the top of [accum_kernels.py](https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/src/struphy/pic/accumulation/accum_kernels.py?ref_type=heads) and [accum_kernels_gc.py](https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/src/struphy/pic/accumulation/accum_kernels_gc.py?ref_type=heads), respectively. The coupling term in Ampère's law can be treated in analogous fashion:

$$
\begin{aligned}
 S^\mu_{ijk} := - \frac{\alpha^2}{\varepsilon} \int\int_{\mathbb{R}^3} \mathbf{v} \cdot DF^{-\top}\vec{\Lambda}^1_{\mu,ijk}\, \hat f \sqrt g \,\, \text{d} \mathbf{v}\textrm d \boldsymbol \eta &= - \frac{\alpha^2}{\varepsilon} \int\int_{\mathbb{R}^3} (DF^{-1}\mathbf{v}) \cdot  \vec{\Lambda}^1_{\mu,ijk}\, \hat f^\textrm{vol} \,\, \text{d} \mathbf{v}\textrm d \boldsymbol \eta
 \\[2mm]
 &\approx -\frac{\alpha^2}{\varepsilon} \frac 1 N\sum_{p=0}^{N-1} w_p (DF^{-1}(\boldsymbol \eta_p) \, \mathbf v_p) \cdot \vec \Lambda^1_{\mu, ijk}(\boldsymbol \eta_p)\,.
\end{aligned}
$$

where we inserted the PIC ansatz {eq}`eq:pic` in the last line. Here, it is important to realize that $\vec \Lambda^1_{\mu, ijk} \in \mathbb R^3$ is vector-valued, as defined in {ref}`geomFE`, and contracted with $DF^{-1}(\boldsymbol \eta_p) \, \mathbf v_p \in \mathbb R^3$ for each $\mu \in \{1, 2, 3\}$. The result of this is a vector $\mathbf S = (S^1_{ijk}, S^2_{ijk}, S^3_{ijk}) \in \mathbb R^{N_1}$, that can be stored as a distributed [BlockVector](https://struphy.pages.mpcdf.de/struphy/tutorials/tutorial_08_data_structures.html#FEEC-data-structures) of the space $V_h^1$.
 We can write this in more compact notation by introducing the matrix

$$
 \mathbb L^1 \in \mathbb R^{N_1 \times (N \times 3)}\,,\qquad \mathbb L^1_{(\mu,ijk),p} = \vec\Lambda^1_{\mu,ijk}(\boldsymbol \eta_p) \in \mathbb R^3\,,
$$

as well as the large matrices

$$
 \bar{DF}^{-1} = \textrm{diag}(DF^{-1}(\boldsymbol \eta_p)) \in \mathbb R^{(N\times 3) \times (N \times 3)}\,,\qquad \bar{\mathbf w} = \textrm{diag}( w_p/N) \otimes I_{3\times 3} \in \mathbb R^{(N \times 3) \times (N \times 3) } 
$$

which leads to

$$
 \mathbf S = - \frac{\alpha^2}{\varepsilon}\mathbb L^1 \bar{DF}^{-1} \bar{\mathbf w}\, \mathbf v\,,
$$

with $\mathbf v = (\mathbf v_p)_{p=0}^{N-1} \in \mathbb R^{N\times 3}$. In summary, this leads to the following semi-discrete Vlasov-Maxwell system, which is a coupled, nonlinear system of ordinary differential equations (ODEs):

$$
\begin{align}
    \dot{\boldsymbol \eta} &= \bar{DF}^{-1} \mathbf v\,,
    \\[2mm]
    \dot{\mathbf v} &= -\frac{1}{\varepsilon}\left( \bar{DF}^{-\top} (\mathbb L^1)^\top \mathbf e   + \bar{\mathbf B}^2_\times\mathbf{v} \right)\,,
    \\[2mm]
    - \mathbb M^1 &\dot{\mathbf e}  + \mathbb C^\top \mathbb M^2 \mathbf b = - \frac{\alpha^2}{\varepsilon} \mathbb L^1 \bar{DF}^{-1} \bar{\mathbf w}\, \mathbf v \,,
    \\[3mm]
    &\dot{\mathbf b} + \mathbb C \mathbf e  = 0 \,,
    \\[2mm]
    &\mathbb G^\top \mathbb M^1\mathbb G \boldsymbol \phi = \frac{\alpha^2}{\varepsilon}\left( \boldsymbol \rho_\textrm{i0} - \mathbb L^0\mathbf w\right)\,, 
    \\[4mm]
    &\mathbf e + \mathbb G\boldsymbol \phi = 0\,.
\end{align}
$$ (eq:semidisc)

where we introduced the short-hand notation for the operator

$$
 \bar{\mathbf B}^2_\times (\cdot) = \textrm{diag} \left( (\cdot)_p\times \frac{DF}{\sqrt g} (\boldsymbol \eta_p)\hat{\mathbf{B}}^2_h (\boldsymbol \eta_p) \right) \in \mathbb R^{(N\times 3) \times (N \times 3)}\,.
$$

(time_disc)=
### Time discretization

In Struphy, the time discretization is usually informed by the Hamiltonian structure of the model equations. We refer to the time discretization of the simpler [electrostatic Vlasov-Poisson system](https://gitlab.mpcdf.mpg.de/struphy/struphy-projects/-/blob/main/running-projects/2024_VlasovPoissonInhomB.md?ref_type=heads#time-discretization) for a brief introduction. In case of a non-ideal model with dissipation, there usually is an ideal "core model" that is Hamiltonian if the dissipative terms are neglected, which can inform the basic time discretization. The Vlasov-Maxwell system considered here has such a Hamiltonian structure, see for instance [Kraus et al.](https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/gempic-geometric-electromagnetic-particleincell-methods/C32D97F1B5281878F094B7E5075D291A). In most cases, the Hamiltonian structure can be retrieved after semi-discretization is space, here thus from the system {eq}`eq:semidisc`. The methodology goes as follows:

1. Write down the energy $H$ conserved by the continuous model (and apply the chosen normalization).
2. Discretize the continuous energy and write it as a function $H(\mathbf Z)$ of the unknowns $\mathbf Z$ (i.e. spline coefficients and particle positions and velocities). Use the same spaces as in the equations.
3. Compute the gradient of the discrete energy with respect to $\mathbf Z$.
4. Deduce the Poisson matrix $\mathbb J(\mathbf Z)$ by comparing $\dot{\mathbf Z} = \mathbb J(\mathbf Z)\nabla H(\mathbf Z)$ to your semi-discrete equations derived beforehand.

Let us execute this program for our Vlasov-Maxwell example. The energy conserved by the continuous model {eq}`eq:model` reads

$$
 H(f, \mathbf E, \mathbf B) = \frac m2 \int |\mathbf v|^2 f\,\textrm d \mathbf v \textrm d \mathbf x + \frac{\epsilon_0}{2} \int |\mathbf E|^2\,\textrm d \mathbf x + \frac{1}{2 \mu_0} \int |\mathbf B|^2\,\textrm d \mathbf x\,.
$$

We urge the reader to verify that this energy is indeed conserved by the dynamics {eq}`eq:model`. Normalizing this energy with respect to the energy scale $\hat H = (\hat B^2/\mu_0) \hat x^3$ and omitting primes yields

$$
 H(f, \mathbf E, \mathbf B) = \frac{\alpha^2}{2} \int |\mathbf v|^2 f\,\textrm d \mathbf v \textrm d \mathbf x + \frac{1}{2} \int |\mathbf E|^2\,\textrm d \mathbf x + \frac{1}{2 } \int |\mathbf B|^2\,\textrm d \mathbf x\,.
$$

We can now insert the chosen solution spaces, $\mathbf E \in H$(curl) and $\mathbf B \in H$(div), then pull-back the expression to the logical domain and insert the above space discretizations to obtain

$$
\begin{equation}
 H(\boldsymbol \eta, \mathbf v, \mathbf e, \mathbf b) = \frac{\alpha^2}{2} \mathbf v^\top \bar{\mathbf w}\, \mathbf v + \frac{1}{2 } \mathbf e^\top \mathbb M^1 \mathbf e + \frac{1}{2 } \mathbf b^\top \mathbb M^2 \mathbf b\,.
 \end{equation}
$$ (eq:H)

Here, the vector of unknowns $\mathbf Z = (\boldsymbol \eta, \mathbf v, \mathbf e, \mathbf b) \in \mathbb R^{(6N + N_1 + N_2)}$ consists of all particle positions and velocities as well as the FE coefficients of the electric and magnetic field. The gradient of $H(\mathbf Z)$ with respect to $\mathbf Z$ reads

$$
 \nabla H(\mathbf Z) = \begin{pmatrix}
 0
 \\
 \alpha^2 \bar{\mathbf w}\mathbf v
 \\
 \mathbb M^1 \mathbf e
 \\
 \mathbb M^2 \mathbf b 
 \end{pmatrix}\,.
$$

We can thus write {eq}`eq:semidisc` as

$$
\begin{equation}
\begin{pmatrix}
 \dot{\boldsymbol \eta}
 \\
 \dot{\mathbf v}
 \\
 \dot{\mathbf e}
 \\
 \dot{\mathbf b} 
 \end{pmatrix} 
 = 
 \begin{pmatrix}
  0 & \frac{1}{\alpha^2} \bar{DF}^{-1}\bar{\mathbf w}^{-1} & 0 & 0
 \\
  - \frac{1}{\alpha^2} \bar{\mathbf w}^{-\top} \bar{DF}^{-\top}& -\frac{1} {\epsilon \alpha^2} \bar{\mathbf B}^2_\times \bar{\mathbf w}^{-1} & -\frac{1}{\varepsilon} \bar{DF}^{-\top} (\mathbb L^1)^\top (\mathbb M^1)^{-1} & 0
 \\
  0 & \frac{1}{\varepsilon} (\mathbb M^1)^{-1} \mathbb L^1 \bar{DF}^{-1}  & 0 & (\mathbb M^1)^{-1} \mathbb C^\top 
 \\
  0 & 0 & -\mathbb C (\mathbb M^1)^{-1} & 0
 \end{pmatrix}
 \begin{pmatrix}
 0
 \\
 \alpha^2 \bar{\mathbf w}\mathbf v
 \\
 \mathbb M^1 \mathbf e
 \\
 \mathbb M^2 \mathbf b 
 \end{pmatrix}\,.
 \end{equation}
$$ (eq:hamilton)

Quite magically, our space discretization led to the skew symmetric Poisson matrix $\mathbb J(\mathbf Z)$ in $\dot{\mathbf Z} = \mathbb J(\mathbf Z)\nabla H(\mathbf Z)$. The skew symmetry of this matrix guarantees energy conservation, because

$$
 \dot H(\mathbf Z) = \nabla H^\top \dot{\mathbf Z} = \nabla H^\top \mathbb J \nabla H = - \nabla H^\top \mathbb J^\top \nabla H = 0\,.
$$

Clearly, solving the whole system {eq}`eq:hamilton` with an implicit mid-point scheme (Crank-Nicolson) would lead to exact conservation of the energy {eq}`eq:H` in the fully time discrete algorithm. However, this system is extremely large and we thus aim for a time splitting algorithm to reduce the system size in each split step. In doing so our aim is to preserve the skew symmetry of $\mathbb J$ when designing the time splitting scheme. This is called **Poisson splitting**. In the above system, we can split as follows:

$$
 \mathbb J = \mathbb J_1 + \mathbb J_2 + \mathbb J_3 + \mathbb J_4\,, 
 $$

 with

 $$
 \begin{align}
 \mathbb J_1 &= \begin{pmatrix}
  0 & \frac{1}{\alpha^2} \bar{DF}^{-1}\bar{\mathbf w}^{-1} & 0 & 0
 \\
  - \frac{1}{\alpha^2} \bar{\mathbf w}^{-\top} \bar{DF}^{-\top}& 0 & 0 & 0
 \\
  0 & 0 & 0 & 0
 \\
  0 & 0 & 0 & 0
 \end{pmatrix}
 \\[2mm]
 \mathbb J_2 &= \begin{pmatrix}
  0 & 0 & 0 & 0
 \\
  0 & -\frac{1} {\epsilon \alpha^2} \bar{\mathbf B}^2_\times \bar{\mathbf w}^{-1} & 0 & 0
 \\
  0 & 0 & 0 & 0
 \\
  0 & 0 & 0 & 0
 \end{pmatrix}
 \\[2mm]
 \mathbb J_3 &= \begin{pmatrix}
  0 & 0 & 0 & 0
 \\
  0 & 0 & -\frac{1}{\varepsilon} \bar{DF}^{-\top} (\mathbb L^1)^\top (\mathbb M^1)^{-1} & 0
 \\
  0 & \frac{1}{\varepsilon} (\mathbb M^1)^{-1} \mathbb L^1 \bar{DF}^{-1} & 0 & 0
 \\
  0 & 0 & 0 & 0
 \end{pmatrix}
 \\[2mm]
 \mathbb J_4 &= \begin{pmatrix}
  0 & 0 & 0 & 0
 \\
  0 & 0 & 0 & 0
 \\
  0 & 0 & 0 & (\mathbb M^1)^{-1} \mathbb C^\top 
 \\
  0 & 0 & -\mathbb C (\mathbb M^1)^{-1} & 0
 \end{pmatrix}\,.
 \end{align}
$$ (eq:Js)

These four split Poisson matrices define the four substeps of the time splitting algorithm, called {ref}`propagators` in Struphy. These are maps $\Phi_t^{n}:\mathbf Z_0 \mapsto \mathbf Z(t)$ defined via the solution of

$$
  \qquad \dot{\mathbf Z} = \mathbb J_n(\mathbf Z)\nabla H(\mathbf Z)\quad \mathbf Z(0) = \mathbf Z_0\,,\qquad n \in\{1,2,3,4\}\,.
$$

If possible, we shall choose an energy conserving time discretization of each substep. An entire time step is then achieved by composition of the substeps; the simplest, first-order composition is Lie-Trotter splitting defined by

$$
 \mathbf Z(t + \Delta t) = (\Phi_{\Delta t}^{4} \circ \Phi_{\Delta t}^{3} \circ \Phi_{\Delta t}^{2} \circ \Phi_{\Delta t}^{1}) \mathbf Z(t)
$$

A more sophisticated, symmetric second-order splitting is the Strang splitting,

$$
 \mathbf Z(t + \Delta t) = (\Phi_{\Delta t/2}^{4} \circ \Phi_{\Delta t/2}^{3} \circ \Phi_{\Delta t/2}^{2} \circ \Phi_{\Delta t}^{1} \circ \Phi_{\Delta t/2}^{2} \circ \Phi_{\Delta t/2}^{3} \circ \Phi_{\Delta t/2}^{4}) \mathbf Z(t)\,.
$$

Once the propagators have been defined and added to a {ref}`struphy_model`, Struphy performs the compositions automatically; the user can choose the splitting algorithm in the {ref}`parameter file <time>`. 


For our example model {class}`~struphy.models.kinetic.VlasovMaxwellOneSpecies`, the four substeps defined by {eq}`eq:Js` are imlemented in the following propagators:

1. $\Phi^1_{t}$ in {class}`~struphy.propagators.propagators_markers.PushEta`,

2. $\Phi^2_{t}$ in {class}`~struphy.propagators.propagators_markers.PushVxB`,

3. $\Phi^3_{t}$ in {class}`~struphy.propagators.propagators_coupling.VlasovAmpere`,

4. $\Phi^4_{t}$ in {class}`~struphy.propagators.propagators_fields.Maxwell`.

Let us revisit the third step $\Phi^3_{t}$, which is the most complicated because it is a particle-field coupling step, where marker velocities and FE coefficients are updated simultaneously. Explicitly, the ODE of this step reads

$$
\begin{aligned}
\dot{\mathbf e} &= \frac{\alpha^2}{\varepsilon} (\mathbb M^1)^{-1}\mathbb L^1 \bar{DF}^{-1} \bar{\mathbf w}\, \mathbf v\,,
\\[2mm]
    \dot{\mathbf v} &= -\frac{1}{\varepsilon} \bar{DF}^{-\top} (\mathbb L^1)^\top \mathbf e  \,,
\end{aligned}
$$

which is discretized with an energy-conserving mid-point rule, 

$$
\begin{aligned}
\frac{\mathbf e^{n+1} - \mathbf e^{n}}{\Delta t}  &= \frac{\alpha^2}{\varepsilon} (\mathbb M^1)^{-1}\mathbb L^1 \bar{DF}^{-1} \bar{\mathbf w}\, \frac{\mathbf v^{n+1} + \mathbf v^{n}}{2} \,,
\\[2mm]
    \frac{\mathbf v^{n+1} - \mathbf v^{n}}{\Delta t} &= -\frac{1}{\varepsilon} \bar{DF}^{-\top} (\mathbb L^1)^\top \frac{\mathbf e^{n+1} + \mathbf e^{n}}{2}   \,. 
\end{aligned}
$$

We note tha the particle positions $\boldsymbol \eta$ are constant in this split step. Multiplying the first equation with $\Delta t\mathbb M^1$, the second equation with $\Delta t$ we obtain a linear system for the unknowns $(\mathbf e^{n+1}, \mathbf v^{n+1})$,

$$
 \begin{pmatrix}
 \mathbb M^1 & - \frac{\Delta t }{2} \frac{\alpha^2}{\varepsilon} \mathbb L^1 \bar{DF}^{-1} \bar{\mathbf w}
 \\
 \frac{\Delta t }{2} \frac{1}{\varepsilon} \bar{DF}^{-\top} (\mathbb L^1)^\top & \textrm{Id}
 \end{pmatrix}
 \begin{pmatrix}
 \mathbf e^{n+1} 
 \\
 \mathbf v^{n+1}
 \end{pmatrix}
 =
 \begin{pmatrix}
 \mathbb M^1 & \frac{\Delta t }{2} \frac{\alpha^2}{\varepsilon} \mathbb L^1 \bar{DF}^{-1} \bar{\mathbf w}
 \\
 -\frac{\Delta t }{2} \frac{1}{\varepsilon} \bar{DF}^{-\top} (\mathbb L^1)^\top & \textrm{Id}
 \end{pmatrix}
 \begin{pmatrix}
 \mathbf e^{n} 
 \\
 \mathbf v^{n}
 \end{pmatrix}\,.
$$

The solution of this system requires the inversion of the large 2x2 block matrix on the left-hand side. In Struphy such systems can be conveniently solved with the {class}`Schur solver class <struphy.linear_algebra.schur_solver>`. For this one needs to invert the Schur complement $S = A - BC$ of the 2x2 block matrix, which in our case reads

$$
 S = \mathbb M^1 + \frac{\Delta t^2 }{4} \frac{\alpha^2}{\varepsilon^2} \mathbb L^1 \bar{DF}^{-1} \bar{\mathbf w} \bar{DF}^{-\top} (\mathbb L^1)^\top \qquad \in \mathbb R^{N_1 \times N_1}\,.
$$

This matrix size $N_1 \times N_1$ is independent of the particle number $N$ and an inversion is thus feasible. Indeed, using the Schur complement amounts to inserting one equation into the other, thereby eliminating one variable from the solution step. Moreover, the term

$$
 M^{\mu, \nu}_{ijk, mno} := \frac{\Delta t^2 }{4} \frac{\alpha^2}{\varepsilon^2} \mathbb L^1_{(\mu,ijk)} \bar{DF}^{-1} \bar{\mathbf w} \bar{DF}^{-\top} (\mathbb L^1)^\top_{(\nu,mno)}\,,
$$

is a classic accumulation term into a matrix $\mathbb M = (M^{\mu, \nu}_{ijk, mno}) \in \mathbb R^{N_1 \times N_1}$ of the same size as the mass matrix $\mathbb M^1$. In Struphy, such a term can be conveniently handled with {class}`Accumulator <struphy.pic.accumulation.particles_to_grid.Accumulator>`.