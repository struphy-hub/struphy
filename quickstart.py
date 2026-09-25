import numpy as np
from matplotlib import pyplot as plt

from struphy import Simulation, domains, grids, perturbations
from struphy.models import Poisson

model = Poisson()

stab_eps = 1e-8

model.propagators.poisson.options = model.propagators.poisson.Options(
    stab_eps=stab_eps,
)

Lx = 2.0 * np.pi
Ly = 4.0 * np.pi
mode = 2
k = mode * 2.0 * np.pi / Lx
source_amp = k**2 + stab_eps

fun = perturbations.ModesCos(ls=(mode,), amps=(source_amp,))

model.em_fields.source.add_perturbation(fun)

domain = domains.Cuboid(r1=Lx, l2=-Ly / 2, r2=Ly / 2)
grid = grids.TensorProductGrid(num_elements=(64, 64, 1))

sim = Simulation(model=model, domain=domain, grid=grid)
out = sim.run()

# Plot phi in 1d along eta1 and along physical coordinate X.
# The evaluate command returns an xarray DataContainer object,
# which can be indexed like a dictionary to access the data arrays.
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

eta1 = np.linspace(0, 1, 100)
phi_1d = out.evaluate("em_fields/phi", eta1=eta1, t=-1)

x = phi_1d["X"]
phi_exact = np.cos(k * x)
phi_exact_logical = np.cos(Lx * k * eta1)

phi_1d.plot(ax=axs[0], label="Struphy")  # Plot along eta1
phi_1d.plot(x="X", ax=axs[1], label="Struphy")  # Plot along the physical coordinate X
axs[0].plot(eta1, phi_exact_logical, "k--", lw=1.8, label="exact")
axs[1].plot(x, phi_exact, "k--", lw=1.8, label="exact")

for i in range(2):
    axs[i].legend()
    axs[i].grid(alpha=0.3)
fig.savefig("quickstart_poisson_phi.png", dpi=150)
plt.show()

# Plot phi in 2d in physical coordinates
phi_2d = out.evaluate("em_fields/phi", eta1=np.linspace(0, 1, 100), eta2=np.linspace(0, 1, 100), t=-1)
phi_2d.plot(x="X", y="Y")
plt.show()
