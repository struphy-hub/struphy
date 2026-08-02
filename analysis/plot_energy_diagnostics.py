import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

DATA = (
    "thesis_postprocess_validation/data/data_proc0.hdf5"
)

OUT = "validation_plots"
os.makedirs(OUT, exist_ok=True)

with h5py.File(DATA, "r") as f:

    t = f["time/value"][:]

    en_B = f["scalar/en_B"][:]
    en_E = f["scalar/en_E"][:]
    en_J = f["scalar/en_J"][:]
    en_f = f["scalar/en_f"][:]
    en_tot = f["scalar/en_tot"][:]

relative_energy_error = (
    (en_tot - en_tot[0])
    / en_tot[0]
)

print("============================================================")
print("ENERGY DIAGNOSTICS")
print("============================================================")

print("number of states =", len(t))
print("time range =", t[0], t[-1])

print("\nInitial energies:")
print("E_B =", en_B[0])
print("E_E =", en_E[0])
print("E_J =", en_J[0])
print("E_f =", en_f[0])
print("E_total =", en_tot[0])

print("\nFinal energies:")
print("E_B =", en_B[-1])
print("E_E =", en_E[-1])
print("E_J =", en_J[-1])
print("E_f =", en_f[-1])
print("E_total =", en_tot[-1])

print("\nEnergy conservation:")
print(
    "final relative change =",
    relative_energy_error[-1],
)

print(
    "maximum absolute relative change =",
    np.max(np.abs(relative_energy_error)),
)

plt.figure(figsize=(10, 6))

plt.plot(t, en_B, label=r"$E_B$")
plt.plot(t, en_E, label=r"$E_E$")
plt.plot(t, en_J, label=r"$E_J$")
plt.plot(t, en_f, label=r"$E_f$")

plt.xlabel("Time")
plt.ylabel("Energy")
plt.title("Energy Evolution — Validation Run")
plt.legend()
plt.grid()

plt.tight_layout()

path = os.path.join(
    OUT,
    "energy_components.png",
)

plt.savefig(path, dpi=200)
plt.close()

print("\nSaved:", path)


plt.figure(figsize=(10, 6))

plt.plot(t, relative_energy_error)

plt.xlabel("Time")
plt.ylabel("Relative total-energy change")
plt.title("Total Energy Conservation")
plt.grid()

plt.tight_layout()

path = os.path.join(
    OUT,
    "total_energy_error.png",
)

plt.savefig(path, dpi=200)
plt.close()

print("Saved:", path)
