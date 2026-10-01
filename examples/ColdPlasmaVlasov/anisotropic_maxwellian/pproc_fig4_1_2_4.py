"""
Historical thesis Figures 4.1, 4.2 and 4.4 postprocessor.

Uses the existing Standard-FEM + PIC Run-1 energy dataset:
    thesis_run1_standard_cv/run1_standard_cv.npz

Figure 4.3 is intentionally NOT produced here. The historical thesis
plotting notebook used a separate Standard-FEM particle run with control=0
for the final marker distribution. See params_fig4_3.py.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("thesis_run1_standard_cv")
DATA = ROOT / "run1_standard_cv.npz"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

d = np.load(DATA)
t = d["time"]
en_E = d["en_E"]
en_B = d["en_B"]
en_Bx = d["en_Bx"]
en_jc = d["en_jc"]
en_jh = d["en_jh"]

en_tot = en_B + en_E + en_jc + en_jh
E0 = en_tot[0]

# Historical Run-1 dispersion values used in the thesis.
omega_r = 0.4742
gamma = 0.0447

# Historical plotting colours.
colors = ["darkorange", "purple", "sienna", "royalblue"]

plt.rc("font", family="serif")
plt.rcParams.update({"font.size": 10})
plt.rc("text", usetex=False)

# ============================================================
# Figure 4.1
# ============================================================
fig = plt.figure(figsize=(6.0, 4.0))
ax = fig.add_subplot(111)

ax.semilogy(t, en_B/E0, linewidth=1, label=r"$\mathcal{E}_{\bar B}$",
            color=colors[0])
ax.semilogy(t, en_E/E0, linewidth=1, label=r"$\mathcal{E}_{\bar E}$",
            color=colors[1])
ax.semilogy(t, en_jc/E0, "--", linewidth=1, label=r"$\mathcal{E}_{\mathrm{c}}$",
            color=colors[2])
ax.semilogy(t, en_jh/E0, linewidth=1, label=r"$\mathcal{E}_{\mathrm{h}}$",
            color=colors[3])

tg = np.linspace(0, 150, 100)
growth = 9e-9*np.exp(2*gamma*tg)
ax.semilogy(tg, growth, "k--", linewidth=1, label="expected growth rate")

ax.set_xlabel(r"$t|\Omega_\mathrm{ce}|$")
ax.set_ylabel(r"$\mathcal{E}/\mathcal{E}(0)$")
ax.set_title("Partition of energy", fontsize=10)
ax.set_ylim(1e-12, 1e1)
ax.set_xlim(0, 200)
ax.legend(loc="upper center", ncol=1, bbox_to_anchor=(1.55, 1.05),
          fontsize=7)
fig.savefig(OUT/"fig4_1.png", dpi=400, bbox_inches="tight")
fig.savefig(OUT/"fig4_1.pdf", bbox_inches="tight")
plt.close(fig)

# ============================================================
# Figure 4.2 — Spectrogram in linear phase
# ============================================================

# Use the magnetic x-component ENERGY directly from the
# completed Standard-FEM/PIC simulation.
en_Bx = np.asarray(d['en_Bx'], float)

# Historical FFT window
Ntend = 8000
DT = 0.0125
OMEGA_R = 0.4742

spec = np.fft.fft(en_Bx[0:Ntend + 1])
spec = np.fft.fftshift(spec)

w = np.linspace(0, Ntend, Ntend + 1) - Ntend / 2
w = 2.0 * np.pi * w / (Ntend * DT)

fig = plt.figure()
fig.set_figheight(4)
fig.set_figwidth(6)

plt.semilogy(
    w,
    np.abs(spec),
    linewidth=2,
    color='purple'
)

# Expected frequency of magnetic ENERGY:
# omega_E = 2 omega_r = 0.9484 |Omega_ce|
plt.semilogy(
    np.ones(10) * 2.0 * OMEGA_R,
    np.linspace(2e-5, 1e-2, 10),
    'k--',
    linewidth=2,
    label='expected frequency'
)

plt.xlim((0.5, 4.0))

# Expanded vertical scale: preserve the actual spectrum while
# allowing the lower-amplitude part to remain visible.
plt.ylim((1e-6, 1e-2))

plt.ylabel(
    r'$|\mathrm{FFT}(\mathcal{E}_{\tilde{B}_x}/\mathcal{E}(0))|$'
)
plt.xlabel(
    r'$\omega/|\Omega_{\mathrm{ce}}|$'
)
plt.title('Spectrogram in linear phase', fontsize=10)
plt.legend()

fig.savefig(
    OUT/'fig4_2.png',
    dpi=400,
    bbox_inches='tight'
)

fig.savefig(
    OUT/'fig4_2.pdf',
    bbox_inches='tight'
)

plt.close(fig)

# ============================================================
# Figure 4.4
# ============================================================

err = np.abs(en_tot - en_tot[0]) / en_tot[0]

fig = plt.figure()
fig.set_figheight(4)
fig.set_figwidth(6)

plt.semilogy(
    t,
    err,
    linewidth=2,
    color="purple"
)

plt.xlim((0, 200))
plt.ylim((1e-12, 1e-2))

plt.xlabel(r"$t|\Omega_{\mathrm{ce}}|$")
plt.ylabel(
    r"$|\mathcal{E}(t)-\mathcal{E}(0)|/\mathcal{E}(0)$"
)

plt.title(
    "Relative error in total energy",
    fontsize=10
)

fig.savefig(
    OUT/"fig4_4.png",
    dpi=400,
    bbox_inches="tight"
)
fig.savefig(
    OUT/"fig4_4.pdf",
    bbox_inches="tight"
)
plt.close(fig)

print("Created Fig. 4.1, 4.2 and 4.4 from the existing Run-1 energy dataset.")
