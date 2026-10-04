from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext

ROOT = Path(__file__).resolve().parent

K_VALUES = np.arange(3.00, 1.49, -0.25)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

# ------------------------------------------------------------
# Explicit physical geometry.
# Do NOT use bbox_inches="tight".
# ------------------------------------------------------------
fig = plt.figure(figsize=(6.0, 2.05))

# [left, bottom, width, height]
# Deliberately wide/short, matching the thesis panel.
ax = fig.add_axes([0.105, 0.235, 0.605, 0.615])

for k in K_VALUES:

    tag = f"{k:.2f}".replace(".", "p")
    path = ROOT / f"thesis_run2_k{tag}" / "run3_fig4_7.npz"

    d = np.load(path)

    t = d["time"]
    en_B = d["en_B"]

    E0 = (
        d["en_E"][0]
        + d["en_B"][0]
        + d["en_jc"][0]
        + d["en_jh"][0]
    )

    EB_norm = en_B / E0

    mask = np.isfinite(EB_norm) & (EB_norm > 0)

    ax.plot(
        t[mask],
        EB_norm[mask],
        linewidth=0.8,
        label=fr"$k={k:.2f}\ |\Omega_{{ce}}|/c$"
    )

# ------------------------------------------------------------
# EXACT THESIS AXIS SCALE
# ------------------------------------------------------------

ax.set_xlim(0, 400)
ax.set_ylim(1e-12, 1e-0)

ax.set_xticks([0, 100, 200, 300, 400])

ax.set_yscale("log")
ax.set_yticks([1e-12, 1e-9, 1e-6, 1e-3, 1e-0])
ax.yaxis.set_major_formatter(LogFormatterMathtext())

# No minor tick labels
ax.tick_params(
    axis="both",
    which="minor",
    left=False,
    right=False,
    bottom=False,
    top=False
)

ax.set_xlabel(r"$t|\Omega_{ce}|$")
ax.set_ylabel(r"$E_{\tilde{B}}/E(0)$")

ax.set_title("Wavenumber scan", pad=3)

ax.grid(False)

# ------------------------------------------------------------
# THESIS-STYLE LEGEND
# ------------------------------------------------------------

ax.legend(
    loc="center left",
    bbox_to_anchor=(1.015, 0.50),
    frameon=True,
    framealpha=1.0,
    edgecolor="0.65",
    borderpad=0.30,
    labelspacing=0.22,
    handlelength=1.8,
    handletextpad=0.35,
)

# ------------------------------------------------------------
# SAVE — preserve exact physical geometry
# ------------------------------------------------------------

png = ROOT / "fig4_5_reproduction.png"
pdf = ROOT / "fig4_5_reproduction.pdf"

fig.savefig(png, dpi=300)
fig.savefig(pdf)

plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
