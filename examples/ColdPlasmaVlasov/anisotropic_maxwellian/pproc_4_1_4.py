#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'thesis_run1_standard_cv' / 'run1_standard_cv.npz'
OUT = ROOT

# Run-1 parameters
GAMMA = 0.0447
OMEGA_R = 0.4742
DT = 0.0125
NH = 0.06 * 2.0**2
WPAR = 0.2
WPERP = 0.53

if not DATA.exists():
    raise FileNotFoundError(f'Missing Standard-FEM dataset: {DATA}')

d = np.load(DATA, allow_pickle=True)

t = np.asarray(d['time'], float)
en_E = np.asarray(d['en_E'], float)
en_B = np.asarray(d['en_B'], float)
en_C = np.asarray(d['en_jc'], float)
en_H = np.asarray(d['en_jh'], float)
en_tot = en_B + en_E + en_C + en_H

# Historical plotting style used for the thesis.
fontsize = 22
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': fontsize})
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)

# ============================================================
# Figure 4.1
# ============================================================
tg = np.linspace(0, 150, 100)
colors = ['darkorange', 'purple', 'sienna', 'royalblue']

fig = plt.figure()
fig.set_figheight(4)
fig.set_figwidth(6)

# The thesis source uses 2e-8, not 1e-6, for the analytical growth line.
growth = 2e-8 * np.exp(tg * GAMMA * 2.0)

plt.semilogy(t, en_B/en_tot[0], linewidth=2,
             label=r'$\mathcal{E}_{\tilde{B}}$', color=colors[0])
plt.semilogy(t, en_E/en_tot[0], linewidth=2,
             label=r'$\mathcal{E}_{\tilde{E}}$', color=colors[1])
plt.semilogy(t, en_C/en_tot[0], '--', linewidth=2,
             label=r'$\mathcal{E}_\mathrm{c}$', color=colors[2])
plt.semilogy(t, en_H/en_tot[0], linewidth=2,
             label=r'$\mathcal{E}_\mathrm{h}$', color=colors[3])
plt.semilogy(tg, growth, 'k--', linewidth=2,
             label='expected growth rate')

plt.xlabel('$t|\\Omega_\\mathrm{ce}|$')
plt.ylabel(r'$\mathcal{E} / \mathcal{E}(0)$')
plt.title('Partition of energy', fontsize=fontsize)
plt.ylim((1e-12, 1e1))
plt.xlim((0, 200))
plt.legend(loc='upper center', ncol=1, bbox_to_anchor=(1.55, 1.05))
fig.savefig(OUT/'fig4_1.png', dpi=400, bbox_inches='tight')
fig.savefig(OUT/'fig4_1.pdf', bbox_inches='tight')
plt.close(fig)

# ============================================================
# Figure 4.2
# ============================================================
en_Bx = np.asarray(d['en_Bx'], float)

# Exact FFT construction used in the historical thesis plotting notebook.
Ntend = 8000
spec = np.fft.fft(en_Bx[0:Ntend + 1])
spec = np.fft.fftshift(spec)
w = np.linspace(0, Ntend, Ntend + 1) - Ntend/2
w = 2*np.pi*w/(Ntend*DT)

fig = plt.figure()
fig.set_figheight(4)
fig.set_figwidth(6)

plt.semilogy(w, np.abs(spec), linewidth=2, color='purple')
plt.semilogy(np.ones(10)*OMEGA_R*2,
             np.linspace(2e-5, 1e-2, 10),
             'k--', linewidth=2, label='expected frequency')

# These are the limits used for the thesis-rendered figure.
plt.xlim((0.5, 4))
plt.ylim((1e-4, 1e-2))
plt.ylabel(r'$|\mathrm{FFT}(\mathcal{E}_{\tilde{B}_x} / \mathcal{E}(0))|$')
plt.xlabel(r'$\omega_{\mathrm{r}} / |\Omega_\mathrm{ce}|$')
plt.title('Spectrogram in linear phase', fontsize=fontsize)
plt.legend()

fig.savefig(OUT/'fig4_2.png', dpi=400, bbox_inches='tight')
fig.savefig(OUT/'fig4_2.pdf', bbox_inches='tight')
plt.close(fig)

# ============================================================
# Figure 4.3
# ============================================================
p0 = np.asarray(d['particles_initial'], float)
pT = np.asarray(d['particles_final'], float)

# Standard runner stores [z, vx, vy, vz, w].
vx0, vz0 = p0[:, 1], p0[:, 3]
vxT, vzT = pT[:, 1], pT[:, 3]

# The initial particle weights are stored with the 1/Np factor.
# During the time integration, the historical update stores wnew
# without that factor. Restore the Monte-Carlo normalization here.
NPART = 100000
wT = pT[:, 4] / NPART

def maxwell_parallel(v):
    return NH/(np.sqrt(2*np.pi)*WPAR) * np.exp(-v**2/(2*WPAR**2))

def maxwell_perp_folded(v):
    return 2*NH/(np.sqrt(2*np.pi)*WPERP) * np.exp(-v**2/(2*WPERP**2))

def weighted_residual(x, weights, edges):
    hist, _ = np.histogram(x, bins=edges, weights=weights)
    widths = np.diff(edges)
    centers = 0.5*(edges[:-1] + edges[1:])
    return centers, hist/widths

# Use the same velocity ranges as the thesis.  The particle data are
# retained as-is; only the post-processing estimator is changed.
edges_par = np.linspace(-1.0, 1.0, 129)
edges_perp = np.linspace(0.0, 2.0, 129)

vk = 0.5*(edges_par[:-1] + edges_par[1:])
vp = 0.5*(edges_perp[:-1] + edges_perp[1:])

# Initial distribution is the exact analytical bi-Maxwellian marginal.
fpar0 = maxwell_parallel(vk)
fperp0 = maxwell_perp_folded(vp)

_, rpar = weighted_residual(vzT, wT, edges_par)
_, rperp = weighted_residual(np.abs(vxT), wT, edges_perp)

fparT = fpar0 + rpar
fperpT = fperp0 + rperp

# Historical figure convention: initial minus final.
dpar = fpar0 - fparT
dperp = fperp0 - fperpT

vres = 0.26

# Figure 4.3 is a four-panel thesis figure; use a compact local font
# rather than inheriting the 22-point font used by the single-panel
# thesis figures 4.1, 4.2 and 4.4.
fig, axs = plt.subplots(2, 2, figsize=(6, 4.8))
panel_fs = 8
tick_fs = 8
label_fs = 9

axs[0, 0].plot(vk, fpar0, color='darkorange', linewidth=1,
               label=r'$t=0$')
axs[0, 0].plot(vk, fparT, color='purple', linewidth=1,
               label=r'$t=200\ |\Omega_{\mathrm{ce}}|$')
axs[0, 0].axvline(-vres, color='black', linestyle='--', linewidth=1,
                  label=r'$v_R$')
axs[0, 0].axvline(vres, color='black', linestyle='--', linewidth=1)
axs[0, 0].set_xlim(-1, 1)
axs[0, 0].set_ylim(0, 0.4)
axs[0, 0].set_xlabel(r'$v_k/c$', fontsize=label_fs)
axs[0, 0].set_ylabel(r'$f_h(v_k)c^3/|\Omega_{ce}|$', fontsize=label_fs)
axs[0, 0].set_title('Initial and final parallel distribution',
                    fontsize=panel_fs)
axs[0, 0].tick_params(labelsize=tick_fs)
axs[0, 0].legend(fontsize=6)

axs[0, 1].plot(vp, fperp0, color='darkorange', linewidth=1,
               label=r'$t=0$')
axs[0, 1].plot(vp, fperpT, color='purple', linewidth=1,
               label=r'$t=200\ |\Omega_{\mathrm{ce}}|$')
axs[0, 1].set_xlim(0, 2)
axs[0, 1].set_ylim(0, 0.2)
axs[0, 1].set_xlabel(r'$v_\perp/c$', fontsize=label_fs)
axs[0, 1].set_ylabel(r'$f_h(v_\perp)c^2/|\Omega_{ce}|$', fontsize=label_fs)
axs[0, 1].set_title('Initial and final perp. distribution',
                    fontsize=panel_fs)
axs[0, 1].tick_params(labelsize=tick_fs)
axs[0, 1].legend(fontsize=6)

axs[1, 0].plot(vk, dpar, color='purple', linewidth=1)
axs[1, 0].axhline(0, color='black', linewidth=1)
axs[1, 0].axvline(-vres, color='black', linestyle='--', linewidth=1,
                  label=r'$v_R$')
axs[1, 0].axvline(vres, color='black', linestyle='--', linewidth=1)
axs[1, 0].set_xlim(-1, 1)
axs[1, 0].set_ylim(-0.05, 0.05)
axs[1, 0].set_xlabel(r'$v_k/c$', fontsize=label_fs)
axs[1, 0].set_ylabel(r'$\delta f_h(v_k)c^3/|\Omega_{ce}|$', fontsize=label_fs)
axs[1, 0].set_title('Difference parallel', fontsize=panel_fs)
axs[1, 0].tick_params(labelsize=tick_fs)
axs[1, 0].legend(fontsize=6)

axs[1, 1].plot(vp, dperp, color='purple', linewidth=1)
axs[1, 1].axhline(0, color='black', linewidth=1)
axs[1, 1].set_xlim(0, 2)
axs[1, 1].set_ylim(-0.005, 0.005)
axs[1, 1].set_xlabel(r'$v_\perp/c$', fontsize=label_fs)
axs[1, 1].set_ylabel(r'$\delta f_h(v_\perp)c^2/|\Omega_{ce}|$', fontsize=label_fs)
axs[1, 1].set_title('Difference perpendicular', fontsize=panel_fs)
axs[1, 1].tick_params(labelsize=tick_fs)

fig.tight_layout()
fig.savefig(OUT/'fig4_3.png', dpi=400, bbox_inches='tight')
fig.savefig(OUT/'fig4_3.pdf', bbox_inches='tight')
plt.close(fig)

# ============================================================
# Figure 4.4
# ============================================================
fig = plt.figure()
fig.set_figheight(4)
fig.set_figwidth(6)

plt.semilogy(t, np.abs(en_tot - en_tot[0])/en_tot[0],
             linewidth=2, color='purple')
plt.ylim((1e-10, 1e-2))
plt.xlim((0, 200))
plt.xlabel('$t|\\Omega_\\mathrm{ce}|$')
plt.ylabel('$|\\mathcal{E}(t) - \\mathcal{E}(0)|/\\mathcal{E}(0)$')
plt.title('Relative error in total energy', fontsize=fontsize)

fig.savefig(OUT/'fig4_4.png', dpi=400, bbox_inches='tight')
fig.savefig(OUT/'fig4_4.pdf', bbox_inches='tight')
plt.close(fig)

print('Generated: fig4_1.png, fig4_2.png, fig4_3.png, fig4_4.png')
