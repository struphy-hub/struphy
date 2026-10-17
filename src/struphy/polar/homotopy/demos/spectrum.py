"""Spectrum explorer of the 2D curl-curl operator on the unit disk (the interactive slide of the talk).

For the polar 1-form spaces with Dirichlet conditions of regularity ``r = 0, ..., 5`` and the enrichments
``"cochain_chain"`` (the paper's spaces) and ``"cochain"`` (the reduced spaces), the eigenproblem
``(curl A, curl B) = lambda (A, B)`` is solved on a 2D Struphy Derham complex (unit disk ``HollowCylinder(a1=0)``,
B-splines of degree 6 on 10 x 14 cells, as the talk). For every eigenpair the explorer shows

* the discrete eigenvalue ``lambda_h`` and the exact one ``(j'_{m,k})^2`` (squared zeros of ``J_m'``),
* the relative eigenvalue error and the *regularity deficit* ``||x - Pi^6 x|| / ||x||`` of the eigenfunction ``x``
  (``Pi^6`` the L2 projection onto the polar 1-forms of regularity 6, L2 norms),
* images of ``A_x``, ``A_y`` and ``curl A`` on the disk and around the centre.

``write_explorer(folder)`` writes ``spectrum_explorer.html`` with ``spectrum_data.js`` and the images into ``folder``;
open the HTML file in a browser.
"""

import json
from pathlib import Path

import numpy as np
import scipy.linalg as sla

from struphy.polar.homotopy.demos import common
from struphy.polar.homotopy.demos.maxwell import exact_spectrum
from struphy.polar.homotopy.demos.plotting import DIVERGING

DEGREE = 6
CELLS = (10, 14)
REGULARITIES = (0, 1, 2, 3, 4, 5)
REFERENCE_REGULARITY = 6
ENRICHMENTS = ("cochain_chain", "cochain")
ZOOM = 0.25
FIELDS = ("A_x", "A_y", "curl A")


def complex_(regularity, enrichment="cochain_chain", degree=DEGREE, cells=CELLS):
    return common.polar_complex(
        regularity,
        degree,
        cells,
        domain=common.unit_disk(),
        boundary="dirichlet",
        enrichment=enrichment,
        check_assumptions=False,
    )


def series(regularity, enrichment="cochain_chain", reference=None, count=None, degree=DEGREE, cells=CELLS):
    """Nonzero eigenpairs of ``C^T M_2 C a = lambda M_1 a`` (ascending), with the matched exact eigenvalues and the
    regularity deficits; the eigenvectors are returned as full tensor-product coefficients."""
    c = complex_(regularity, enrichment, degree, cells)
    C = c.derivative(1)
    K = (C.T @ c.mass_matrix(2) @ C).toarray()
    lam, a = sla.eigh((K + K.T) / 2, c.mass_matrix(1).toarray())
    kernel = int(np.sum(lam < 1e-8 * lam.max()))
    lam, a = lam[kernel:], a[:, kernel:]
    x = c.extraction(1) @ a
    count = len(lam) if count is None else min(count, len(lam))
    exact = exact_spectrum(lambda_max=1.5 * lam[count - 1])
    deficit = np.full(count, np.nan)
    if reference is not None:
        # L2 projection onto the polar space of the reference regularity, relative L2 norm (as the talk)
        import scipy.sparse.linalg as spla

        M = c.full_mass_matrix(1)
        E = reference.extraction(1)
        xs = x[:, :count]
        projected = E @ spla.splu((E.T @ M @ E).tocsc()).solve(E.T @ (M @ xs))
        difference = xs - projected
        deficit = np.sqrt(np.einsum("ij,ij->j", difference, M @ difference) / np.einsum("ij,ij->j", xs, M @ xs))
    return dict(
        complex=c,
        regularity=regularity,
        enrichment=enrichment,
        dimension=c.dimension(1),
        kernel=kernel,
        lam=lam[:count],
        vectors=x[:, :count],
        exact=[e[0] if n < len(exact) else None for n, e in enumerate(exact[:count])] + [None] * (count - len(exact)),
        m=[e[1] for e in exact[:count]] + [None] * (count - len(exact)),
        k=[e[2] for e in exact[:count]] + [None] * (count - len(exact)),
        deficit=deficit,
    )


# ---------------------------------------------------------------------------------------------------------- images
class ModeImages:
    """Rasters of ``A_x``, ``A_y`` (physical components) and ``curl A`` (density) of 1-form coefficients on the full
    disk and around the centre, by precomputed evaluation matrices."""

    def __init__(self, complex_, pixels=96):
        self.pixels = pixels
        mapping = complex_.spaces[1].mapping
        logical1, logical2 = complex_.spaces[1].logical, complex_.spaces[2].logical
        self.views = []
        for extent in (1.0, ZOOM):
            t = (np.arange(pixels) + 0.5) / pixels * 2 - 1
            X, Y = np.meshgrid(extent * t, -extent * t, indexing="xy")
            inside = np.hypot(X, Y) < 1.0
            x, y = mapping.inverse(X[inside], Y[inside])
            J = mapping.jacobian_matrix(np.maximum(x, 1e-12), y)
            det = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
            G1 = [self._evaluation(logical1, c, x, y) for c in range(2)]
            G2 = self._evaluation(logical2, 0, x, y)
            self.views.append(dict(inside=inside, J=J, det=det, G1=G1, G2=G2))
        self.curl = complex_.spaces[1].derivative_matrix

    @staticmethod
    def _evaluation(logical, component, x, y):
        c = logical.components[component]
        Bx, By = c.fx.basis_matrix(x).toarray(), c.fy.basis_matrix(y).toarray()
        G = (Bx[:, :, None] * By[:, None, :]).reshape(len(x), -1)  # Struphy ordering i * ny + j
        offset = logical.offsets[component]
        full = np.zeros((len(x), logical.n))
        full[:, offset : offset + G.shape[1]] = G
        return full

    def fields(self, coefficients):
        """Per view, the three fields on the pixels (NaN outside the disk)."""
        curl = self.curl @ coefficients
        out = []
        for v in self.views:
            ax, ay = v["G1"][0] @ coefficients, v["G1"][1] @ coefficients
            J, det = v["J"], v["det"]
            # covariant pushforward J^{-T} (a_x, a_y)
            px = (J[..., 1, 1] * ax - J[..., 1, 0] * ay) / det
            py = (-J[..., 0, 1] * ax + J[..., 0, 0] * ay) / det
            b = (v["G2"] @ curl) / det
            images = []
            for values in (px, py, b):
                image = np.full(v["inside"].shape, np.nan)
                image[v["inside"]] = values
                images.append(image)
            out.append(images)
        return out

    def sprite(self, coefficients):
        """RGB sprite: columns ``A_x, A_y, curl A``, rows full disk and centre zoom (own symmetric scale per tile)."""
        tiles = self.fields(coefficients)
        p = self.pixels
        sprite = np.full((2 * p, 3 * p, 3), 255, dtype=np.uint8)
        for row, images in enumerate(tiles):
            for col, image in enumerate(images):
                limit = np.nanmax(np.abs(image)) or 1.0
                rgba = DIVERGING(0.5 + 0.5 * np.nan_to_num(image / limit))
                rgb = (255 * rgba[..., :3]).astype(np.uint8)
                rgb[np.isnan(image)] = 255
                sprite[row * p : (row + 1) * p, col * p : (col + 1) * p] = rgb
        return sprite


# ---------------------------------------------------------------------------------------------------------- output
def compute(count=None, regularities=REGULARITIES, enrichments=ENRICHMENTS, degree=DEGREE, cells=CELLS, verbose=True):
    """The spectra of all spaces (``count`` eigenpairs each, default: all)."""
    data = {}
    for enrichment in enrichments:
        reference = complex_(REFERENCE_REGULARITY, "cochain_chain", degree, cells)
        data[enrichment] = []
        for r in regularities:
            s = series(r, enrichment, reference, count, degree, cells)
            data[enrichment].append(s)
            if verbose:
                lam_exact = np.array([e if e is not None else np.nan for e in s["exact"]])
                error = np.abs(s["lam"] - lam_exact) / lam_exact
                print(
                    f"{enrichment:13s} r = {r}: dim V1 = {s['dimension']}, kernel {s['kernel']}, "
                    f"lambda_1 error {error[0]:.1e}, median deficit {np.nanmedian(s['deficit']):.1e}"
                )
    return data


def write_explorer(folder, count=120, pixels=96, regularities=REGULARITIES, degree=DEGREE, cells=CELLS, verbose=True):
    """Compute the spectra and write the explorer (HTML, data and images) into ``folder``."""
    from PIL import Image

    folder = Path(folder)
    images = folder / "spectrum"
    images.mkdir(parents=True, exist_ok=True)
    data = compute(count, regularities, degree=degree, cells=cells, verbose=verbose)
    spaces = []
    for enrichment, list_ in data.items():
        series_out = []
        for s in list_:
            raster = ModeImages(s["complex"], pixels)
            names = []
            for n in range(len(s["lam"])):
                name = f"{enrichment}_r{s['regularity']}_{n + 1:03d}.png"
                Image.fromarray(raster.sprite(s["vectors"][:, n])).convert("P", palette=Image.ADAPTIVE).save(
                    images / name, optimize=True
                )
                names.append(name)
            series_out.append(
                dict(
                    r=s["regularity"],
                    dimension=s["dimension"],
                    **{"lambda": [float(v) for v in s["lam"]]},
                    exact=[None if e is None else float(e) for e in s["exact"]],
                    m=s["m"],
                    k=s["k"],
                    deficit=[None if not np.isfinite(d) else float(d) for d in s["deficit"]],
                    image=names,
                )
            )
        spaces.append(dict(enrichment=enrichment, series=series_out))
    meta = dict(
        degree=degree,
        radialCells=cells[0],
        angularCells=cells[1],
        boundary="dirichlet",
        referenceRegularity=REFERENCE_REGULARITY,
        zoom=ZOOM,
        tile=pixels,
        fields=list(FIELDS),
        domain="HollowCylinder(a1=0, a2=1) (unit disk)",
    )
    exact = [dict(k=k, m=m, **{"lambda": float(lam)}) for lam, m, k in exact_spectrum(lambda_max=2000.0)]
    payload = json.dumps(dict(meta=meta, exact=exact, spaces=spaces), separators=(",", ":"))
    (folder / "spectrum_data.js").write_text(
        "// Generated by struphy.polar.homotopy.demos.spectrum\nwindow.SPECTRUM = " + payload + ";\n"
    )
    (folder / "spectrum_explorer.html").write_text(_HTML)
    if verbose:
        print(f"wrote {folder / 'spectrum_explorer.html'}")
    return folder / "spectrum_explorer.html"


_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Curl-curl spectrum explorer</title>
<style>
  :root {
    --ink: #1d2327; --ink-2: #4f5a60; --rule: #e3e7e8; --panel: #f7f9f9; --bg: #ffffff; --accent: #006c66;
    --r0: #0072B2; --r1: #E69F00; --r2: #009E73; --r3: #CC79A7; --r4: #56B4E9; --r5: #D55E00;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) { --ink: #e8ecee; --ink-2: #a9b4b9; --rule: #33393c; --panel: #1d2225; --bg: #14181a; }
  }
  :root[data-theme="dark"] { --ink: #e8ecee; --ink-2: #a9b4b9; --rule: #33393c; --panel: #1d2225; --bg: #14181a; }
  body { margin: 0; font: 14px/1.4 system-ui, -apple-system, sans-serif; color: var(--ink); background: var(--bg); }
  header { padding: 12px 16px; border-bottom: 1px solid var(--rule); }
  h1 { font-size: 18px; margin: 0 0 4px; }
  .sub { color: var(--ink-2); font-size: 13px; }
  main { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); gap: 16px; padding: 12px 16px; }
  @media (max-width: 900px) { main { grid-template-columns: minmax(0, 1fr); } }
  .controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 8px; }
  button, select { font: inherit; color: var(--ink); background: var(--panel); border: 1px solid var(--rule);
    border-radius: 6px; padding: 4px 9px; cursor: pointer; }
  button.on { border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }
  .chip { display: inline-flex; align-items: center; gap: 6px; }
  .swatch { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
  #plot { width: 100%; height: 440px; background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; display: block; }
  .panel { background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; padding: 10px; }
  .view { position: relative; width: 100%; max-width: 420px; aspect-ratio: 1; margin: 6px auto; }
  .view .img { position: absolute; inset: 0; background-repeat: no-repeat; image-rendering: auto; border-radius: 4px; }
  .view svg { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  td { padding: 2px 4px; border-bottom: 1px solid var(--rule); }
  td:first-child { color: var(--ink-2); }
  .strip { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; }
  .strip div { width: 64px; text-align: center; font-size: 11px; color: var(--ink-2); cursor: pointer; }
  .strip .thumb { width: 64px; height: 64px; background-repeat: no-repeat; border-radius: 4px; border: 1px solid var(--rule); }
  .keys { color: var(--ink-2); font-size: 12px; margin-top: 6px; }
  #tip { position: fixed; pointer-events: none; background: var(--bg); border: 1px solid var(--rule); border-radius: 6px;
    padding: 4px 7px; font-size: 12px; display: none; }
</style>
</head>
<body>
<header>
  <h1>Spectrum of the 2D curl&ndash;curl operator on the unit disk</h1>
  <div class="sub" id="subtitle"></div>
</header>
<main>
  <section>
    <div class="controls">
      <select id="space" aria-label="polar 1-form space"></select>
      <span id="chips"></span>
    </div>
    <div class="controls">
      <span class="sub">vertical axis</span>
      <button id="y-deficit" class="on">regularity deficit</button>
      <button id="y-error">eigenvalue error</button>
      <button id="reset">reset view</button>
    </div>
    <canvas id="plot" aria-label="eigenvalues against regularity deficit or error"></canvas>
    <div class="keys">&larr; &rarr; eigenpair &middot; &uarr; &darr; regularity &middot; f field &middot; z centre &middot; g mesh &middot; hover to preview, click to pin &middot; drag to zoom</div>
  </section>
  <section class="panel">
    <div class="controls">
      <button data-field="2" class="field on">curl A</button>
      <button data-field="0" class="field">A, x-component</button>
      <button data-field="1" class="field">A, y-component</button>
    </div>
    <div class="controls">
      <button id="view-full" class="on">full disk</button>
      <button id="view-zoom">centre zoom</button>
      <button id="mesh" class="on">mesh</button>
    </div>
    <div class="view"><div class="img" id="image"></div><svg id="overlay" viewBox="-1 -1 2 2"></svg></div>
    <table id="info"></table>
    <div class="sub" style="margin-top:8px">same eigenpair index, all regularities</div>
    <div class="strip" id="strip"></div>
  </section>
</main>
<div id="tip"></div>
<script src="spectrum_data.js"></script>
<script>
(() => {
  const D = window.SPECTRUM, META = D.meta, IMG = 'spectrum/';
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const COLORS = ['--r0', '--r1', '--r2', '--r3', '--r4', '--r5'].map(css);
  const state = { space: 0, r: 0, n: 0, hover: null, field: 2, view: 0, mesh: true, ymode: 'deficit',
                  visible: D.spaces[0].series.map(() => true), xr: null, yr: null };
  const sub = document.getElementById('subtitle');
  sub.textContent = `polar 1-forms with Dirichlet conditions, B-splines of degree ${META.degree} on ` +
    `${META.radialCells} x ${META.angularCells} cells, ${META.domain}; deficit: ||x - Pi^${META.referenceRegularity} x|| / ||x||`;
  const spaceSel = document.getElementById('space');
  D.spaces.forEach((s, i) => spaceSel.add(new Option(s.enrichment === 'cochain_chain' ?
    'cochain-chain (enriched polar spaces of the paper)' : 'cochain (reduced spaces)', i)));
  const series = () => D.spaces[state.space].series;
  const relerr = (s, n) => s.exact[n] == null ? null : Math.abs(s.lambda[n] - s.exact[n]) / s.exact[n];
  const yval = (s, n) => state.ymode === 'deficit' ? s.deficit[n] : relerr(s, n);
  const FLOOR = 1e-16;
  // ---------------------------------------------------------------- chips
  function chips() {
    const el = document.getElementById('chips'); el.innerHTML = '';
    series().forEach((s, i) => {
      const b = document.createElement('button');
      b.className = 'chip' + (state.visible[i] ? ' on' : '');
      b.innerHTML = `<span class="swatch" style="background:${COLORS[i]}"></span>r = ${s.r}`;
      b.onclick = () => { state.visible[i] = !state.visible[i]; render(); };
      el.appendChild(b);
    });
  }
  // ---------------------------------------------------------------- plot
  const canvas = document.getElementById('plot'), ctx = canvas.getContext('2d');
  let points = [], drag = null;
  function ranges() {
    let xs = [], ys = [];
    series().forEach((s, i) => { if (!state.visible[i]) return;
      s.lambda.forEach((l, n) => { const y = yval(s, n); if (y != null) { xs.push(Math.log10(l)); ys.push(Math.log10(Math.max(y, FLOOR))); } }); });
    const pad = (a, b) => [a - 0.05 * (b - a || 1), b + 0.05 * (b - a || 1)];
    return { x: state.xr ?? pad(Math.min(...xs), Math.max(...xs)), y: state.yr ?? pad(Math.min(...ys), Math.max(...ys)) };
  }
  function plot() {
    const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const R = ranges(), L = 56, B = 34, T = 10, Rm = 10;
    const X = v => L + (v - R.x[0]) / (R.x[1] - R.x[0]) * (w - L - Rm), Y = v => h - B - (v - R.y[0]) / (R.y[1] - R.y[0]) * (h - B - T);
    ctx.strokeStyle = css('--rule'); ctx.fillStyle = css('--ink-2'); ctx.font = '11px system-ui'; ctx.lineWidth = 1;
    for (let e = Math.ceil(R.x[0]); e <= R.x[1]; e++) { ctx.beginPath(); ctx.moveTo(X(e), T); ctx.lineTo(X(e), h - B); ctx.stroke(); ctx.fillText('1e' + e, X(e) - 10, h - B + 14); }
    for (let e = Math.ceil(R.y[0]); e <= R.y[1]; e++) { ctx.beginPath(); ctx.moveTo(L, Y(e)); ctx.lineTo(w - Rm, Y(e)); ctx.stroke(); ctx.fillText('1e' + e, 6, Y(e) + 4); }
    ctx.fillText('eigenvalue', w / 2 - 20, h - 6);
    ctx.save(); ctx.translate(12, h / 2 + 40); ctx.rotate(-Math.PI / 2);
    ctx.fillText(state.ymode === 'deficit' ? 'regularity deficit' : 'relative eigenvalue error', 0, 0); ctx.restore();
    points = [];
    series().forEach((s, i) => { if (!state.visible[i]) return;
      ctx.fillStyle = COLORS[i];
      s.lambda.forEach((l, n) => { const y = yval(s, n); if (y == null) return;
        const px = X(Math.log10(l)), py = Y(Math.log10(Math.max(y, FLOOR)));
        points.push({ px, py, i, n });
        ctx.beginPath(); ctx.arc(px, py, 2.6, 0, 2 * Math.PI); ctx.fill(); }); });
    const cur = state.hover ?? { i: state.r, n: state.n };
    const s = series()[cur.i], y = s && yval(s, cur.n);
    if (s && y != null) { ctx.strokeStyle = css('--ink'); ctx.lineWidth = 2; ctx.beginPath();
      ctx.arc(X(Math.log10(s.lambda[cur.n])), Y(Math.log10(Math.max(y, FLOOR))), 6, 0, 2 * Math.PI); ctx.stroke(); }
    canvas._map = { X, Y, R, L, B, T, Rm, w, h };
  }
  function nearest(ev) {
    const rect = canvas.getBoundingClientRect(), x = ev.clientX - rect.left, y = ev.clientY - rect.top;
    let best = null, d = 64;
    points.forEach(p => { const q = (p.px - x) ** 2 + (p.py - y) ** 2; if (q < d) { d = q; best = p; } });
    return best;
  }
  const tip = document.getElementById('tip');
  canvas.addEventListener('mousemove', ev => {
    if (drag) return;
    const p = nearest(ev);
    state.hover = p ? { i: p.i, n: p.n } : null;
    if (p) { const s = series()[p.i]; tip.style.display = 'block'; tip.style.left = ev.clientX + 12 + 'px'; tip.style.top = ev.clientY + 12 + 'px';
      tip.textContent = `r = ${s.r}, #${p.n + 1}: lambda_h = ${s.lambda[p.n].toPrecision(8)}`; } else tip.style.display = 'none';
    plot(); panel();
  });
  canvas.addEventListener('mouseleave', () => { state.hover = null; tip.style.display = 'none'; plot(); panel(); });
  canvas.addEventListener('mousedown', ev => { drag = { x: ev.offsetX, y: ev.offsetY }; });
  canvas.addEventListener('mouseup', ev => {
    const m = canvas._map;
    if (drag && (Math.abs(ev.offsetX - drag.x) > 8 || Math.abs(ev.offsetY - drag.y) > 8)) {
      const ix = v => m.R.x[0] + (v - m.L) / (m.w - m.L - m.Rm) * (m.R.x[1] - m.R.x[0]);
      const iy = v => m.R.y[0] + (m.h - m.B - v) / (m.h - m.B - m.T) * (m.R.y[1] - m.R.y[0]);
      state.xr = [ix(Math.min(drag.x, ev.offsetX)), ix(Math.max(drag.x, ev.offsetX))];
      state.yr = [iy(Math.max(drag.y, ev.offsetY)), iy(Math.min(drag.y, ev.offsetY))];
    } else { const p = nearest(ev); if (p) { state.r = p.i; state.n = p.n; } }
    drag = null; render();
  });
  // ---------------------------------------------------------------- panel
  function tileStyle(el, s, n, field, view, size) {
    el.style.backgroundImage = `url(${IMG}${s.image[n]})`;
    el.style.backgroundSize = `${3 * size}px ${2 * size}px`;
    el.style.backgroundPosition = `${-field * size}px ${-view * size}px`;
  }
  function overlay() {
    const svg = document.getElementById('overlay'); svg.innerHTML = '';
    if (!state.mesh) return;
    const scale = state.view === 0 ? 1 : 1 / META.zoom, ns = 'http://www.w3.org/2000/svg', stroke = 'rgba(0,0,0,0.25)';
    for (let i = 1; i <= META.radialCells; i++) { const c = document.createElementNS(ns, 'circle');
      c.setAttribute('r', i / META.radialCells * scale); c.setAttribute('fill', 'none'); c.setAttribute('stroke', stroke);
      c.setAttribute('stroke-width', 0.006); svg.appendChild(c); }
    for (let j = 0; j < META.angularCells; j++) { const a = 2 * Math.PI * j / META.angularCells, l = document.createElementNS(ns, 'line');
      l.setAttribute('x1', 0); l.setAttribute('y1', 0); l.setAttribute('x2', Math.cos(a) * scale); l.setAttribute('y2', -Math.sin(a) * scale);
      l.setAttribute('stroke', stroke); l.setAttribute('stroke-width', 0.006); svg.appendChild(l); }
  }
  function panel() {
    const cur = state.hover ?? { i: state.r, n: state.n }, s = series()[cur.i], n = cur.n;
    const img = document.getElementById('image'), size = img.clientWidth || 380;
    tileStyle(img, s, n, state.field, state.view, size);
    overlay();
    const e = relerr(s, n), fmt = v => v == null ? '–' : v.toExponential(2);
    document.getElementById('info').innerHTML =
      `<tr><td>space</td><td>${D.spaces[state.space].enrichment}, r = ${s.r} (dim ${s.dimension})</td></tr>` +
      `<tr><td>eigenpair</td><td>#${n + 1}${s.m[n] != null ? `, mode (m, k) = (${s.m[n]}, ${s.k[n]})` : ''}</td></tr>` +
      `<tr><td>discrete eigenvalue</td><td>${s.lambda[n].toPrecision(10)}</td></tr>` +
      `<tr><td>exact eigenvalue</td><td>${s.exact[n] == null ? '–' : s.exact[n].toPrecision(10)}</td></tr>` +
      `<tr><td>relative error</td><td>${fmt(e)}</td></tr>` +
      `<tr><td>regularity deficit</td><td>${fmt(s.deficit[n])}</td></tr>`;
    const strip = document.getElementById('strip'); strip.innerHTML = '';
    series().forEach((t, i) => { if (n >= t.lambda.length) return;
      const d = document.createElement('div'), th = document.createElement('div'); th.className = 'thumb';
      tileStyle(th, t, n, state.field, state.view, 64); d.appendChild(th);
      d.appendChild(document.createTextNode(`r = ${t.r}`)); d.style.color = COLORS[i];
      d.onclick = () => { state.r = i; render(); }; strip.appendChild(d); });
  }
  function render() { chips(); plot(); panel(); }
  // ---------------------------------------------------------------- controls
  spaceSel.onchange = () => { state.space = +spaceSel.value; state.xr = state.yr = null; render(); };
  const setY = m => { state.ymode = m; state.yr = null;
    document.getElementById('y-deficit').classList.toggle('on', m === 'deficit');
    document.getElementById('y-error').classList.toggle('on', m === 'error'); render(); };
  document.getElementById('y-deficit').onclick = () => setY('deficit');
  document.getElementById('y-error').onclick = () => setY('error');
  document.getElementById('reset').onclick = () => { state.xr = state.yr = null; render(); };
  document.querySelectorAll('.field').forEach(b => b.onclick = () => { state.field = +b.dataset.field;
    document.querySelectorAll('.field').forEach(c => c.classList.toggle('on', c === b)); render(); });
  const setView = v => { state.view = v; document.getElementById('view-full').classList.toggle('on', v === 0);
    document.getElementById('view-zoom').classList.toggle('on', v === 1); render(); };
  document.getElementById('view-full').onclick = () => setView(0);
  document.getElementById('view-zoom').onclick = () => setView(1);
  document.getElementById('mesh').onclick = () => { state.mesh = !state.mesh;
    document.getElementById('mesh').classList.toggle('on', state.mesh); render(); };
  document.addEventListener('keydown', ev => {
    const s = series()[state.r];
    if (ev.key === 'ArrowRight') state.n = Math.min(state.n + 1, s.lambda.length - 1);
    else if (ev.key === 'ArrowLeft') state.n = Math.max(state.n - 1, 0);
    else if (ev.key === 'ArrowUp') state.r = Math.min(state.r + 1, series().length - 1);
    else if (ev.key === 'ArrowDown') state.r = Math.max(state.r - 1, 0);
    else if (ev.key === 'f') { state.field = (state.field + 1) % 3;
      document.querySelectorAll('.field').forEach(c => c.classList.toggle('on', +c.dataset.field === state.field)); }
    else if (ev.key === 'z') { setView(1 - state.view); return; }
    else if (ev.key === 'g') { state.mesh = !state.mesh; document.getElementById('mesh').classList.toggle('on', state.mesh); }
    else return;
    ev.preventDefault(); state.hover = null; render();
  });
  window.addEventListener('resize', render);
  render();
})();
</script>
</body>
</html>
"""
