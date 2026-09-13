const $ = id => document.getElementById(id);
const worker = new Worker("./worker.js", { type: "module" });
const pending = new Map();
let sequence = 0, catalog, eqdskText = "", busy = false, hasGeometry = false;
const status = (text, error = false) => { $("status").textContent = text; $("status").classList.toggle("error", error); };
function request(action, payload = {}) {
  return new Promise((resolve, reject) => {
    const id = ++sequence; pending.set(id, { resolve, reject }); worker.postMessage({ id, action, ...payload });
  });
}
function setBusy(value) {
  busy = value;
  for (const id of ["construct", "domain", "params", "point", "resolution", "archive", "eqdsk", "clear-flux"]) $(id).disabled = value;
  $("save").disabled = value || !hasGeometry;
}
worker.onmessage = async ({ data }) => {
  if (data.type === "fatal") { status(data.error, true); setBusy(true); return; }
  if (data.type === "ready") {
    catalog = data.catalog;
    for (const name of Object.keys(catalog).sort()) $("domain").add(new Option(name, name));
    $("domain").value = "IGAPolarTorus";
    chooseDomain(); setBusy(false); await construct(); return;
  }
  const promise = pending.get(data.id);
  if (promise) { pending.delete(data.id); data.error ? promise.reject(new Error(data.error)) : promise.resolve(data.result); }
};
worker.onerror = error => { status(error.message, true); setBusy(true); };
function chooseDomain() {
  $("params").value = JSON.stringify(catalog[$("domain").value], null, 2);
  $("flux-controls").hidden = $("domain").value !== "Tokamak";
}
$("domain").addEventListener("change", chooseDomain);
$("eqdsk").addEventListener("change", async event => {
  const file = event.target.files[0];
  eqdskText = file ? await file.text() : "";
  $("flux-note").textContent = file ? `Using ${file.name}` : "Using analytic circular flux surfaces.";
});
$("clear-flux").addEventListener("click", () => { eqdskText = ""; $("eqdsk").value = ""; $("flux-note").textContent = "Using analytic circular flux surfaces."; });
function probeOptions() { return { resolution: Number($("resolution").value), point: $("point").value.split(",").map(Number) }; }
async function construct() {
  if (busy) return;
  setBusy(true); status("Constructing geometry…");
  try {
    const result = await request("render", { name: $("domain").value, params: JSON.parse($("params").value), eqdsk: eqdskText, ...probeOptions() });
    display(result);
  } catch (error) { status(error.message, true); } finally { setBusy(false); }
}
$("controls").addEventListener("submit", event => { event.preventDefault(); construct(); });
$("save").addEventListener("click", async () => {
  setBusy(true);
  try {
    const data = await request("save");
    const bytes = Uint8Array.from(atob(data), c => c.charCodeAt(0));
    const url = URL.createObjectURL(new Blob([bytes], { type: "application/octet-stream" }));
    const link = document.createElement("a"); link.href = url; link.download = "geometry.npz"; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) { status(error.message, true); } finally { setBusy(false); }
});
$("archive").addEventListener("change", async event => {
  const file = event.target.files[0]; if (!file) return;
  setBusy(true); status("Loading geometry…");
  try {
    const bytes = new Uint8Array(await file.arrayBuffer());
    let binary = ""; for (const byte of bytes) binary += String.fromCharCode(byte);
    display(await request("load", { data: btoa(binary), ...probeOptions() }));
  } catch (error) { status(error.message, true); } finally { event.target.value = ""; setBusy(false); }
});
const format = x => Number(x).toPrecision(6);
function display(result) {
  hasGeometry = true; surface = result.xyz; draw();
  $("diagnostics").hidden = false;
  $("coordinates").textContent = "F(η) = [" + result.mapped.map(format).join(", ") + "]";
  for (const key of ["jacobian", "metric"]) $(key).textContent = result[key].map(row => row.map(x => format(x).padStart(12)).join(" ")).join("\n");
  $("determinant").textContent = "det(DF) = " + format(result.determinant);
  status(`${result.name} · surface and probe evaluated in ${result.elapsed.toFixed(3)} s`);
}

// Local canvas rendering; all mapping construction and evaluation runs in Python.
const canvas = $("view"), ctx = canvas.getContext("2d");
let surface, yaw = -.6, pitch = .65, zoom = 1, dragging;
function draw() {
  const ratio = devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * ratio; canvas.height = canvas.clientHeight * ratio;
  ctx.scale(ratio, ratio); const width = canvas.clientWidth, height = canvas.clientHeight;
  ctx.clearRect(0, 0, width, height); if (!surface) return;
  const n = surface[0].length, m = surface[0][0].length;
  const bounds = surface.map(a => { const flat = a.flat(); return [Math.min(...flat), Math.max(...flat)]; });
  const center = bounds.map(b => (b[0] + b[1]) / 2);
  const extent = Math.max(...bounds.map(b => b[1] - b[0]), 1e-8);
  const scale = Math.min(width, height) * .64 * zoom / extent;
  const points = Array.from({ length: n }, (_, i) => Array.from({ length: m }, (_, j) => {
    const [x,y,z] = surface.map((a,k) => a[i][j] - center[k]);
    const u = Math.cos(yaw)*x - Math.sin(yaw)*y, v = Math.sin(yaw)*x + Math.cos(yaw)*y;
    return [width/2+scale*u, height/2-scale*(Math.cos(pitch)*z-Math.sin(pitch)*v), Math.sin(pitch)*z+Math.cos(pitch)*v];
  }));
  const faces = [];
  for (let i=0;i<n-1;i++) for (let j=0;j<m-1;j++) faces.push([points[i][j],points[i+1][j],points[i+1][j+1],points[i][j+1]]);
  faces.sort((a,b) => a.reduce((s,p)=>s+p[2],0)-b.reduce((s,p)=>s+p[2],0));
  for (const face of faces) {
    ctx.beginPath(); face.forEach((p,i) => i ? ctx.lineTo(p[0],p[1]) : ctx.moveTo(p[0],p[1])); ctx.closePath();
    ctx.fillStyle = "rgba(119,180,195,.86)"; ctx.fill(); ctx.strokeStyle = "rgba(25,80,107,.56)"; ctx.lineWidth = .65; ctx.stroke();
  }
}
canvas.addEventListener("pointerdown", event => { dragging = [event.clientX,event.clientY]; canvas.setPointerCapture(event.pointerId); });
canvas.addEventListener("pointermove", event => { if (!dragging) return; yaw += (event.clientX-dragging[0])*.008; pitch += (event.clientY-dragging[1])*.008; dragging=[event.clientX,event.clientY]; draw(); });
for (const event of ["pointerup","pointercancel"]) canvas.addEventListener(event,()=>dragging=null);
canvas.addEventListener("wheel", event => { event.preventDefault(); zoom=Math.max(.3,Math.min(4,zoom*Math.exp(-event.deltaY*.001))); draw(); },{passive:false});
new ResizeObserver(draw).observe(canvas);
