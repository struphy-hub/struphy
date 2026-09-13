import { loadPyodide } from "./vendor/pyodide.mjs";

let pyodide;
const ready = (async () => {
  const config = await (await fetch("./wheel.json")).json();
  pyodide = await loadPyodide({ indexURL: new URL("./vendor/", import.meta.url).href });
  await pyodide.loadPackage(["numpy", "scipy", "micropip"]);
  const micropip = pyodide.pyimport("micropip");
  try {
    for (const wheel of [...config.dependencies].reverse()) {
      await micropip.install(new URL(wheel, import.meta.url).href);
    }
    await micropip.install(new URL(config.wheel, import.meta.url).href);
  } finally { micropip.destroy(); }
  pyodide.runPython(`
import base64, inspect, json, time
import numpy as np
from struphy_web import domains, CircularFlux, EQDSKFlux, sample_surface, save_geometry, load_geometry
current_domain = None

def catalog():
    result = {}
    for name, cls in vars(domains).items():
        if inspect.isclass(cls) and cls.__module__ == domains.__name__:
            result[name] = {k: v.default for k, v in inspect.signature(cls).parameters.items() if k != "equilibrium"}
    return result

def snapshot(domain, request):
    resolution = int(request.get("resolution", 24))
    if not 4 <= resolution <= 64:
        raise ValueError("Resolution must be between 4 and 64")
    point = request.get("point", [.37, .23, .41])
    if np.shape(point) != (3,) or not np.isfinite(point).all() or not np.all((np.array(point) > 0) & (np.array(point) < 1)):
        raise ValueError("Probe coordinates must be three finite values strictly between 0 and 1")
    started = time.perf_counter()
    xyz = sample_surface(domain, resolution)
    mapped = domain(*point, squeeze_out=True)
    jacobian = domain.jacobian(*point, squeeze_out=True)
    metric = domain.metric(*point, squeeze_out=True)
    determinant = float(domain.jacobian_det(*point, squeeze_out=True))
    if not all(np.isfinite(x).all() for x in (*xyz, mapped, jacobian, metric, determinant)):
        raise ValueError("The mapping is singular or non-finite at the requested points")
    return dict(name=type(domain).__name__, xyz=[a.tolist() for a in xyz], mapped=mapped.tolist(),
                jacobian=jacobian.tolist(), metric=metric.tolist(), determinant=determinant,
                elapsed=time.perf_counter()-started)

def handle(request):
    global current_domain
    action = request["action"]
    if action == "render":
        name = request["name"]
        if name not in catalog():
            raise ValueError("Unknown domain")
        params = request["params"]
        if name == "Tokamak":
            params["equilibrium"] = EQDSKFlux.from_text(request["eqdsk"]) if request.get("eqdsk") else CircularFlux()
        candidate = getattr(domains, name)(**params)
        result = snapshot(candidate, request)
        current_domain = candidate
        return result
    if action == "load":
        candidate = load_geometry(base64.b64decode(request["data"]))
        result = snapshot(candidate, request)
        current_domain = candidate
        return result
    if action == "save":
        if current_domain is None:
            raise ValueError("Construct a domain first")
        return base64.b64encode(save_geometry(current_domain)).decode()
    raise ValueError("Unknown action")
`);
  postMessage({ type: "ready", catalog: JSON.parse(pyodide.runPython("json.dumps(catalog())")) });
})();
ready.catch(error => postMessage({ type: "fatal", error: String(error) }));

// Queue requests: each operation owns the Python globals until it completes.
let queue = ready;
self.onmessage = ({ data }) => {
  queue = queue.catch(() => {}).then(async () => {
    try {
      if (data.action === "tests") {
        await ready;
        await pyodide.loadPackage("pytest");
        pyodide.FS.writeFile("/test_geometry.py", await (await fetch("./test_geometry.py")).text());
        const output = [];
        pyodide.setStdout({ batched: line => output.push(line) });
        pyodide.setStderr({ batched: line => output.push(line) });
        const code = pyodide.runPython(`import os, pytest\nos.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'\nint(pytest.main(['-q', '/test_geometry.py', '-p', 'no:cacheprovider']))`);
        postMessage({ id: data.id, result: { code, output: output.join("\n") } });
      } else {
        await ready;
        pyodide.globals.set("request_json", JSON.stringify(data));
        const result = JSON.parse(pyodide.runPython("json.dumps(handle(json.loads(request_json)), allow_nan=False)"));
        postMessage({ id: data.id, result });
      }
    } catch (error) { postMessage({ id: data.id, error: String(error) }); }
  });
};
