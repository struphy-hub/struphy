#!/usr/bin/env bash
# Regenerate the profiling figures used in the "Code Profiling" section of
# doc/sections/userguide.rst.
#
# Runs the default Vlasov example with profiling enabled, post-processes it
# with `scope-profiler plot all`, and copies the resulting Gantt chart and flame
# graph into doc/pics/.
#
# Requires `struphy` and `scope-profiler` to be installed and on PATH.
#
# Usage: ./doc/generate_profiling_figures.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PICS_DIR="$SCRIPT_DIR/pics"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

cd "$WORK_DIR"

struphy params Vlasov -y

python - <<'PY'
path = "params_Vlasov.py"
content = open(path).read()
needle = "env = EnvironmentOptions()"
if needle not in content:
    raise SystemExit(f"Could not find '{needle}' in {path} -- template may have changed.")
content = content.replace(
    needle,
    "env = EnvironmentOptions(\n"
    "    profiling_activated=True,\n"
    ")",
)
open(path, "w").write(content)
PY

python params_Vlasov.py

scope-profiler plot all sim_1/profiling_data.h5 \
    --include '^setup: total$' '^model\.integrate$' '^prop: ' '^kernel: ' \
    -o figures

cp figures/gantt_plot.png "$PICS_DIR/profiling_gantt_chart.png"
cp figures/flame_plot.png "$PICS_DIR/profiling_flame_graph.png"

echo "Updated:"
echo "  $PICS_DIR/profiling_gantt_chart.png"
echo "  $PICS_DIR/profiling_flame_graph.png"
