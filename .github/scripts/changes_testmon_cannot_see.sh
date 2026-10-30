#!/bin/bash
# Decide which unit tests testmon may select, from the files changed in this run.
#
# testmon only sees changes in Python code executed by the tests, not in compiled kernels (*kernels*.py) or in the
# compiled parts of the feectools submodule. Writes to $GITHUB_ENV:
#   FULL_RUN=true      run all tests (--testmon-noselect): kernels, feectools or the build changed, or the run is
#                      not a pull request (push to devel, schedule, manual dispatch). Pushes always run all tests:
#                      a cancelled or failed devel run would otherwise leave testmon data that misses a kernel
#                      change, which PRs then restore.
#   OPENMP_RUN=true    run the OpenMP checks: FULL_RUN, or changes in the PIC code or in the compile command
#
# usage: changes_testmon_cannot_see.sh <repo dir> <event name> <base branch>
set -u
cd "$1"
event="$2"
base_ref="$3"

full_run=false
openmp_run=false
case "$event" in
    pull_request)
        base=$(git merge-base "origin/$base_ref" HEAD) || base="" ;;
    *)
        base="" ;;
esac

if [ -z "$base" ] || ! changed=$(git diff --name-only "$base" HEAD); then
    echo "Event '$event' (or no diff base): run all tests."
    full_run=true
    openmp_run=true
else
    echo "Files changed since $base:"
    echo "$changed"
    if echo "$changed" | grep -qE '(^|/)[^/]*kernels[^/]*\.py$|^feectools$|(^|/)compile_struphy\.mk$|^src/struphy/console/compile\.py$|^src/struphy/dependencies\.py$|^pyproject\.toml$|^\.github/actions/compile/'; then
        full_run=true
    fi
    if echo "$changed" | grep -qE '^src/struphy/pic/|^src/struphy/console/'; then
        openmp_run=true
    fi
fi
[ "$full_run" = true ] && openmp_run=true

echo "FULL_RUN=$full_run"
echo "OPENMP_RUN=$openmp_run"
if [ -n "${GITHUB_ENV:-}" ]; then
    echo "FULL_RUN=$full_run" >> "$GITHUB_ENV"
    echo "OPENMP_RUN=$openmp_run" >> "$GITHUB_ENV"
fi
