#!/bin/bash
# Decide which unit tests testmon may select, from the files changed in this run.
#
# testmon only sees changes in Python code executed by the tests, not in compiled kernels (*kernels*.py) or in the
# compiled parts of the feectools submodule. Writes to $GITHUB_ENV:
#   FULL_RUN=true      run all tests (--testmon-noselect): kernels, feectools or the build changed,
#                      or the run is not a pull request / push (schedule, manual dispatch)
#   OPENMP_RUN=true    run the OpenMP checks: FULL_RUN, or changes in the PIC code or in the compile command
#
# usage: changes_testmon_cannot_see.sh <repo dir> <event name> <base branch> <sha before push>
set -u
cd "$1"
event="$2"
base_ref="$3"
before="$4"

full_run=false
openmp_run=false
case "$event" in
    pull_request)
        base=$(git merge-base "origin/$base_ref" HEAD) || base="" ;;
    push)
        base="$before" ;;
    *)
        base="" ;;
esac

if [ -z "$base" ] || ! changed=$(git diff --name-only "$base" HEAD); then
    echo "No diff base for event '$event': run all tests."
    full_run=true
    openmp_run=true
else
    echo "Files changed since $base:"
    echo "$changed"
    if echo "$changed" | grep -qE '(^|/)[^/]*kernels[^/]*\.py$|^feectools$|(^|/)compile_struphy\.mk$|^pyproject\.toml$|^\.github/actions/compile/'; then
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
