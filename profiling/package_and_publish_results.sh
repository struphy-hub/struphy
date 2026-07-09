#!/usr/bin/env bash

set -euo pipefail

RESULTS_ROOT=""
OUTPUT_ROOT="profiling-results-export"
TARGET_REPO="git@github.com:struphy-hub/profiling-data.git"
TARGET_BRANCH="main"
CLONE_DIR="profiling-data"
LANGUAGE=""
COMMIT_SHA=""
LATEST_RESULTS_ROOT_FILE="results/profiling/latest_run_root.txt"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results-root)
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --language)
      LANGUAGE="$2"
      shift 2
      ;;
    --commit)
      COMMIT_SHA="$2"
      shift 2
      ;;
    --target-repo)
      TARGET_REPO="$2"
      shift 2
      ;;
    --target-branch)
      TARGET_BRANCH="$2"
      shift 2
      ;;
    --clone-dir)
      CLONE_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$LANGUAGE" ]]; then
  echo "Missing required argument: --language" >&2
  exit 1
fi

if [[ -z "$COMMIT_SHA" ]]; then
  COMMIT_SHA="$(git rev-parse HEAD)"
fi

if [[ -z "$RESULTS_ROOT" ]]; then
  if [[ -f "$LATEST_RESULTS_ROOT_FILE" ]]; then
    RESULTS_ROOT="$(cat "$LATEST_RESULTS_ROOT_FILE")"
    echo "Using latest profiling run root from marker: $RESULTS_ROOT"
  else
    echo "Missing --results-root and no marker file found at $LATEST_RESULTS_ROOT_FILE" >&2
    exit 1
  fi
fi

python profiling/package_profiling_results.py \
  --results-root "$RESULTS_ROOT" \
  --language "$LANGUAGE" \
  --commit "$COMMIT_SHA" \
  --output-root "$OUTPUT_ROOT"

rm -rf "$CLONE_DIR"
git clone "$TARGET_REPO" "$CLONE_DIR"
cp -R "$OUTPUT_ROOT"/. "$CLONE_DIR"/

pushd "$CLONE_DIR" >/dev/null
git config user.name "${GIT_AUTHOR_NAME:-github-actions[bot]}"
git config user.email "${GIT_AUTHOR_EMAIL:-github-actions[bot]@users.noreply.github.com}"
git add .
if git diff --cached --quiet; then
  echo "No profiling data changes to commit."
  popd >/dev/null
  exit 0
fi
git commit -m "Add profiling data (${COMMIT_SHA::8}, ${LANGUAGE})"
git push origin "$TARGET_BRANCH"
popd >/dev/null
