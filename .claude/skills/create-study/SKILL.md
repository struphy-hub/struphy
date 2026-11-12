---
name: create-study
description: Use when creating, extending or publishing a study in struphy-studies (a page of the public site with Struphy runs, their figures and videos, findings and the commands to reproduce them) - planning the runs, post-processing, comparison figures, tools/study.py create/link/check, study.yaml, the local preview and the media upload to MPCDF DataShare. Triggers on requests like "make a study out of these runs", "add the runs to the study", "draft the study page", "publish the study".
---

# Creating a Struphy study

Studies live in the repository struphy-hub/struphy-studies (https://github.com/struphy-hub/struphy-studies, site
https://struphy-hub.github.io/struphy-studies/), not in struphy: clone it (`git clone
git@github.com:struphy-hub/struphy-studies.git`) and run the commands of this skill in that clone. This skill is the
same as `.claude/skills/create-study/SKILL.md` of struphy-studies; keep the two in sync.

A study is a folder `studies/<yyyy-mm>-<slug>/` with `study.yaml` (texts, runs, commands), `params/` (the parameter,
post-processing and comparison scripts) and `thumbnail.png`. Figures and videos are **not in git**: they live in a
media folder on MPCDF DataShare, shared with a public link, and the site embeds them from there. Read the `README.md`
of struphy-studies first; `studies/2026-11-lowmode-tae-shear-alfven/` there is a complete example.

## 1. Plan the study with the user

- The question (one or two sentences), the runs and what changes between them (one thing per run against a base run
  where possible), the end time and the quantities to compare (frequency, growth rate, mode structure, energy, ...).
- Estimate the cost of each run from a few time steps and ask the user before starting runs that take long.
- Run with a released Struphy or a pushed commit: the study records the commit the runs were made with
  (`run_metadata.json`; `study.py create --commit` sets another one), and readers install exactly that. Runs from an
  unmerged branch are fine for a draft; update `struphy.commit` once it is merged and check that the scripts still
  run with it.
- Keep the parameters of the runs consistent: if a default of the example changes while the study is running, pass
  the old value explicitly to the remaining runs and say so in the description.

## 2. Runs and post-processing

- One output folder per run (`--sim-folder <name>`); names are short and say what changes (`base`, `r32`, `polar`,
  `hdiv_24x88x8`). Keep the folders: `study.py create` reads them.
- Post-process every run with the same script, writing figures and videos into `<run>/results/` (the folder
  `study.py create` collects): the standard snapshots, the energies, the analysis of the study, and video frames,
  made into mp4 with ffmpeg (`frame_%04d.png`, `-pix_fmt yuv420p`, even width and height). A small script that runs
  the pproc and ffmpeg for one run keeps this reproducible (e.g. `run_case.sh` of the low-mode TAE study).
- `study.py create` flattens `<run>/results/` into `runs/<name>/{figures,videos}/<file>`: file names must be unique
  across subfolders (equal names overwrite each other), and images inside `frames/` subfolders are skipped (the mp4
  files there are kept).
- Comparisons of several runs go into a `compare.py` in `params/` that reads the run folders next to it (or
  `$STUDY_RUNS`) and writes `compare_*.png` and a table (`compare.md`) for the findings. For waves and eigenmodes, use
  `struphy.post_processing.mode_analysis` (spectra, (r, omega) maps with continua, filtered mode structure); compare
  runs by filtered quantities, not by snapshots.
- If the post-processing changes after a run was processed, process it again so that all runs of the study look the
  same.

## 3. Create the study

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # once
source .venv/bin/activate
python tools/study.py create <yyyy-mm>-<slug> --title "..." \
    --run base=/path/to/base --run fine=/path/to/fine \
    --figure /path/to/compare_spectra.png --script /path/to/compare.py --script /path/to/pproc_case.sh \
    --thumbnail /path/to/a/figure.png
```
It writes `studies/<slug>/` (study.yaml with TODOs, the params file of each run from its `run_metadata.json` and
the `pproc_*.py` next to it, the `--script` files, the thumbnail) and the media folder
`~/struphy-studies-media/<slug>/`. `--force` overwrites **both, including a filled-in study.yaml**: when runs are added
later, save the texts of `study.yaml` first and merge them into the new one. Replace the thumbnail with
`python tools/study.py thumbnail <slug> <image>` (no `--force` needed).

Make the scripts in `params/` portable: `python` from PATH, paths relative to the script folder, no paths of your
machine. The run commands in `study.yaml` are run in `params/` (`mpirun -n 4 python params_x.py --sim-folder base
...`), without `--out-folder` to local paths.

## 4. Fill in study.yaml

- `summary`: one sentence with the answer. `thumbnail_caption`: which run and what the picture shows.
- `description`: **Question.**, **Setup.** (model, equilibrium, domain, grid, time step, end time, ranks, reference),
  **Analysis.** (what is measured and how, the frequency resolution 2 pi / T for spectra).
- `findings`: a table of the runs with the measured quantities, then numbered findings, each with the numbers that
  support it. Embed figures as `![caption](media:figures/<file>)`.
- Captions say which runs a figure shows and their colors or line styles, so that a figure can be read on its own.
- `figures:` as `- path: figures/x.png` with `caption:`; each run gets a `description` (what distinguishes it),
  `command` and `pproc`.
- While runs are pending, mark them as pending in the table and say so in the summary; keep `draft: true`.

## 5. Preview

```bash
python tools/build_site.py --drafts --local-media && mkdocs serve
```
`--local-media` shows the media of studies without a share link from `~/struphy-studies-media/<slug>/` (copied into
`docs/`), so the page can be checked before the upload. Never deploy such a build. Look at the page: figures, videos,
captions, the table, the reproduce section.

## 6. Upload the media and link them (the user)

Uploading to DataShare needs the user's MPCDF account. Tell the user to:
1. upload the folder `~/struphy-studies-media/<slug>/` to https://datashare.mpcdf.mpg.de (web interface or desktop
   client);
2. share only that folder with a link, remove the password, and copy the link.

Then run `python tools/study.py link <slug> <link>`: it stores the link and checks that every file is reachable with
the right content type (and range requests for the videos). `python tools/study.py check <slug>` checks it again
later; CI checks the published studies weekly. If media change after the upload, the user replaces the files in the
same DataShare folder (the link stays) and `study.py check` confirms them.

## 7. Publish

Preview again without `--local-media` (now from DataShare), set `draft: false`, run the checks of the CI (on every
PR: `mkdocs build --strict` and the media check),
```bash
python tools/build_site.py --drafts && mkdocs build --strict && python tools/study.py check <slug>
```
and open a pull request into `main`. After the merge, the site is rebuilt.
