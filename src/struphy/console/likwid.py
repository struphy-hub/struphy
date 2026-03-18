import logging


import glob
import os
import sys

from struphy.post_processing.likwid.plot_likwidproject import load_projects, plot_files

logger = logging.getLogger("struphy")

def struphy_likwid_profile(dir, title, output, groups, skip, plots):
    # Expand wildcard directories
    expanded_dirs = []
    for d in dir:
        expanded_dirs.extend(glob.glob(d))

    # Pass the expanded directories to load_projects
    projects = load_projects(expanded_dirs)
    if len(projects) == 0:
        logger.info("projects not finished")
        sys.exit(1)

    procs_per_clone = "any"

    logger.info(f"# Plotting simulation: {title}")
    plot_files(
        projects=projects,
        output_path=output,
        title=title,
        plots=plots,
        groups_include=groups,
        groups_skip=skip,
    )
    logger.info("done")
