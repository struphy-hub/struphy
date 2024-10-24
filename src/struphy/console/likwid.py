import glob
import os

from struphy.post_processing.likwid.plot_likwidproject import load_projects, plot_files


def struphy_likwid_profile(dir, title, output, groups, skip, plots):
    # Expand wildcard directories
    expanded_dirs = []
    for d in dir:
        expanded_dirs.extend(glob.glob(d))

    # Pass the expanded directories to load_projects
    projects = load_projects(expanded_dirs)
    if len(projects) == 0:
        print("projects not finished")
        exit()

    procs_per_clone = "any"

    print(f"# Plotting simulation: {title}")
    plot_files(
        projects=projects,
        output_path=output,
        title=title,
        plots=plots,
        groups_include=groups,
        groups_skip=skip,
    )
    print("done")
