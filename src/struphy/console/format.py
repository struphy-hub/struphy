"""
Struphy Linting and Formatting Tools

This module provides utilities to lint, format, and analyze Python files in StruPhy.
It is accessible when compiling StruPhy in editable mode.

Functions
---------
parse_path(directory)
    Traverse a directory to find Python files, excluding '__XYZ__.py'.

get_python_files(input_type, path=None)
    Retrieve Python files based on the specified input type ('all', 'path', 'staged', or 'branch').

struphy_lint(config, verbose)
    Lint Python files based on the given configuration, using specified linters.

struphy_format(config, verbose, yes=False)
    Format Python files with specified linters, optionally iterating multiple times.

print_stats_table(stats_list, linters, print_header=True, pathlen=0)
    Print statistics for Python files in a tabular format.

print_stats_plain(stats, linters)
    Print statistics for Python files in a plain-text format.

analyze_file(file_path, linters=None, verbose=False)
    Analyze a Python file, reporting on code structure and linter compliance.

generate_report(python_files, linters=["ruff"], verbose=False)
    Generate a linting report in HTML format for specified files.

parse_json_file_to_html(json_file_path, html_output_path)
    Parse a JSON linting report into an HTML file with detailed context for each issue.

check_omp_flags(file_path)
    Check if a file contains OpenMP-like flags (`# $`).

check_ruff(file_path, verbose=False)
    Check if a file passes Ruff linting.

check_isort(file_path, verbose=False)
    Check if a file is sorted according to isort.

check_autopep8(file_path, verbose=False)
    Check if a file is formatted according to autopep8.

check_flake8(file_path, verbose=False)
    Check if a file is formatted according to flake8.

get_pylint_score(file_path, verbose=False, pass_score=8.0)
    Get pylint score for a file and determine if it passes based on a minimum score.

check_trailing_commas(file_path, verbose=False)
    Check if a file contains trailing commas required by add-trailing-commas.

files_require_formatting(python_files, linters)
    Determine if any specified files require formatting based on the given linters.

run_linters_on_files(linters, python_files, flags, verbose)
    Run specified linters on the provided files with appropriate flags.

confirm_formatting(python_files, linters, yes)
    Confirm with the user whether to format the listed Python files.

replace_backticks_with_code_tags(text)
    Replace inline backticks with <code> tags, handling multiple or nested occurrences.

generate_html_table_from_combined_data(combined_data, sort_descending=True)
    Generate an HTML table from combined data for code issues.

"""

import ast
import fileinput
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict

from tabulate import tabulate

import struphy

LIBPATH = struphy.__path__[0]

GREEN_COLOR = "\033[92m"
RED_COLOR = "\033[91m"
BLACK_COLOR = "\033[0m"

FAIL_RED = f"{RED_COLOR}FAIL{BLACK_COLOR}"
PASS_GREEN = f"{GREEN_COLOR}PASS{BLACK_COLOR}"


MODELS_INIT_PATH = os.path.join(LIBPATH, "models/__init__.py")
PROPAGATORS_INIT_PATH = os.path.join(LIBPATH, "propagators/__init__.py")


def check_omp_flags(file_path, verbose=False):
    """Checks if a file contains incorrect OpenMP-like flags (`# $`).

    Parameters:
    -----------
    file_path : str
        Path to the file to check.

    Returns:
    --------
    bool
        True if no incorrect OpenMP-like flags (`# $`) are found, False otherwise.
    """
    try:
        with open(file_path, "r") as f:
            if verbose:
                for iline, line in enumerate(f):
                    if line.lstrip().startswith("# $"):
                        print(f"Error on line {iline}: {line}")
            return all(not line.lstrip().startswith("# $") for line in f)
    except (IOError, FileNotFoundError) as e:
        raise ValueError(f"Error reading file: {e}")


def check_ssort(file_path, verbose=False):
    """Check if a file is sorted according to ssort.

    Parameters
    ----------
    file_path : str
        Path to the Python file.

    verbose : bool, optional
        If True, enables detailed output (default=False).

    Returns
    -------
    bool
        True if ssort check passes, False otherwise.
    """
    result = subprocess.run(
        ["ssort", "--check", file_path],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if verbose:
        print("stdout:", result.stdout.decode("utf-8"))
        print("stderr:", result.stderr.decode("utf-8"))
    return result.returncode == 0


def check_ruff(file_path, verbose=False):
    """Check if a file passes Ruff linting.

    Parameters
    ----------
    file_path : str
        Path to the Python file.

    verbose : bool, optional
        If True, enables detailed output (default=False).

    Returns
    -------
    bool
        True if Ruff check passes, False otherwise.
    """

    commands = [
        ["ruff", "format", "--diff", file_path],
        ["ruff", "check", "--select", "I", file_path],  # Check with isort
    ]
    returncodes = []
    for command in commands:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if verbose:
            print("stdout:", result.stdout.decode("utf-8"))
            print("stderr:", result.stderr.decode("utf-8"))

        if result.returncode == 0:
            returncodes.append(0)
        else:
            # Default to no error
            returncode = 0
            for line in result.stdout.decode("utf-8").split("\n"):
                # Skip empty lines and filename headers
                if not line or line.startswith("+++ "):
                    continue
                # Check for lines with actual changes
                if line.startswith("+ ") and not line[1:].lstrip().startswith("# $"):
                    returncode = 1
            returncodes.append(returncode)

    return all(returncode == 0 for returncode in returncodes)


def check_isort(file_path, verbose=False):
    """Check if a file is sorted according to isort.

    Parameters
    ----------
    file_path : str
        Path to the Python file.

    verbose : bool, optional
        If True, enables detailed output (default=False).

    Returns
    -------
    bool
        True if isort check passes, False otherwise.
    """
    result = subprocess.run(
        ["isort", "--check-only", file_path],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if verbose:
        print("stdout:", result.stdout.decode("utf-8"))
        print("stderr:", result.stderr.decode("utf-8"))
    return result.returncode == 0


def check_autopep8(file_path, verbose=False):
    """Check if a file is formatted according to autopep8.

    Parameters
    ----------
    file_path : str
        Path to the Python file.

    verbose : bool, optional
        If True, enables detailed output (default=False).

    Returns
    -------
    bool
        True if autopep8 check passes, False otherwise.
    """

    result = subprocess.run(
        ["autopep8", "--diff", file_path],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # If there's any output, autopep8 suggests changes, so it doesn't pass
    if verbose:
        print("stdout:", result.stdout.decode("utf-8"))
        print("stderr:", result.stderr.decode("utf-8"))
    return result.stdout == b""


def check_flake8(file_path, verbose=False):
    """Check if a file is formatted according to flake8.

    Parameters
    ----------
    file_path : str
        Path to the Python file.

    verbose : bool, optional
        If True, enables detailed output (default=False).

    Returns
    -------
    bool
        True if flake8 check passes, False otherwise.
    """

    result = subprocess.run(
        ["flake8", file_path],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if verbose:
        print("stdout:", result.stdout.decode("utf-8"))
        print("stderr:", result.stderr.decode("utf-8"))
    return result.returncode == 0


def get_pylint_score(file_path, verbose=False, pass_score=8.0):
    """Get pylint score for a file and determine if it passes.

    Parameters
    ----------
    file_path : str
        Path to the Python file.

    verbose : bool, optional
        If True, enables detailed output (default=False).

    pass_score : float, optional
        Minimum pylint score for passing (default=8.0).

    Returns
    -------
    tuple
        Pylint score (float) and pass status (bool).
    """

    result = subprocess.run(
        ["pylint", "--score=y", file_path],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    output = result.stdout
    score = None
    passes_pylint = False

    if verbose:
        print(f"\nPylint report for {file_path}:")
        print(output)

    # Parse the output to get the score
    for line in output.splitlines():
        if line.startswith("Your code has been rated at"):
            try:
                match = re.search(r"(\d+\.\d+)/10", line)
                if match:
                    score = float(
                        match.group(1),
                    )  # Extract the score part (e.g., "9.73")
                else:
                    score = 0.0
                passes_pylint = score >= pass_score
            except (IndexError, ValueError):
                pass
            break

    return score, passes_pylint


def check_trailing_commas(file_path, verbose=False):
    """Check if a file contains trailing commas as required by add-trailing-commas.

    Parameters
    ----------
    file_path : str
        Path to the Python file.

    verbose : bool, optional
        If True, enables detailed output (default=False).

    Returns
    -------
    bool
        True if trailing commas check passes, False otherwise.
    """
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tempdir:
        # Copy the file
        temp_file = os.path.join(tempdir, os.path.basename(file_path))
        shutil.copy(file_path, temp_file)

        # Run add-trailing-comma on the temporary file
        subprocess.run(
            ["add-trailing-comma", temp_file],
            capture_output=True,
            text=True,
            check=False,
        )

        # Compare to the original file
        diff_result = subprocess.run(
            ["diff", file_path, temp_file],
            capture_output=True,
            text=True,
            check=False,
        )

        if diff_result.stdout:
            if verbose:
                print(f"Changes by add-trailing-comma {file_path}\n")
                print(diff_result.stdout)
            return False
        return True


def parse_path(directory):
    """Traverse a directory to find Python files, excluding '__XYZ__.py'.

    Parameters
    ----------
    directory : str
        The directory to search for Python files.

    Returns
    -------
    list
        Paths of Python files in the directory.
    """

    python_files = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if re.search(r"__\w+__", root):
                continue
            if filename.endswith(".py") and not re.search(r"__\w+__", filename):
                file_path = os.path.join(root, filename)
                python_files.append(file_path)
    # exit()
    return python_files


def get_python_files(input_type, path=None):
    """Retrieve Python files based on the specified input type.

    Parameters
    ----------
    input_type : str
        Specifies the files to retrieve ('all', 'path', 'staged', or 'branch').

    path : str, optional
        Path to a directory or file (default=None).

    Returns
    -------
    list
        Paths to the relevant Python files.
    """

    repo_command = ["git", "rev-parse", "--show-toplevel"]
    repopath = subprocess.check_output(
        repo_command,
        universal_newlines=True,
        cwd=LIBPATH,
    ).strip()

    if path is None:
        path = LIBPATH
    if input_type == "all":
        # Get the struphy library path
        python_files = parse_path(LIBPATH)

    elif input_type == "path":
        print(path)
        if os.path.isfile(path):
            print("isfile")
            python_files = [path]
        else:
            python_files = parse_path(path)

    elif input_type == "staged":
        git_command = ["git", "diff", "--cached", "--name-only"]

        git_output = subprocess.check_output(
            git_command,
            universal_newlines=True,
            cwd=LIBPATH,
        )
        files = git_output.strip().splitlines()
        python_files = [
            os.path.join(repopath, f) for f in files if f.endswith(".py") and os.path.isfile(os.path.join(repopath, f))
        ]

        if not python_files:
            print("No Python files to analyze.")
            return []

    elif input_type == "branch":
        # Compare the current branch against origin/devel
        git_command = ["git", "diff", "--name-only", "origin/devel"]

        git_output = subprocess.check_output(
            git_command,
            universal_newlines=True,
            cwd=LIBPATH,
        )
        files = git_output.strip().splitlines()

        # python_files = [f for f in files if f.endswith(".py") and os.path.isfile(f)]
        python_files = [
            os.path.join(repopath, f) for f in files if f.endswith(".py") and os.path.isfile(os.path.join(repopath, f))
        ]

        if not python_files:
            print(
                f"No Python files changed between the current branch and '{path}' branch.",
            )
            return []

    else:
        print(f"Unhandled input_type '{input_type}'.")
        sys.exit(1)

    if not python_files:
        print("No Python files found to check.")
        return []

    python_files = [
        f for f in python_files if not (re.match(r"^__\w+__\.py$", os.path.basename(f)) and "__init__.py" not in f)
    ]
    return python_files


def struphy_lint(config, verbose):
    """Lint Python files based on the given configuration and specified linters.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing the following keys:
            - input_type : str, optional
                The type of files to lint ('all', 'path', 'staged', or 'branch'). Defaults to 'all'.
            - output_format: str, optional
                The format of the lint output ('table', or 'plain'). Defaults to 'table'
            - path : str, optional
                Directory or file path to lint.
            - linters : list
                List of linter names to apply.

    verbose : bool
        If True, enables detailed output.
    """

    # Extract individual settings from config
    input_type = config.get("input_type", "all")
    path = config.get("path")
    output_format = config.get("output_format", "table")
    linters = config.get("linters", [])

    if input_type is None and path is not None:
        input_type = "path"
    # Define standard linters which will be checked in the CI
    ci_linters = ["ruff", "omp_flags"]
    python_files = get_python_files(input_type, path)
    if len(python_files) == 0:
        sys.exit(0)

    print(
        tabulate(
            [[file] for file in python_files],
            headers=[f"The following files will be linted with {linters}"],
        ),
    )
    print("\n")

    if output_format == "report":
        generate_report(python_files, linters=linters, verbose=verbose)
        sys.exit(0)

    max_pathlen = max(len(os.path.relpath(file_path)) for file_path in python_files)
    stats_list = []

    # Check if all ci_linters are included in linters
    if all(ci_linter in linters for ci_linter in ci_linters):
        print(f"Passes CI if {ci_linters} passes")
        print("-" * 40)
        check_ci_pass = True
    else:
        skipped_ci_linters = [ci_linter for ci_linter in ci_linters if ci_linter not in linters]
        print(
            f'The "Pass CI" check is skipped since not --linters {" ".join(skipped_ci_linters)} is used.',
        )
        check_ci_pass = False
    # Collect statistics for each file
    for ifile, file_path in enumerate(python_files):
        stats = analyze_file(file_path, linters=linters, verbose=verbose)
        stats_list.append(stats)

        # Print the statistics in a table
        if output_format == "table":
            print_stats_table(
                [stats],
                linters,
                print_header=(ifile == 0),
                pathlen=max_pathlen,
            )
        elif output_format == "plain":
            print_stats_plain(stats, linters)

    if check_ci_pass:
        passes_ci = True
        for stats in stats_list:
            if not all(stats[f"passes_{ci_linter}"] for ci_linter in ci_linters):
                passes_ci = False
        if passes_ci:
            print("All files will pass CI")
            sys.exit(0)
        else:
            print("Not all files will pass CI")
            sys.exit(1)
    print("Not all CI linters were checked, unknown if all files will pass CI")
    sys.exit(1)


def generate_report(python_files, linters=["ruff"], verbose=False):
    for linter in linters:
        if linter == "ruff":
            for python_file in python_files:
                report_json_filename = "code_analysis_report.json"
                report_html_filename = "code_analysis_report.html"
                command = [
                    "ruff",
                    "check",
                    "--preview",
                    "--select",
                    "ALL",
                    "--ignore",
                    "D211,D213",
                    "--output-format",
                    "json",
                    "-o",
                    report_json_filename,
                ] + python_files
                subprocess.run(command, check=False)
                parse_json_file_to_html(report_json_filename, report_html_filename)
                if os.path.exists(report_json_filename):
                    os.remove(report_json_filename)
                sys.exit(0)


def confirm_formatting(python_files, linters, yes):
    """Confirm with the user whether to format the listed Python files."""
    print(
        tabulate(
            [[file] for file in python_files],
            headers=[f"The following files will be formatted with {linters}"],
        ),
    )
    print("\n")
    if not yes:
        ans = input("Format files (Y/n)?\n")
        if ans.lower() not in ("y", "yes", ""):
            print("Exiting...")
            sys.exit(1)


def files_require_formatting(python_files, linters):
    """Check if any of the specified files still require formatting based on the specified linters.

    Parameters
    ----------
    python_files : list
        List of Python file paths to check.

    linters : list
        List of linter names to check against (e.g., ['autopep8', 'isort']).

    Returns
    -------
    bool
        True if any files still require formatting, False otherwise.
    """
    linter_check_functions = {
        "autopep8": check_autopep8,
        "isort": check_isort,
        "add-trailing-comma": check_trailing_commas,
    }

    for file_path in python_files:
        for linter in linters:
            check_function = linter_check_functions.get(linter)
            if check_function and not check_function(file_path):
                return True
    return False


def run_linters_on_files(linters, python_files, flags, verbose):
    """Run each linter on the specified files with appropriate flags."""
    for linter in linters:
        for python_file in python_files:
            print(f"Formatting {python_file}")
            linter_flags = flags.get(linter, [])
            if isinstance(linter_flags[0], list):
                # If linter_flags is a list, run each separately
                for flag in linter_flags:
                    command = [linter] + flag + [python_file]
                    if verbose:
                        print(f"Running command: {' '.join(command)}")

                    subprocess.run(command, check=False)
            else:
                # If linter_flags is not a list, treat it as a single value
                command = [linter] + linter_flags + [python_file]
                if verbose:
                    print(f"Running command: {' '.join(command)}")
                subprocess.run(command, check=False)

            # Loop over each line and replace '# $' with '#$' in place
            for line in fileinput.input(python_file, inplace=True):
                if line.lstrip().startswith("# $"):
                    print(line.replace("# $", "#$"), end="")
                else:
                    print(line, end="")


def struphy_format(config, verbose, yes=False):
    """Format Python files with specified linters, optionally iterating multiple times.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing the following keys:
            - input_type : str, optional
                The type of files to format ('all', 'path', 'staged', 'branch', or '__init__.py'). Defaults to 'all'.
            - path : str, optional
                Directory or file path where files will be formatted.
            - linters : list
                List of formatter names to apply.
            - iterations : int, optional
                Maximum number of times to apply formatting (default=5).

    verbose : bool
        If True, enables detailed output, showing each command and iteration.

    yes : bool, optional
        If True, skips the confirmation prompt before formatting.
    """

    # Extract individual settings from config
    input_type = config.get("input_type", "all")
    path = config.get("path")
    linters = config.get("linters", [])
    iterations = config.get("iterations", 5)

    if input_type is None and path is not None:
        input_type = "path"

    if input_type == "__init__.py":
        print(f"Rewriting {PROPAGATORS_INIT_PATH}")
        propagators_init = construct_propagators_init_file()
        with open(PROPAGATORS_INIT_PATH, "w") as f:
            f.write(propagators_init)

        print(f"Rewriting {MODELS_INIT_PATH}")
        models_init = construct_models_init_file()
        with open(MODELS_INIT_PATH, "w") as f:
            f.write(models_init)

        python_files = [PROPAGATORS_INIT_PATH, MODELS_INIT_PATH]
        input_type = "path"
    else:
        python_files = get_python_files(input_type, path)

    if len(python_files) == 0:
        print("No Python files to format.")
        sys.exit(0)

    confirm_formatting(python_files, linters, yes)

    flags = {
        "autopep8": ["--in-place"],
        "isort": [],
        "add-trailing-comma": ["--exit-zero-even-if-changed"],
        "ruff": [["check", "--fix", "--select", "I"], ["format"]],
    }

    # Skip linting with add-trailing-comma since it disagrees with autopep8
    skip_linters = ["add-trailing-comma"]

    if python_files:
        for iteration in range(iterations):
            if verbose:
                print(f"Iteration {iteration + 1}: Running formatters...")

            run_linters_on_files(
                linters,
                python_files,
                flags,
                verbose,
            )

            # Check if any files still require changes
            if not files_require_formatting(
                python_files,
                [lint for lint in linters if lint not in skip_linters],
            ):
                print("All files are properly formatted.")
                break
        else:
            if verbose:
                print(
                    "Max iterations reached. The following files may still require manual checks:",
                )
                for file_path in python_files:
                    if files_require_formatting([file_path], linters):
                        print(f" - {file_path}")
                print("Contact Max about this")
    else:
        print("No Python files to format.")


def print_stats_plain(stats, linters, ci_linters=["ruff"]):
    """Print statistics for a single file in plain text format.

    Parameters
    ----------
    stats : dict
        Dictionary containing statistics for a single file.

    linters : list
        List of linters to display in the output.
    """
    print(f"File: {os.path.relpath(stats['path'])}")
    print(f"  Lines: {stats['num_lines']}")
    print(f"  Functions: {stats['num_functions']}")
    print(f"  Classes: {stats['num_classes']}")
    print(f"  Variables: {stats['num_variables']}")

    if "pylint" in linters:
        print(f"  Pylint Score: {stats['pylint_score']}/10")

    for linter in linters:
        status = PASS_GREEN if stats[f"passes_{linter}"] else FAIL_RED
        print(f"  {linter}: {status}")

    # Check for CI pass status if both linters are present
    if all(linter in linters for linter in ci_linters):
        # Check if all linters in ci_linters pass
        passes_ci = all(stats[f"passes_{linter}"] for linter in ci_linters)
        ci_status = PASS_GREEN if passes_ci else FAIL_RED
        print(f"  Full CI check: {ci_status}")
    print("-" * 40)  # Divider between files


def print_stats_table(stats_list, linters, print_header=True, pathlen=0, ci_linters=["ruff"]):
    """Print statistics for Python files in a tabular format.

    Parameters
    ----------
    stats_list : list
        List of file statistics dictionaries.

    linters : list
        List of linters to display in the table.

    print_header : bool, optional
        If True, print the table header (default=True).

    pathlen : int, optional
        Maximum length for path column formatting (default=0).
    """

    file_header = " " * int((pathlen - 4) * 0.5)
    file_header += "File"
    file_header += " " * int((pathlen - 4) * 0.5)
    headers = [
        file_header,
        "Lines",
        "Funcs",
        "Classes",
        "Vars",
    ]
    return python_files


def replace_backticks_with_code_tags(text):
    """Recursively replaces inline backticks with <code> tags.
    Handles multiple or nested occurrences.

    Args:
        text (str): Input string with backticks to be replaced.

    Returns:
        str: Formatted string with <code> tags.
    """
    # Regular expression to match text inside single backtick pairs
    pattern = r"`([^`]*)`"

    # Replace one level of backticks with <code> tags
    new_text = re.sub(pattern, r"<code>\1</code>", text)

    # If additional backticks are found, process recursively
    if "`" in new_text:
        return replace_backticks_with_code_tags(new_text)

    return new_text


def generate_html_table_from_combined_data(combined_data, sort_descending=True):
    html = "<table class='table-style'>"
    html += "<thead><tr><th>Count</th><th>Codes</th></tr></thead><tbody>"
    sorted_items = sorted(combined_data.items(), reverse=sort_descending)
    for count, info in sorted_items:
        codes_links = ", ".join(info["Codes"])
        html += f"<tr><td>{count}</td><td>{codes_links}</td></tr>"
    html += "</tbody></table>"
    return html


def parse_json_file_to_html(json_file_path, html_output_path):
    """Parses a JSON file containing code issues and writes an HTML report.
    Parses a JSON file containing code issues, groups them by filename,
    reads the source code to extract context, and writes an HTML report.
    Each file's section is foldable using <details> and <summary> tags.

    Args:
        json_file_path (str): The path to the JSON file containing code issues.
        html_output_path (str): The path where the HTML report will be saved.
    """

    try:
        with open(json_file_path, "r") as file:
            data = json.load(file)

        if not isinstance(data, list):
            print("Invalid JSON format: Expected a list of objects.")
            return

        # Group issues by filename
        issues_by_file = defaultdict(list)
        for issue in data:
            filename = issue.get("filename", "Unknown file")
            issues_by_file[filename].append(issue)

        # Start building the HTML content
        html_content = []
        html_content.extend(
            [
                "<!DOCTYPE html>",
                "<html lang='en'>",
                "<head>",
                "<meta charset='UTF-8'>",
                "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
                "<title>Code Analysis Report</title>",
            ],
        )

        # Include external CSS and JS libraries
        html_content.extend(
            [
                "<link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/bulma@0.9.3/css/bulma.min.css'>",
                "<link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/default.min.css'>",
                "<script src='https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js'></script>",
                "<script>hljs.configure({ignoreUnescapedHTML: true}); hljs.highlightAll();</script>",
            ],
        )

        # Custom CSS for light mode and code prettification
        html_content.append("<style>")
        html_content.append(
            """
body {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    margin: 20px;
    background-color: #ffffff; /* Light background */
    color: #333333; /* Dark text color */
}
h1 {
    color: #333333;
    font-size: 2em;
}
a {
    color: #3273dc;
}
code {
    font-family: 'Fira Code', Consolas, 'Courier New', monospace;
}
pre {
    background-color: #f5f5f5; /* Light grey background */
    padding: 15px;
    border-radius: 5px;
    overflow: auto;
    position: relative;
    color: #333333;
}
.error {
    padding: 2px 4px;
    border-radius: 3px;
}
details {
    margin-bottom: 20px;
}
summary {
    font-size: 1.2em;
    font-weight: bold;
    cursor: pointer;
    padding: 10px;
    background-color: #f5f5f5;
    color: #333333;
    border-radius: 5px;
}
summary:hover {
    background-color: #e5e5e5;
}
.issue {
    margin-bottom: 20px;
}
.issue-header {
    font-weight: bold;
}
.fix {
    margin-left: 20px;
}
.table-style {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 20px;
    color: #333333;
}
.table-style th, .table-style td {
    border: 1px solid #dddddd;
    padding: 10px;
    text-align: left;
}
.table-style th {
    background-color: #f5f5f5;
}
.table-style tr:nth-child(even) {
    background-color: #f9f9f9;
}
.table-style tr:hover {
    background-color: #f1f1f1;
}
/* Highlighted code segment */
.highlighted-code, mark {
    background-color: #ff3860;
    color: #000000;
    padding: 0;
    border-radius: 0;
}
/* Line numbers */
.line-number {
    position: absolute;
    left: -10px;
    width: 40px;
    text-align: right;
    padding-right: 10px;
    color: #999999;
    user-select: none;
}
.code-line {
    display: block;
    position: relative;
    padding-left: 50px;
    margin: 0;
    line-height: 1.2; /* Adjust line height to reduce spacing */
}
nav ul {
    margin-top: 20px;
}
nav li {
    margin-bottom: 5px;
}
footer {
    margin-top: 40px;
    text-align: center;
    color: #999999;
}
""",
        )
        html_content.append("</style>")

        # JavaScript to initialize Highlight.js with custom options
        html_content.append(
            """
<script>
document.addEventListener('DOMContentLoaded', (event) => {
    hljs.configure({ignoreUnescapedHTML: true});
    document.querySelectorAll('pre code').forEach((block) => {
        hljs.highlightElement(block);
    });
});
</script>
""",
        )

        html_content.extend(["</head>", "<body>", "<h1>Code Issues Report</h1>"])

        # Add summary statistics
        total_issues = sum(len(issues) for issues in issues_by_file.values())
        total_files = len(issues_by_file)
        html_content.append(
            f"""
<section>
    <p><strong>Total Issues:</strong> {total_issues}</p>
    <p><strong>Number of files:</strong> {total_files}</p>
</section>
""",
        )

        # Navigation menu
        #         html_content.append("<nav><ul style='list-style: none; padding: 0;'>")
        #         for filename in issues_by_file.keys():
        #             anchor = filename.replace(LIBPATH, 'src/struphy').replace('/', '_').replace('\\', '_')
        #             display_name = filename.replace(LIBPATH, 'src/struphy')
        #             html_content.append(f"""
        # <li>
        #     <a href='#{anchor}'>{display_name}</a>
        # </li>
        # """)
        #         html_content.append("</ul></nav>")

        for filename, issues in issues_by_file.items():
            print(f"Parsing {filename}")
            # Start foldable section for the file
            anchor = filename.replace(LIBPATH, "src/struphy").replace("/", "_").replace("\\", "_")
            display_name = filename.replace(LIBPATH, "src/struphy")
            html_content.append(
                f"""
<details id='{anchor}'>
    <summary>File: <code>{display_name}</code></summary>
""",
            )

            issue_data = {}
            for issue in issues:
                code = issue.get("code", "Unknown code")
                message = replace_backticks_with_code_tags(issue.get("message", "No message"))
                url = issue.get("url", "No URL provided")
                if code in issue_data:
                    issue_data[code]["Count"] += 1
                else:
                    issue_data[code] = {
                        "Count": 1,
                        "Message": message,
                        "url": url,
                    }

            combined_data = {}
            for code, info in issue_data.items():
                count = info["Count"]
                url = info["url"]
                link = f"<a href='{url}' target='_blank'><code>{code}</code></a>"
                if count in combined_data:
                    combined_data[count]["Codes"].append(link)
                else:
                    combined_data[count] = {
                        "Codes": [link],
                    }
            # Generate the HTML table
            html_content.append(generate_html_table_from_combined_data(combined_data, sort_descending=True))

            for issue in issues:
                code = issue.get("code", "Unknown code")
                message = replace_backticks_with_code_tags(issue.get("message", "No message"))
                location = issue.get("location", {})
                row = location.get("row", None)
                column = location.get("column", None)
                end_location = issue.get("end_location", {})
                # end_row = end_location.get("row", row)
                end_column = end_location.get("column", column)
                fix = issue.get("fix", None)
                url = issue.get("url", "No URL provided")

                html_content.append("<div class='issue'>")
                html_content.append("<p class='issue-header'>")
                html_content.append(
                    f"<strong>Issue:</strong> "
                    f"<a href='{url}' target='_blank'><code>{code}</code></a> - "
                    f"<span class='error'>{message}</span><br>"
                    f"<strong>Location:</strong> "
                    f"<code>{display_name}:{row}:{column}</code><br>",
                )
                html_content.append("</p>")

                # Read the file and extract the code snippet
                if os.path.exists(filename) and row is not None:
                    with open(filename, "r") as source_file:
                        lines = source_file.readlines()
                        total_lines = len(lines)
                        # Adjust indices for zero-based indexing
                        context_radius = 2  # Number of lines before and after the issue line
                        start_line = max(row - context_radius - 1, 0)
                        end_line = min(row + context_radius, total_lines)
                        snippet_lines = lines[start_line:end_line]

                        # Build the code snippet
                        code_lines = []
                        for idx, line_content in enumerate(snippet_lines, start=start_line + 1):
                            line_content = line_content.rstrip("\n")
                            # Fix HTML special characters
                            line_content = line_content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                            # Highlight the error
                            if idx == row and column is not None and end_column is not None:
                                start_col = column - 1  # Adjust for zero-based indexing
                                end_col = end_column - 1

                                start_col = max(start_col, 0)
                                end_col = min(end_col, len(line_content))

                                before = line_content[:start_col]
                                problem = line_content[start_col:end_col]
                                after = line_content[end_col:]
                                # Wrap the problematic part with <mark>
                                highlighted_line = f"{before}<mark>{problem}</mark>{after}"
                                code_lines.append((idx, highlighted_line))
                            else:
                                code_lines.append((idx, line_content))
                        # Make code block with line numbers
                        html_content.append("<pre>")
                        for line_number, line_content in code_lines:
                            html_content.append(
                                # f"<div class='code-line'><span class='line-number'>"
                                # f"{line_number}</span>{line_content}</div>"
                                f"{line_number}:  {line_content}",
                            )
                        html_content.append("</pre>")
                    # Include fix details if available
                    if fix:
                        html_content.append("<div class='fix'>")
                        html_content.append(
                            f"<p>Fix Available (<code>{fix.get('applicability', 'Unknown')}</code>): "
                            f"<code>ruff check --select ALL --fix {display_name}</code></p>",
                        )
                        html_content.append("</div>")
                else:
                    html_content.append(
                        f"<p>Cannot read file <code>{filename}</code> or invalid row <code>{row}</code>.</p>",
                    )

                html_content.append("</div>")
                html_content.append("<hr>")

            html_content.append("</details>")

        # Footer
        html_content.append(
            f"""
<footer>
    <p>Generated by on {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
</footer>
""",
        )

        html_content.extend(["</body>", "</html>"])

        # Write the HTML content to the output file
        with open(html_output_path, "w") as html_file:
            html_file.write("\n".join(html_content))

        print(f"HTML report generated at {html_output_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def construct_models_init_file() -> str:
    """
    Constructs the content for the __init__.py file for the models module.

    Returns:
        str: The content for the __init__.py file as a string.
    """
    import struphy.models.fluid as fluid
    import struphy.models.hybrid as hybrid
    import struphy.models.kinetic as kinetic
    import struphy.models.toy as toy
    from struphy.models.base import StruphyModel

    models_init = ""

    model_names = []
    for model_type in [toy, fluid, hybrid, kinetic]:
        for _, cls in model_type.__dict__.items():
            if isinstance(cls, type) and issubclass(cls, StruphyModel) and cls != StruphyModel:
                model_names.append(cls.__name__)
                models_init += f"from {model_type.__name__} import {cls.__name__}\n"
    models_init += "\n\n"
    models_init += f"__all__ = {model_names}\n"
    return models_init


def construct_propagators_init_file() -> str:
    """
    Constructs the content for the __init__.py file for the propagators module.

    Returns:
        str: The content for the __init__.py file as a string.
    """
    import struphy.propagators.propagators_coupling as propagators_coupling
    import struphy.propagators.propagators_fields as propagators_fields
    import struphy.propagators.propagators_markers as propagators_markers
    from struphy.propagators.base import Propagator

    propagators_init = ""
    propagators_names = []
    for model_type in [propagators_coupling, propagators_fields, propagators_markers]:
        for _, cls in model_type.__dict__.items():
            if isinstance(cls, type) and issubclass(cls, Propagator) and cls != Propagator:
                propagators_names.append(cls.__name__)
                propagators_init += f"from {model_type.__name__} import {cls.__name__}\n"
    propagators_init += "\n\n"
    propagators_init += f"__all__ = {propagators_names}\n"
    return propagators_init
