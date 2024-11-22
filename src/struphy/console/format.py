"""
Struphy Linting and Formatting Tools

This module provides utilities to lint, format, and analyze Python files in struphy.
It is only accessible when compiling struphy in editable mode.

Functions
---------
parse_path(directory)
    Traverse a directory to find Python files, excluding '__XYZ__.py'.

get_python_files(input_type, path=None)
    Retrieve Python files based on the specified input type (all, path, staged, or branch).

struphy_lint(config, verbose)
    Lint Python files based on the given type, using specified linters.

struphy_format(config, verbose, yes=False)
    Format Python files with specified linters, optionally iterating multiple times.

print_stats_table(stats_list, linters, print_header=True, pathlen=0)
    Print statistics for Python files in a tabular format.

analyze_file(file_path, linters=["isort", "autopep8"], verbose=False)
    Analyze a Python file, reporting on code structure and linter compliance.

print_file_stats(stats)
    Display statistics of a single file in a readable format.

check_isort(file_path, verbose=False)
    Check if a file is sorted according to isort.

check_autopep8(file_path, verbose=False)
    Check if a file is formatted according to autopep8.

check_flake8(file_path, verbose=False)
    Check if a file is formatted according to flake8.

get_pylint_score(file_path, verbose=False, pass_score)
    Get pylint score for a file and determine if it passes.

check_trailing_commas(file_path, verbose=False)
    Check if a file is formatted according to add-trailing-comma.
"""

import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile

import argparse
import json
import pandas as pd
import os
from collections import defaultdict

from tabulate import tabulate

import struphy

LIBPATH = struphy.__path__[0]

GREEN_COLOR = "\033[92m"
RED_COLOR = "\033[91m"
BLACK_COLOR = "\033[0m"

FAIL_RED = f"{RED_COLOR}FAIL{BLACK_COLOR}"
PASS_GREEN = f"{GREEN_COLOR}PASS{BLACK_COLOR}"


def check_ruff(file_path, verbose=False):
    """
    Check if a file passes Ruff linting.

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
    import subprocess

    result = subprocess.run(
        ["ruff", "format", "--check", file_path],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if verbose:
        print("stdout:", result.stdout.decode("utf-8"))
        print("stderr:", result.stderr.decode("utf-8"))
    return result.returncode == 0


def check_isort(file_path, verbose=False):
    """
    Check if a file is sorted according to isort.

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
    """
    Check if a file is formatted according to autopep8.

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
    """
    Check if a file is formatted according to flake8.

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
    """
    Get pylint score for a file and determine if it passes.

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
    """
    Check if a file contains trailing commas as required by add-trailing-commas.

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
    """
    Traverse a directory to find Python files, excluding '__XYZ__.py'.

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
            if filename.endswith(".py") and not re.search(r"__\w+__", filename):
                file_path = os.path.join(root, filename)
                python_files.append(file_path)
    return python_files


def get_python_files(input_type, path=None):
    """
    Retrieve Python files based on the specified input type

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
        if os.path.isfile(path):
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

    python_files = [file for file in python_files if not re.search(r"__\w+__", file)]

    return python_files


def struphy_lint(config, verbose, report=False):
    """
    Lint Python files based on the given configuration and specified linters.

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
    ci_linters = ["ruff"]
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

    if report:
        generate_report(python_files, linters=linters, verbose=verbose)


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
                command = ["ruff", "check", "--preview", "--select", "ALL", "--output-format", "json", "-o", report_json_filename] + python_files
                
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
    """
    Check if any of the specified files still require formatting based on the specified linters.

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


def struphy_format(config, verbose, yes=False):
    """
    Format Python files with specified linters, optionally iterating multiple times.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing the following keys:
            - input_type : str, optional
                The type of files to format ('all', 'path', 'staged', or 'branch'). Defaults to 'all'.
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

    python_files = get_python_files(input_type, path)
    if len(python_files) == 0:
        sys.exit(0)

    confirm_formatting(python_files, linters, yes)

    flags = {
        "autopep8": ["--in-place"],
        "isort": [],
        "add-trailing-comma": ["--exit-zero-even-if-changed"],
        "ruff": [["check", "--fix"], ["format"]],
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
    """
    Print statistics for a single file in plain text format.

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
    """
    Print statistics for Python files in a tabular format.

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

    table = []
    for stats in stats_list:
        path = os.path.relpath(stats["path"])
        row = [
            path,
            stats["num_lines"],
            stats["num_functions"],
            stats["num_classes"],
            stats["num_variables"],
        ]

        if "pylint" in linters:
            headers.append("Pylint #")
            row.append(f"{stats['pylint_score']}/10")
        for linter in linters:
            headers.append(linter)
        headers.append("Passes CI")

        for linter in linters:
            row.append(PASS_GREEN if stats[f"passes_{linter}"] else FAIL_RED)
        if all(linter in linters for linter in ci_linters):
            passes_ci = all(stats[f"passes_{linter}"] for linter in ci_linters)
            row.append(PASS_GREEN if passes_ci else FAIL_RED)
        table.append(row)
    if print_header:
        print(tabulate(table, headers=headers, tablefmt="grid"))
    else:
        lines = tabulate(table, headers=headers, tablefmt="grid").split("\n")
        print("\n".join(lines[-2:]))


def analyze_file(file_path, linters=None, verbose=False):
    """
    Analyze a Python file with list of linters

    Parameters
    ----------
    file_path : str
        Path to the Python file.

    linters : list
        Linters to apply for analysis (default=["isort", "autopep8"]).

    verbose : bool, optional
        If True, enables detailed output (default=False).

    Returns
    -------
    dict
        Analysis results including line count, function count, and linter pass status.
    """

    # We set the default linters here rather than in the function signature to avoid
    # using a mutable list as a default argument, which can lead to unexpected behavior
    # due to shared state across function calls.
    if linters is None:
        linters = ["isort", "autopep8"]

    stats = {
        "path": file_path,
        "num_lines": 0,
        "num_functions": 0,
        "num_classes": 0,
        "num_variables": 0,
        "pylint_score": None,
        "passes_isort": False,
        "passes_autopep8": False,
        "passes_flake8": False,
        "passes_pylint": False,
        "passes_add-trailing-comma": False,
        "passes_ruff": False,
    }

    # Read the file content
    with open(file_path, "r", encoding="utf-8") as file:
        source_code = file.read()
        stats["num_lines"] = len(source_code.splitlines())

    # Parse the AST
    tree = ast.parse(source_code)
    stats["num_functions"] = sum(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
    stats["num_classes"] = sum(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
    stats["num_variables"] = sum(isinstance(node, (ast.Assign, ast.AnnAssign)) for node in ast.walk(tree))

    # Run code analysis tools
    # TODO: This should be a loop
    if "isort" in linters:
        stats["passes_isort"] = check_isort(file_path, verbose=verbose)
    if "autopep8" in linters:
        stats["passes_autopep8"] = check_autopep8(file_path, verbose=verbose)
    if "flake8" in linters:
        stats["passes_flake8"] = check_flake8(file_path, verbose=verbose)
    if "pylint" in linters:
        stats["pylint_score"], stats["passes_pylint"] = get_pylint_score(
            file_path,
            verbose=verbose,
        )
    if "add-trailing-comma" in linters:
        stats["passes_add-trailing-comma"] = check_trailing_commas(
            file_path,
            verbose=verbose,
        )
    if "ruff" in linters:
        stats["passes_ruff"] = check_ruff(
            file_path,
            verbose=verbose,
        )
    return stats

def parse_json_file_to_html(json_file_path, html_output_path):
    """
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
        html_content.append("<!DOCTYPE html>")
        html_content.append("<html lang='en'>")
        html_content.append("<head>")
        html_content.append("<meta charset='UTF-8'>")
        html_content.append("<meta name='viewport' content='width=device-width, initial-scale=1.0'>")
        html_content.append("<title>Code Analysis Report</title>")
        html_content.append("<style>")
        html_content.append(
            """
            body { font-family: Arial, sans-serif; }
            code { font-family: Consolas, 'Courier New', monospace; }
            pre { background-color: #f8f8f8; padding: 10px; border: 1px solid #ddd; }
            .error { color: red; }
            details { margin-bottom: 20px; }
            summary { font-size: 1.2em; font-weight: bold; }
            .issue { margin-bottom: 20px; }
            .issue-header { font-weight: bold; }
            .fix { margin-left: 20px; }
            /* includes alternating gray and white with on-hover color */
            .table-style {
                font-size: 11pt; 
                font-family: Arial;
                border-collapse: collapse; 
                border: 1px solid silver;

            }

            .table-style td, th {
                padding: 5px;
            }

            .table-style tr:nth-child(even) {
                background: #E0E0E0;
            }

            .table-style tr:hover {
                background: silver;
                cursor: pointer;
            }
        """
        )
        html_content.append("</style>")
        html_content.append("</head>")
        html_content.append("<body>")
        html_content.append("<h1>Code Issues Report</h1>")

        for filename, issues in issues_by_file.items():
            print(f"Parsing {filename}")
            # Start foldable section for the file
            html_content.append(f"<details>")
            html_content.append(f"<summary>File: <code>{filename}</code></summary>")

            data = {}
            for issue in issues:
                code = issue.get("code", "Unknown code")
                message = issue.get("message", "No message")
                url = issue.get("url", "No URL provided")
                if code in data:
                    data[code]['count'] += 1
                else:
                    data[code] = {
                        'count':1,
                        'message':message,
                        'url':url,
                    }
            
            combined_data = {}
            for code, info in data.items():
                count = info['count']
                if count in combined_data:
                    combined_data[count]['codes'].append(code)
                else:
                    combined_data[count] = {
                        'codes': [code],
                    }
            df = pd.DataFrame([
                {'Count': count, 'Code(s)': ", ".join(info['codes'])}
                for count, info in combined_data.items()
            ])
            df = df.sort_values('Count', ascending=False)
            # print(df.to_html())
            pd.set_option('colheader_justify', 'center')   # FOR TABLE <th>

            html_string = '''
            <html>
            <head><title>HTML Pandas Dataframe with CSS</title></head>
            <link rel="stylesheet" type="text/css" href="df_style.css"/>
            <body>
                {table}
            </body>
            </html>.
            '''
            html_content.append(html_string.format(table=df.to_html(classes='table-style', index=False)))
            # html_content.append(df.to_html(index=False))
            for issue in issues:
                code = issue.get("code", "Unknown code")
                message = issue.get("message", "No message")
                location = issue.get("location", {})
                row = location.get("row", None)
                column = location.get("column", None)
                end_location = issue.get("end_location", {})
                end_row = end_location.get("row", row)
                end_column = end_location.get("column", column)
                fix = issue.get("fix", None)
                url = issue.get("url", "No URL provided")

                html_content.append("<div class='issue'>")
                html_content.append(f"<p class='issue-header'>")
                html_content.append(f"<a href='{url}' target='_blank'><code>{code}</code></a> <span class='error'>{message}</span><br>")
                html_content.append(f"<code>{filename}:{row}:{column}</code><br>")
                html_content.append("</p>")

                # Read the file and extract the code snippet
                if os.path.exists(filename) and row is not None:
                    with open(filename, "r") as source_file:
                        lines = source_file.readlines()
                        # Adjust indices for zero-based indexing
                        start_line = max(row - 3, 0)
                        end_line = min(row + 2, len(lines))
                        snippet_lines = lines[start_line:end_line]

                        # Build the code snippet with line numbers
                        code_snippet = ""
                        for idx, line_content in enumerate(snippet_lines, start=start_line + 1):
                            line_content = line_content.rstrip("\n")
                            # Escape HTML special characters
                            line_content = line_content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                            # Mark the problematic line
                            if idx == row:
                                if column is not None and end_column is not None and idx == end_row:
                                    # Highlight the problematic part in the line
                                    start_col = column - 1  # Adjust for zero-based indexing
                                    end_col = end_column - 1
                                    # Ensure indices are within the line length
                                    start_col = max(start_col, 0)
                                    end_col = min(end_col, len(line_content))
                                    before = line_content[:start_col]
                                    problem = line_content[start_col:end_col]
                                    after = line_content[end_col:]
                                    highlighted_line = f"{before}<span class='error'>{problem}</span>{after}"
                                    code_snippet += f"{idx}: {highlighted_line}\n"
                                else:
                                    # Highlight the entire line
                                    highlighted_line = f"<span class='error'>{line_content}</span>"
                                    code_snippet += f"{idx}: {highlighted_line}\n"
                            else:
                                code_snippet += f"{idx}: {line_content}\n"

                        html_content.append("<pre><code>")
                        html_content.append(code_snippet)
                        html_content.append("</code></pre>")
                    # Include fix details if available
                    if fix:
                        html_content.append("<div class='fix'>")
                        html_content.append(f"<p>Fix Available (<code>{fix.get('applicability', 'Unknown')}</code>): <code>ruff check --select ALL --fix {filename}</code></p>")
                        html_content.append(f"<p>Applicability: <code>{fix.get('applicability', 'Unknown')}</code></p>")
                        # html_content.append("<p>Edits:</p>")
                        # html_content.append("<ul>")
                        # for edit in fix.get("edits", []):
                        #     edit_content = edit.get("content", "No content")
                        #     edit_content = edit_content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                        #     edit_location = edit.get("location", {})
                        #     edit_row = edit_location.get("row", "Unknown row")
                        #     edit_column = edit_location.get("column", "Unknown column")
                        #     html_content.append("<li>")
                        #     html_content.append(f"Content: <span class='error'>{edit_content}</span><br>")
                        #     html_content.append(
                        #         f"Location: Row <code>{edit_row}</code>, Column <code>{edit_column}</code>"
                        #     )
                        #     html_content.append("</li>")
                        # html_content.append("</ul>")
                        html_content.append("</div>")
                else:
                    html_content.append(
                        f"<p>Cannot read file <code>{filename}</code> or invalid row <code>{row}</code>.</p>"
                    )

                html_content.append("</div>")  # Close issue div
                html_content.append("<hr>")

            html_content.append("</details>")

        html_content.append("</body>")
        html_content.append("</html>")

        # Write the HTML content to the output file
        with open(html_output_path, "w") as html_file:
            html_file.write("\n".join(html_content))

        print(f"HTML report generated at {html_output_path}")

    except FileNotFoundError:
        print(f"Error: File not found at {json_file_path}")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Convert a JSON file to an HTML report.")
#     parser.add_argument(
#         '-i', 
#         '--input', 
#         required=True, 
#         help="Path to the input JSON file."
#     )
#     parser.add_argument(
#         '-o', 
#         '--output', 
#         default='ruff_report.html', 
#         help="Path to the output HTML file (default: ruff_report.html)."
#     )
    
#     args = parser.parse_args()
#     parse_json_file_to_html(args.input, args.output)