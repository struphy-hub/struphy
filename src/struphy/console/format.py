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

from tabulate import tabulate

import struphy

LIBPATH = struphy.__path__[0]

GREEN_COLOR = "\033[92m"
RED_COLOR = "\033[91m"
BLACK_COLOR = "\033[0m"

NO_RED = f"{RED_COLOR}No{BLACK_COLOR}"
YES_GREEN = f"{GREEN_COLOR}Yes{BLACK_COLOR}"


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
            os.path.join(repopath, f)
            for f in files
            if f.endswith(".py") and os.path.isfile(os.path.join(repopath, f))
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
            os.path.join(repopath, f)
            for f in files
            if f.endswith(".py") and os.path.isfile(os.path.join(repopath, f))
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


def struphy_lint(config, verbose):
    """
    Lint Python files based on the given configuration and specified linters.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing the following keys:
            - input_type : str, optional
                The type of files to lint ('all', 'path', 'staged', or 'branch'). Defaults to 'all'.
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
    linters = config.get("linters", [])

    if input_type is None and path is not None:
        input_type = "path"
    # Define standard linters which will be checked in the CI
    ci_linters = ["add-trailing-comma", "isort", "autopep8"]
    python_files = get_python_files(input_type, path)

    print(
        tabulate(
            [[file] for file in python_files],
            headers=[f"The following files will be linted with {linters}"],
        ),
    )
    print("\n")
    max_pathlen = max(len(os.path.relpath(file_path)) for file_path in python_files)
    stats_list = []

    # Check if all ci_linters are included in linters
    if all(ci_linter in linters for ci_linter in ci_linters):
        print("Passes CI if both isort and autopep8 passes")
        check_ci_pass = True
    else:
        skipped_ci_linters = [
            ci_linter for ci_linter in ci_linters if ci_linter not in linters
        ]
        print(
            f'The "Pass CI" check is skipped since not --linters {" ".join(skipped_ci_linters)} is used.',
        )
        check_ci_pass = False
    # Collect statistics for each file
    for ifile, file_path in enumerate(python_files):
        stats = analyze_file(file_path, linters=linters, verbose=verbose)
        stats_list.append(stats)

        # Print the statistics in a table
        print_stats_table(
            [stats],
            linters,
            print_header=(ifile == 0),
            pathlen=max_pathlen,
        )

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
        command = [linter] + flags.get(linter, []) + python_files
        if verbose:
            print(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True)


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

    confirm_formatting(python_files, linters, yes)

    flags = {
        "autopep8": ["--in-place"],
        "isort": [],
        "add-trailing-comma": ["--exit-zero-even-if-changed"],
    }

    if python_files:
        for iteration in range(iterations):
            if verbose:
                print(f"Iteration {iteration + 1}: Running formatters...")

            run_linters_on_files(linters, python_files, flags, verbose)

            # Check if any files still require changes
            if not files_require_formatting(python_files, linters):
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


def print_stats_table(stats_list, linters, print_header=True, pathlen=0):
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
            row.append(YES_GREEN if stats[f"passes_{linter}"] else NO_RED)
        if "isort" in linters and "autopep8" in linters:
            passes_ci = stats["passes_isort"] and stats["passes_autopep8"]
            row.append(YES_GREEN if passes_ci else NO_RED)
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
    }

    # Read the file content
    with open(file_path, "r", encoding="utf-8") as file:
        source_code = file.read()
        stats["num_lines"] = len(source_code.splitlines())

    # Parse the AST
    tree = ast.parse(source_code)
    stats["num_functions"] = sum(
        isinstance(node, ast.FunctionDef) for node in ast.walk(tree)
    )
    stats["num_classes"] = sum(
        isinstance(node, ast.ClassDef) for node in ast.walk(tree)
    )
    stats["num_variables"] = sum(
        isinstance(node, (ast.Assign, ast.AnnAssign)) for node in ast.walk(tree)
    )

    # Run code analysis tools
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
    return stats
