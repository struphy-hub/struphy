"""Programmatic interface for compiling Struphy kernels, see also the `struphy compile` CLI command."""

import json
from typing import Literal, Optional

from struphy.console.compile import struphy_compile

Language = Literal["fortran", "c"]
CompilerFamily = Literal["GNU", "intel", "PGI", "nvidia", "LLVM"]


class Compiler:
    """Compile Struphy's computational kernels (transpiled to Fortran/C via pyccel).

    This is the programmatic equivalent of the `struphy compile` command line tool.
    Options can be set at construction time and/or overridden on each `compile()` call;
    the values last used are kept on the instance and are what `to_dict()`/`to_json()` report.

    Parameters
    ----------
    language : str
        Either "fortran" (default) or "c".

    compiler : str
        Either "GNU" (default), "intel", "PGI", "nvidia", or "LLVM".
        Only "GNU" is regularly tested at the moment.

    compiler_config : str, optional
        Path to a JSON compiler config file. Takes precedence over `compiler` if given.

    omp_pic : bool
        Whether to compile PIC kernels with OpenMP (default=False).

    omp_feec : bool
        Whether to compile FEEC kernels with OpenMP (default=False).

    verbose : bool
        Call pyccel in verbose mode (default=False).

    time_execution : bool
        Print the time spent in each section of the pyccelization (default=False).

    yes : bool
        Automatically answer yes to the prompt when changing the compilation language (default=False).

    Examples
    --------
    >>> from struphy import Compiler
    >>> compiler = Compiler()
    >>> compiler.compile(language="c")
    >>> compiler.status()
    >>> compiler.to_dict()
    """

    def __init__(
        self,
        language: Language = "fortran",
        compiler: CompilerFamily = "GNU",
        compiler_config: Optional[str] = None,
        omp_pic: bool = False,
        omp_feec: bool = False,
        verbose: bool = False,
        time_execution: bool = False,
        yes: bool = False,
    ):
        self.language = language.lower()
        self.compiler = compiler
        self.compiler_config = compiler_config
        self.omp_pic = omp_pic
        self.omp_feec = omp_feec
        self.verbose = verbose
        self.time_execution = time_execution
        self.yes = yes

    def compile(
        self,
        language: Optional[Language] = None,
        compiler: Optional[CompilerFamily] = None,
        compiler_config: Optional[str] = None,
        omp_pic: Optional[bool] = None,
        omp_feec: Optional[bool] = None,
        verbose: Optional[bool] = None,
        time_execution: Optional[bool] = None,
        yes: Optional[bool] = None,
    ) -> None:
        """Compile Struphy kernels. All files containing "kernels" are auto-detected and saved to state.yml.

        Any option left as None keeps the value the instance was constructed with
        (or the value used on the previous `compile()` call).

        Parameters
        ----------
        language : str
            Either "fortran" (default) or "c".

        compiler : str
            Either "GNU" (default), "intel", "PGI", "nvidia", or "LLVM".
            Only "GNU" is regularly tested at the moment.

        compiler_config : str, optional
            Path to a JSON compiler config file. Takes precedence over `compiler` if given.

        omp_pic : bool
            Whether to compile PIC kernels with OpenMP (default=False).

        omp_feec : bool
            Whether to compile FEEC kernels with OpenMP (default=False).

        verbose : bool
            Call pyccel in verbose mode (default=False).

        time_execution : bool
            Print the time spent in each section of the pyccelization (default=False).

        yes : bool
            Automatically answer yes to the prompt when changing the compilation language (default=False).
        """
        if language is not None:
            self.language = language.lower()
        if compiler is not None:
            self.compiler = compiler
        if compiler_config is not None:
            self.compiler_config = compiler_config
        if omp_pic is not None:
            self.omp_pic = omp_pic
        if omp_feec is not None:
            self.omp_feec = omp_feec
        if verbose is not None:
            self.verbose = verbose
        if time_execution is not None:
            self.time_execution = time_execution
        if yes is not None:
            self.yes = yes

        struphy_compile(
            language=self.language,
            compiler=self.compiler,
            compiler_config=self.compiler_config,
            omp_pic=self.omp_pic,
            omp_feec=self.omp_feec,
            delete=False,
            status=False,
            verbose=self.verbose,
            dependencies=False,
            time_execution=self.time_execution,
            yes=self.yes,
        )

    def status(self) -> None:
        """Print the current Struphy compilation status to screen."""
        struphy_compile(
            language="fortran",
            compiler="GNU",
            compiler_config=None,
            omp_pic=False,
            omp_feec=False,
            delete=False,
            status=True,
            verbose=False,
            dependencies=False,
            time_execution=False,
            yes=False,
        )

    def delete(self) -> None:
        """Remove generated Fortran/C and .so files, reverting to pure Python kernels."""
        struphy_compile(
            language="fortran",
            compiler="GNU",
            compiler_config=None,
            omp_pic=False,
            omp_feec=False,
            delete=True,
            status=False,
            verbose=False,
            dependencies=False,
            time_execution=False,
            yes=False,
        )

    def dependencies(self) -> None:
        """Print Struphy kernels (to be compiled) and their dependencies to screen."""
        struphy_compile(
            language="fortran",
            compiler="GNU",
            compiler_config=None,
            omp_pic=False,
            omp_feec=False,
            delete=False,
            status=False,
            verbose=False,
            dependencies=True,
            time_execution=False,
            yes=False,
        )

    def to_dict(self) -> dict:
        """Serialize the compiler options currently set on this instance to a dictionary."""
        return {
            "language": self.language,
            "compiler": self.compiler,
            "compiler_config": self.compiler_config,
            "omp_pic": self.omp_pic,
            "omp_feec": self.omp_feec,
            "verbose": self.verbose,
            "time_execution": self.time_execution,
            "yes": self.yes,
        }

    def to_json(self, file_path: str = None) -> str:
        """Serialize the compiler options currently set on this instance to a JSON string.

        Parameters
        ----------
        file_path : str, optional
            If given, also write the JSON string to this file.

        Returns
        -------
        str
            The JSON-encoded compiler options.
        """
        json_str = json.dumps(self.to_dict(), indent=4)
        if file_path is not None:
            with open(file_path, "w") as f:
                f.write(json_str)
        return json_str
