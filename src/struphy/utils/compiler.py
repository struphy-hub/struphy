"""Programmatic interface for compiling Struphy kernels, see also the `struphy compile` CLI command."""

from typing import Literal, Optional

from struphy.console.compile import struphy_compile

Language = Literal["fortran", "c"]
CompilerFamily = Literal["GNU", "intel", "PGI", "nvidia", "LLVM"]


class Compiler:
    """Compile Struphy's computational kernels (transpiled to Fortran/C via pyccel).

    This is the programmatic equivalent of the `struphy compile` command line tool.

    Examples
    --------
    >>> from struphy import Compiler
    >>> compiler = Compiler()
    >>> compiler.compile(language="c")
    >>> compiler.status()
    """

    def compile(
        self,
        language: Language = "fortran",
        compiler: CompilerFamily = "GNU",
        compiler_config: Optional[str] = None,
        omp_pic: bool = False,
        omp_feec: bool = False,
        verbose: bool = False,
        time_execution: bool = False,
        yes: bool = False,
    ) -> None:
        """Compile Struphy kernels. All files containing "kernels" are auto-detected and saved to state.yml.

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
        struphy_compile(
            language=language.lower(),
            compiler=compiler,
            compiler_config=compiler_config,
            omp_pic=omp_pic,
            omp_feec=omp_feec,
            delete=False,
            status=False,
            verbose=verbose,
            dependencies=False,
            time_execution=time_execution,
            yes=yes,
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
