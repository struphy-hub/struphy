"""Prototype for issue #690: generate a ``numba.cuda`` kernel from the Python source of a pyccel kernel.

Not used by struphy. It exists to compare code generation with the hand-written ``<name>_cuda.cu`` files on one
real kernel (``bstar_parallel_3form``); the result of the comparison is in ``CUDA_STRATEGY.md`` (decision for
#690). The module name contains no ``kernels``, so ``struphy compile`` ignores it.

The generator works on the source files (the imported kernel modules are usually the compiled pyccel extensions):

- it parses the entry kernel's module, follows every call into struphy modules (``import a.b as x`` and
  ``from a.b import f``) and translates each reached function once, as a numba device function named
  ``<module>__<function>``;
- it strips the pyccel decorators and type annotations, maps the NumPy scalar functions to ``math``, and
  ``numpy.mod`` to the Fortran ``MODULO`` pyccel generates;
- ``empty``/``zeros`` become ``cuda.local.array`` with a constant size (a run-time size such as ``zeros(pn)`` gets
  ``MAX_LOCAL`` entries, like the hand-written ``MAX_SPLINE_DEGREE`` arrays);
- array expressions on slices (``eta[:] = alpha[:3] * eta_k + ...``, ``bn[:] = 0.0``) become element loops, since
  numba.cuda cannot allocate the temporaries;
- the entry kernel's marker loop ``for ip in range(n): ...`` becomes ``ip = cuda.grid(1)`` and ``continue`` becomes
  ``return``;
- argument objects (``MarkerArguments``, ``DomainArguments``, ``DerhamArguments``) are passed as the fields of
  their CUDA structs (``CudaMarkerArguments.struct.fields``, one kernel parameter each) and rebuilt as named tuples
  at the top of the kernel; ``DerhamArguments`` gets its scratch arrays ``bn1``, ..., ``bd3`` as per-thread local
  arrays there, so the helpers can stay one to one with pyccel.

With ``target="cpu"`` the same translation is compiled with ``numba.njit`` (NumPy allocations, the marker loop kept),
which checks that the generated code types in nopython mode; numba.cuda itself needs a GPU (or NVVM) to compile, and
its simulator (``NUMBA_ENABLE_CUDASIM=1``) only interprets the Python.
"""

import ast
import importlib.util
import re
import sys
import tempfile
from pathlib import Path

from struphy.kernel_arguments.pusher_args_cuda import CudaDerhamArguments, CudaDomainArguments, CudaMarkerArguments

MAX_LOCAL = 9  # MAX_SPLINE_DEGREE + 1 of bsplines_kernels.cuh

STRUCT_CLASSES = {
    "MarkerArguments": CudaMarkerArguments,
    "DerhamArguments": CudaDerhamArguments,
    "DomainArguments": CudaDomainArguments,
}
# pyccel scratch arrays of the argument classes: per-thread local arrays in the generated kernel
SCRATCH = {"DerhamArguments": ("bn1", "bn2", "bn3", "bd1", "bd2", "bd3")}

NUMPY_MATH = {
    "cos": "cos",
    "sin": "sin",
    "tan": "tan",
    "arctan": "atan",
    "arcsin": "asin",
    "arccos": "acos",
    "arctan2": "atan2",
    "sqrt": "sqrt",
    "exp": "exp",
    "log": "log",
    "floor": "floor",
}
NUMPY_ALLOC = ("empty", "zeros")


def _source_file(module):
    """The ``.py`` file of `module` (the import system would find the compiled pyccel extension first)."""
    spec = importlib.util.find_spec(module)
    return Path(spec.origin).parent / (module.rsplit(".", 1)[-1] + ".py")


class _Module:
    """Functions and imports of one source module."""

    def __init__(self, name):
        self.name = name
        self.short = name.rsplit(".", 1)[-1]
        tree = ast.parse(_source_file(name).read_text())
        self.functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.module_aliases = {}  # alias -> module
        self.imported = {}  # name -> (module, name)
        for node in tree.body:
            if isinstance(node, ast.Import):
                for a in node.names:
                    self.module_aliases[a.asname or a.name] = a.name
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    self.imported[a.asname or a.name] = (node.module, a.name)

    def resolve(self, func):
        """``(module, name)`` of a called function node, or ``None``."""
        if isinstance(func, ast.Name):
            if func.id in self.functions:
                return self.name, func.id
            return self.imported.get(func.id)
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            module = self.module_aliases.get(func.value.id)
            if module is not None:
                return module, func.attr
        return None


def _array_ndim(annotation):
    """Number of dimensions of a pyccel array annotation such as ``"float[:,:]"``, or 0."""
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        match = re.search(r"\[([:,\s]+)\]", annotation.value)
        if match:
            return match.group(1).count(":")
    return 0


def _struct_type(annotation):
    if isinstance(annotation, ast.Constant) and annotation.value in STRUCT_CLASSES:
        return annotation.value
    if isinstance(annotation, ast.Name) and annotation.id in STRUCT_CLASSES:
        return annotation.id
    return None


def _expr(source):
    return ast.parse(source, mode="eval").body


def _stmt(source):
    return ast.parse(source).body


class _Elementwise(ast.NodeTransformer):
    """Index a slice expression at element ``index``: ``a[lo:hi]`` -> ``a[lo + index]``, 1D array ``x`` -> ``x[index]``."""

    def __init__(self, index, arrays):
        self.index = index
        self.arrays = arrays

    def _element(self, slc):
        lower = slc.lower if slc.lower is not None else ast.Constant(0)
        return ast.BinOp(lower, ast.Add(), ast.Name(self.index, ast.Load()))

    def visit_Subscript(self, node):
        if isinstance(node.slice, ast.Slice):
            return ast.Subscript(node.value, self._element(node.slice), node.ctx)
        if isinstance(node.slice, ast.Tuple) and any(isinstance(e, ast.Slice) for e in node.slice.elts):
            elts = [self._element(e) if isinstance(e, ast.Slice) else e for e in node.slice.elts]
            return ast.Subscript(node.value, ast.Tuple(elts, ast.Load()), node.ctx)
        return node

    def visit_Name(self, node):
        if self.arrays.get(node.id) == 1:
            return ast.Subscript(ast.Name(node.id, ast.Load()), ast.Name(self.index, ast.Load()), node.ctx)
        if node.id in self.arrays and self.arrays[node.id] > 1:
            raise NotImplementedError(f"array expression on the {self.arrays[node.id]}D array {node.id}")
        return node


def _slice_length(target):
    """Number of elements written by a sliced assignment target (one slice per target)."""
    slices = target.slice.elts if isinstance(target.slice, ast.Tuple) else [target.slice]
    positions = [i for i, s in enumerate(slices) if isinstance(s, ast.Slice)]
    if len(positions) != 1:
        raise NotImplementedError(f"assignment to {ast.unparse(target)}: exactly one slice is supported")
    slc = slices[positions[0]]
    if slc.step is not None:
        raise NotImplementedError(f"assignment to {ast.unparse(target)}: slice steps are not supported")
    if slc.upper is None:
        upper = ast.Subscript(ast.Attribute(target.value, "shape", ast.Load()), ast.Constant(positions[0]), ast.Load())
    else:
        upper = slc.upper
    if slc.lower is None:
        return upper
    return ast.BinOp(upper, ast.Sub(), slc.lower)


def _has_slice(target):
    if not isinstance(target, ast.Subscript):
        return False
    s = target.slice
    return isinstance(s, ast.Slice) or (isinstance(s, ast.Tuple) and any(isinstance(e, ast.Slice) for e in s.elts))


class _ContinueToReturn(ast.NodeTransformer):
    """``continue`` of the marker loop -> ``return`` (nested loops keep theirs)."""

    def visit_Continue(self, node):
        return ast.Return(None)

    def visit_For(self, node):
        return node

    def visit_While(self, node):
        return node


class Generator:
    """Translate a pyccel entry kernel and every struphy function it calls into one numba module."""

    def __init__(self, target="cuda"):
        assert target in ("cuda", "cpu")
        self.target = target
        self.modules = {}
        self.translated = {}  # (module, name) -> mangled name
        self.sources = []  # translated functions, callees first
        self.counter = 0

    def _module(self, name):
        if name not in self.modules:
            self.modules[name] = _Module(name)
        return self.modules[name]

    def _is_struphy_function(self, ref):
        if ref is None or not ref[0].startswith("struphy."):
            return False
        try:
            return ref[1] in self._module(ref[0]).functions
        except (ModuleNotFoundError, FileNotFoundError, AttributeError):
            return False

    def _numpy_name(self, module, node):
        """The NumPy name a call or name node refers to in `module`, or ``None``."""
        if isinstance(node, ast.Name):
            ref = module.imported.get(node.id)
            if ref is not None and ref[0] == "numpy":
                return ref[1]
        return None

    def _fresh(self, prefix):
        self.counter += 1
        return f"{prefix}{self.counter}"

    # ------------------------------------------------------------------ allocations

    def _allocation(self, module, target, call, arrays):
        kind = self._numpy_name(module, call.func)
        shape = call.args[0]
        ndim = len(shape.elts) if isinstance(shape, ast.Tuple) else 1
        dtype = "int64"
        for kw in call.keywords:
            if kw.arg == "dtype" and isinstance(kw.value, ast.Name) and kw.value.id == "float":
                dtype = "float64"
        if len(call.args) > 1 and isinstance(call.args[1], ast.Name) and call.args[1].id == "float":
            dtype = "float64"
        if not call.keywords and len(call.args) == 1:
            dtype = "float64"
        arrays[target] = ndim
        if self.target == "cpu":
            return _stmt(f"{target} = np.{kind}({ast.unparse(shape)}, dtype=np.{dtype})")
        elts = shape.elts if isinstance(shape, ast.Tuple) else [shape]
        sizes = [e.value if isinstance(e, ast.Constant) else MAX_LOCAL for e in elts]
        size = sizes[0] if ndim == 1 else "(" + ", ".join(map(str, sizes)) + ")"
        stmts = _stmt(f"{target} = cuda.local.array({size}, {dtype})")
        if kind == "zeros":
            loops = "".join(f"{'    ' * d}for _z{d} in range({s}):\n" for d, s in enumerate(sizes))
            index = ", ".join(f"_z{d}" for d in range(ndim))
            stmts += _stmt(f"{loops}{'    ' * ndim}{target}[{index}] = 0")
        return stmts

    # ------------------------------------------------------------------ functions

    def translate(self, module_name, name, entry=False):
        """Translate function `name` of `module_name`; returns its name in the generated module."""
        key = (module_name, name)
        if key in self.translated:
            return self.translated[key]
        module = self._module(module_name)
        mangled = f"{module.short}__{name}"
        self.translated[key] = mangled
        node = ast.parse(ast.unparse(module.functions[name])).body[0]  # a fresh copy

        # parameters: array dimensions and argument objects from the pyccel annotations
        arrays = {a.arg: _array_ndim(a.annotation) for a in node.args.args if _array_ndim(a.annotation)}
        structs = {a.arg: _struct_type(a.annotation) for a in node.args.args if _struct_type(a.annotation)}
        for a in node.args.args:
            a.annotation = None
        node.returns = None
        node.decorator_list = []
        node.name = mangled
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
            node.body = node.body[1:] or [ast.Pass()]

        node.body = self._statements(module, node.body, arrays)

        if entry:
            self._make_entry(node, structs)
            decorator = "@cuda.jit" if self.target == "cuda" else "@njit"
        else:
            decorator = "@cuda.jit(device=True)" if self.target == "cuda" else "@njit"
        self.sources.append(f"{decorator}\n{ast.unparse(ast.fix_missing_locations(node))}\n")
        return mangled

    def _statements(self, module, body, arrays):
        out = []
        for stmt in body:
            out += self._statement(module, stmt, arrays)
        return out

    def _statement(self, module, stmt, arrays):
        # local arrays
        if (
            isinstance(stmt, ast.Assign)
            and isinstance(stmt.value, ast.Call)
            and self._numpy_name(module, stmt.value.func) in NUMPY_ALLOC
        ):
            (target,) = stmt.targets
            return self._allocation(module, target.id, stmt.value, arrays)
        # array expressions on slices -> element loops
        if isinstance(stmt, (ast.Assign, ast.AugAssign)):
            target = stmt.targets[0] if isinstance(stmt, ast.Assign) else stmt.target
            if _has_slice(target):
                index = self._fresh("_k")
                length = _slice_length(target)
                element = _Elementwise(index, arrays)
                value = self._expressions(module, element.visit(stmt.value))
                new_target = element.visit(target)
                if isinstance(stmt, ast.Assign):
                    inner = ast.Assign([new_target], value)
                else:
                    inner = ast.AugAssign(new_target, stmt.op, value)
                loop = ast.For(
                    ast.Name(index, ast.Store()),
                    ast.Call(ast.Name("range", ast.Load()), [length], []),
                    [inner],
                    [],
                    lineno=stmt.lineno,
                )
                return [loop]
        # compound statements: recurse into their bodies
        for field in ("body", "orelse"):
            if hasattr(stmt, field) and isinstance(getattr(stmt, field), list):
                setattr(stmt, field, self._statements(module, getattr(stmt, field), arrays))
        for field in ("value", "test", "iter"):
            if hasattr(stmt, field) and isinstance(getattr(stmt, field), ast.AST):
                setattr(stmt, field, self._expressions(module, getattr(stmt, field)))
        if isinstance(stmt, ast.Assign):
            stmt.targets = [self._expressions(module, t) for t in stmt.targets]
        if isinstance(stmt, ast.AugAssign):
            stmt.target = self._expressions(module, stmt.target)
        return [stmt]

    def _expressions(self, module, node):
        generator = self

        class Calls(ast.NodeTransformer):
            def visit_Call(self, call):
                self.generic_visit(call)
                ref = module.resolve(call.func)
                numpy_name = generator._numpy_name(module, call.func)
                if generator._is_struphy_function(ref):
                    call.func = ast.Name(generator.translate(*ref), ast.Load())
                elif numpy_name in NUMPY_MATH:
                    call.func = _expr(f"math.{NUMPY_MATH[numpy_name]}")
                elif numpy_name == "mod":
                    call.func = ast.Name("_numpy_mod", ast.Load())
                elif numpy_name == "shape":
                    return ast.Attribute(call.args[0], "shape", ast.Load())
                elif numpy_name is not None:
                    raise NotImplementedError(f"numpy.{numpy_name} in {module.name}")
                return call

            def visit_Name(self, name):
                if generator._numpy_name(module, name) == "pi":
                    return _expr("math.pi")
                return name

        return Calls().visit(node)

    def _make_entry(self, node, structs):
        """Flatten the argument objects into struct fields, rebuild them, and map the marker loop to threads."""
        params, prologue = [], []
        for a in node.args.args:
            kind = structs.get(a.arg)
            if kind is None:
                params.append(a.arg)
                continue
            fields = [f.name for f in STRUCT_CLASSES[kind].struct.fields]
            params += [f"{a.arg}__{f}" for f in fields]
            values = [f"{a.arg}__{f}" for f in fields]
            if self.target == "cuda":
                values += [f"cuda.local.array({MAX_LOCAL}, float64)" for _ in SCRATCH.get(kind, ())]
            else:
                values += [f"np.empty({MAX_LOCAL})" for _ in SCRATCH.get(kind, ())]
            prologue += _stmt(f"{a.arg} = {kind}_T({', '.join(values)})")
        node.args.args = [ast.arg(p) for p in params]

        body = prologue + node.body
        if self.target == "cuda":
            for i, stmt in enumerate(body):
                if (
                    isinstance(stmt, ast.For)
                    and isinstance(stmt.iter, ast.Call)
                    and getattr(stmt.iter.func, "id", None) == "range"
                    and len(stmt.iter.args) == 1
                ):
                    ip, n = stmt.target.id, ast.unparse(stmt.iter.args[0])
                    loop_body = [_ContinueToReturn().visit(s) for s in stmt.body]
                    body = body[:i] + _stmt(f"{ip} = cuda.grid(1)\nif {ip} >= {n}:\n    return") + loop_body
                    break
            else:
                raise NotImplementedError("no marker loop 'for ip in range(n)' in the entry kernel")
        node.body = body

    def module_source(self):
        decorator = "@cuda.jit(device=True)" if self.target == "cuda" else "@njit"
        header = [
            f"# Generated by struphy.pic.tests.codegen_spike.numba_codegen (target={self.target}); do not edit.",
            "import math",
            "from collections import namedtuple",
            "",
            "import numpy as np",
            "from numba import cuda, float64, int64, njit",
            "",
        ]
        for kind, cls in STRUCT_CLASSES.items():
            names = [f.name for f in cls.struct.fields] + list(SCRATCH.get(kind, ()))
            header.append(f"{kind}_T = namedtuple({kind + '_T'!r}, {names!r})")
        header += [
            "",
            "",
            decorator,
            "def _numpy_mod(a, b):",
            "    # numpy.mod as pyccel compiles it (Fortran MODULO)",
            "    return a - math.floor(a / b) * b",
            "",
            "",
        ]
        return "\n".join(header) + "\n\n".join(self.sources)


def generate(module_name, name, target="cuda"):
    """Source of the numba module for the pyccel kernel `name` in `module_name` (and every function it calls)."""
    generator = Generator(target)
    generator.translate(module_name, name, entry=True)
    return generator.module_source()


def load(module_name, name, target="cuda", directory=None):
    """Generate, write to `directory` (a temporary one by default) and import; returns the kernel function."""
    source = generate(module_name, name, target)
    directory = Path(directory or tempfile.mkdtemp(prefix="struphy_numba_codegen_"))
    stem = f"generated_{name}_{target}"
    path = directory / f"{stem}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[stem] = module
    spec.loader.exec_module(module)
    return getattr(module, f"{module_name.rsplit('.', 1)[-1]}__{name}"), path


def flatten(args):
    """Kernel arguments with every argument object replaced by the fields of its CUDA struct (host arrays)."""
    from struphy.pic.tests.cuda_emulation import _struct_field

    flat = []
    for value in args:
        kind = type(value).__name__
        if kind in STRUCT_CLASSES:
            flat += [_struct_field(value, f.name) for f in STRUCT_CLASSES[kind].struct.fields]
        else:
            flat.append(value)
    return flat
