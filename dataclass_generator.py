from struphy.io.options import Time, EnvironmentOptions, BaseUnits, DerhamOptions, FieldsBackground

import ast
import inspect


def generate_params_dataclass(cls):
    """Generate a dataclass AST for parameters from a class's __init__."""
    # Get the signature and type hints
    sig = inspect.signature(cls.__init__)
    type_hints = {}
    try:
        import typing
        type_hints = typing.get_type_hints(cls.__init__)
    except Exception:
        # If we can't get type hints, we'll fall back to Any
        pass

    # Build dataclass fields
    fields = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue

        # Try to get the type annotation
        if name in type_hints:
            # Convert type hint to AST
            type_hint = type_hints[name]
            annotation = _type_to_ast(type_hint)
        elif param.annotation is not param.empty:
            # Try to use the annotation directly
            annotation = _annotation_to_ast(param.annotation)
        else:
            # Fall back to Any
            annotation = ast.Name(id="Any", ctx=ast.Load())

        if param.default is not param.empty:
            # Has a default value
            if isinstance(param.default, tuple):
                value = ast.Tuple(
                    elts=[ast.Constant(v) for v in param.default], ctx=ast.Load()
                )
            else:
                value = ast.Constant(param.default)
        else:
            # No default - required field
            value = None

        ann_assign = ast.AnnAssign(
            target=ast.Name(id=name, ctx=ast.Store()),
            annotation=annotation,
            value=value,
            simple=1,
        )
        fields.append(ann_assign)

    # Create the dataclass
    class_def = ast.ClassDef(
        name=f"{cls.__name__}Params",
        bases=[],
        keywords=[],
        body=fields if fields else [ast.Pass()],
        decorator_list=[ast.Name(id="dataclass", ctx=ast.Load())],
    )

    # Create a module with necessary imports and the dataclass
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="dataclasses", names=[ast.alias(name="dataclass", asname=None)], level=0
            ),
            ast.ImportFrom(
                module="typing", names=[ast.alias(name="Any", asname=None)], level=0
            ),
            class_def,
        ],
        type_ignores=[],
    )

    ast.fix_missing_locations(module)
    return ast.unparse(module)


def _type_to_ast(type_hint):
    """Convert a type hint to an AST node."""
    import typing
    
    # Handle basic types
    if type_hint is type(None):
        return ast.Constant(value=None)
    elif hasattr(type_hint, "__name__"):
        # Simple type like int, str, float, bool
        return ast.Name(id=type_hint.__name__, ctx=ast.Load())
    elif hasattr(type_hint, "__origin__"):
        # Generic types like Optional[int], list[str], etc.
        origin = type_hint.__origin__
        
        if origin is typing.Union:
            # Union or Optional
            args = type_hint.__args__
            if len(args) == 2 and type(None) in args:
                # Optional[X]
                inner_type = args[0] if args[1] is type(None) else args[1]
                return ast.Subscript(
                    value=ast.Name(id="Optional", ctx=ast.Load()),
                    slice=_type_to_ast(inner_type),
                    ctx=ast.Load(),
                )
            else:
                # Union[X, Y, ...]
                return ast.Subscript(
                    value=ast.Name(id="Union", ctx=ast.Load()),
                    slice=ast.Tuple(
                        elts=[_type_to_ast(arg) for arg in args],
                        ctx=ast.Load(),
                    ),
                    ctx=ast.Load(),
                )
        elif hasattr(origin, "__name__"):
            # list, tuple, dict, etc.
            args = type_hint.__args__
            return ast.Subscript(
                value=ast.Name(id=origin.__name__, ctx=ast.Load()),
                slice=ast.Tuple(
                    elts=[_type_to_ast(arg) for arg in args],
                    ctx=ast.Load(),
                ) if len(args) > 1 else _type_to_ast(args[0]),
                ctx=ast.Load(),
            )
    
    # Fall back to Any
    return ast.Name(id="Any", ctx=ast.Load())


def _annotation_to_ast(annotation):
    """Convert an annotation object to AST."""
    if isinstance(annotation, type):
        return ast.Name(id=annotation.__name__, ctx=ast.Load())
    elif isinstance(annotation, str):
        # Forward reference
        return ast.Name(id=annotation, ctx=ast.Load())
    else:
        # Try to extract from the annotation
        try:
            return _type_to_ast(annotation)
        except Exception:
            return ast.Name(id="Any", ctx=ast.Load())


if __name__ == "__main__":
    source = generate_params_dataclass(EnvironmentOptions)
    print(source)