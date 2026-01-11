import ast

import inspect


def import_from(module, names):
    return ast.ImportFrom(
        module=module,
        names=[ast.alias(name=n, asname=None) for n in names],
        level=0,
    )


def assign_constructor(var, cls, **kwargs):
    """Create AST for: var = cls(**kwargs)"""

    def ast_value(v):
        if isinstance(v, tuple):
            return ast.Tuple(elts=[ast_value(x) for x in v], ctx=ast.Load())
        return ast.Constant(v)

    return ast.Assign(
        targets=[ast.Name(id=var, ctx=ast.Store())],
        value=ast.Call(
            func=ast.Name(id=cls, ctx=ast.Load()),
            args=[],
            keywords=[
                ast.keyword(arg=k, value=ast_value(v)) for k, v in kwargs.items()
            ],
        ),
    )


def assign_constructor_with_defaults(
    var_name, cls_or_str, cls_name_for_ast=None, **overrides
):
    """Create AST for: var_name = cls(**all_defaults_and_overrides)

    Args:
        var_name: Variable name for the assignment
        cls_or_str: Class object to introspect for defaults, OR a string class name if no introspection needed
        cls_name_for_ast: Optional string for how to reference the class in AST (e.g., "domains.Cuboid")
                          If not provided, will use cls.__name__ or cls_or_str
        **overrides: Keyword arguments that override the defaults
    """
    keywords = []

    # If we have a class object, introspect it for defaults
    if not isinstance(cls_or_str, str):
        cls = cls_or_str
        sig = inspect.signature(cls.__init__)

        for name, param in sig.parameters.items():
            if name == "self":
                continue

            # Use override if provided, otherwise use default
            if name in overrides:
                value = overrides[name]
            elif param.default is not param.empty:
                value = param.default
            else:
                # No default and no override - skip or set to None
                continue

            # Convert to AST constant or tuple if needed
            if isinstance(value, tuple):
                ast_value = ast.Tuple(
                    elts=[ast.Constant(v) for v in value], ctx=ast.Load()
                )
            else:
                ast_value = ast.Constant(value)
            keywords.append(ast.keyword(arg=name, value=ast_value))

        # Determine the class name for AST
        if cls_name_for_ast is None:
            cls_name_for_ast = cls.__name__
    else:
        # It's just a string, use the overrides
        cls_name_for_ast = cls_or_str
        for name, value in overrides.items():
            if isinstance(value, tuple):
                ast_value = ast.Tuple(
                    elts=[ast.Constant(v) for v in value], ctx=ast.Load()
                )
            else:
                ast_value = ast.Constant(value)
            keywords.append(ast.keyword(arg=name, value=ast_value))

    # Build the function AST node from the class name string
    if "." in cls_name_for_ast:
        parts = cls_name_for_ast.split(".")
        func = ast.Name(id=parts[0], ctx=ast.Load())
        for part in parts[1:]:
            func = ast.Attribute(value=func, attr=part, ctx=ast.Load())
    else:
        func = ast.Name(id=cls_name_for_ast, ctx=ast.Load())

    return ast.Assign(
        targets=[ast.Name(id=var_name, ctx=ast.Store())],
        value=ast.Call(func=func, args=[], keywords=keywords),
    )


def call_attr(obj, attr, args=None, keywords=None):
    return ast.Expr(
        value=ast.Call(
            func=ast.Attribute(value=obj, attr=attr, ctx=ast.Load()),
            args=args or [],
            keywords=keywords or [],
        )
    )


def attr_chain(names, ctx=ast.Load()):
    """Create a nested Attribute node from a list: e.g., model.em_fields.b_field"""
    node = ast.Name(id=names[0], ctx=ctx)
    for name in names[1:]:
        node = ast.Attribute(value=node, attr=name, ctx=ctx)
    return node
