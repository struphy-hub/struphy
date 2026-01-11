import ast


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
