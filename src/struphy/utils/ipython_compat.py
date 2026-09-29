"""Lazy wrappers around ``IPython.display``.

Importing IPython costs about a second, so it is deferred until one of the
wrappers is actually called. If IPython is not installed the wrappers degrade
to no-ops: ``HTML``/``Markdown`` return their argument, ``display`` returns
its first argument.
"""


def HTML(data):
    try:
        from IPython.display import HTML as _HTML
    except ImportError:
        return data
    return _HTML(data)


def Markdown(data):
    try:
        from IPython.display import Markdown as _Markdown
    except ImportError:
        return data
    return _Markdown(data)


def display(*objects, **kwargs):
    try:
        from IPython.display import display as _display
    except ImportError:
        return objects[0] if objects else None
    return _display(*objects, **kwargs)
