"""Sphinx extension to use __doc_rst__ attribute for RST documentation."""


def process_docstring(app, what, name, obj, options, lines):
    """Replace docstring with __doc_rst__ if available."""
    if hasattr(obj, '__doc_rst__'):
        # Clear existing lines
        lines.clear()
        # Add lines from __doc_rst__
        rst_doc = obj.__doc_rst__
        if rst_doc:
            for line in rst_doc.split('\n'):
                lines.append(line)


def setup(app):
    """Setup the extension."""
    app.connect('autodoc-process-docstring', process_docstring)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
