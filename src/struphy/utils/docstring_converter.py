"""Utility to convert RST docstrings to HTML for VS Code display."""

import re
from typing import Optional


def rst_to_html(rst_text: str) -> str:
    """
    Convert RST docstring to HTML for VS Code/Pylance display.

    This is a lightweight converter that handles common RST patterns without
    requiring the full docutils library.

    Args:
        rst_text: RST formatted text

    Returns:
        HTML formatted text
    """
    if not rst_text:
        return ""

    html = rst_text

    # Extract and convert math blocks
    math_blocks = []

    def save_math(match):
        math_blocks.append(match.group(1))
        return f"<!--MATH{len(math_blocks) - 1}-->"

    # Handle .. math:: blocks
    html = re.sub(r"\.\. math::\s*\n\n((?:[ \t]+.*\n)*)", save_math, html)

    # Convert bold (**text**)
    html = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", html)

    # Convert italic (*text*)
    html = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", html)

    # Convert inline code (``code``)
    html = re.sub(r"``([^`]+)``", r"<code>\1</code>", html)

    # Convert :class: references to code
    html = re.sub(r":class:`~?([^`]+)`", r"<code>\1</code>", html)

    # Convert :ref: references to bold
    html = re.sub(r":ref:`([^`]+)`", r"<strong>\1</strong>", html)

    # Convert :meth:, :func:, :mod: to code
    html = re.sub(r":(?:meth|func|mod|attr):`~?([^`]+)`", r"<code>\1</code>", html)

    # Restore math blocks as code blocks
    for i, math in enumerate(math_blocks):
        # Clean up the math (remove leading spaces)
        math_lines = math.strip().split("\n")
        cleaned_math = "\n".join(line.strip() for line in math_lines if line.strip())
        html = html.replace(f"<!--MATH{i}-->", f"<pre><code>{cleaned_math}</code></pre>")

    # Convert section headers (words followed by underlines)
    lines = html.split("\n")
    result_lines = []
    i = 0
    while i < len(lines):
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            # Check for RST section markers
            if next_line and all(c in '=-~^"#' for c in next_line.strip()) and len(next_line.strip()) > 0:
                level = {"=": 1, "-": 2, "~": 3, "^": 4, '"': 5, "#": 6}.get(next_line.strip()[0], 3)
                result_lines.append(f"<h{level}>{lines[i].strip()}</h{level}>")
                i += 2  # Skip both the title and underline
                continue
        result_lines.append(lines[i])
        i += 1

    html = "\n".join(result_lines)

    # Wrap paragraphs (simple approach)
    html = re.sub(r"\n\n+", "\n<p></p>\n", html)

    return html


def rst_to_markdown(rst_text: str) -> str:
    """
    Convert RST docstring to Markdown for Jupyter notebook display.

    Args:
        rst_text: RST formatted text

    Returns:
        Markdown formatted text
    """
    if not rst_text:
        return ""

    md = rst_text

    # Extract and convert math blocks
    math_blocks = []

    def save_math(match):
        math_content = match.group(1)
        # Clean up the math (remove leading spaces)
        math_lines = math_content.strip().split("\n")
        cleaned_math = "\n".join(line.strip() for line in math_lines if line.strip())
        math_blocks.append(cleaned_math)
        return f"<!--MATH{len(math_blocks) - 1}-->"

    # Handle .. math:: blocks
    md = re.sub(r"\.\. math::\s*\n\n((?:[ \t]+.*\n)*)", save_math, md)

    # Convert bold (**text**) - already markdown compatible
    # Convert italic (*text*) - already markdown compatible

    # Convert inline code (``code``) to `code`
    md = re.sub(r"``([^`]+)``", r"`\1`", md)

    # Convert :class: references to code
    md = re.sub(r":class:`~?([^`]+)`", r"`\1`", md)

    # Convert :ref: references to bold
    md = re.sub(r":ref:`([^`]+)`", r"**\1**", md)

    # Convert :meth:, :func:, :mod: to code
    md = re.sub(r":(?:meth|func|mod|attr):`~?([^`]+)`", r"`\1`", md)

    # Restore math blocks as LaTeX math
    for i, math in enumerate(math_blocks):
        md = md.replace(f"<!--MATH{i}-->", f"$$\n{math}\n$$")

    return md


def render_docstring(obj, use_rst=False):
    """
    Render a class or function's docstring in a Jupyter notebook.

    This function returns an IPython display object that will render
    the docstring with proper formatting in Jupyter notebooks.

    Args:
        obj: Class or function whose docstring to display
        use_rst: If True and __doc_rst__ exists, use that instead of __doc__

    Returns:
        IPython.display object for rendering in Jupyter

    Examples:
        >>> from struphy.models.maxwell import Maxwell
        >>> from struphy.utils.docstring_converter import render_docstring
        >>> render_docstring(Maxwell)  # Shows HTML version
        >>> render_docstring(Maxwell, use_rst=True)  # Shows RST as Markdown
    """
    try:
        from IPython.display import HTML, Markdown, display
    except ImportError:
        print("IPython not available. Install jupyter to use this feature.")
        return None

    # Determine which docstring to use
    if use_rst and hasattr(obj, "__doc_rst__"):
        doc_text = obj.__doc_rst__
        # Convert RST to Markdown for better Jupyter rendering
        md_text = rst_to_markdown(doc_text)
        return Markdown(md_text)
    elif hasattr(obj, "__doc__") and obj.__doc__:
        # Check if it's HTML (contains tags)
        doc_text = obj.__doc__
        if "<" in doc_text and ">" in doc_text:
            # It's HTML
            return HTML(doc_text)
        else:
            # Plain text or RST, show as is
            return Markdown(doc_text)
    else:
        return Markdown("*No docstring available*")


def auto_convert_docstring(cls):
    """
    Decorator/hook to automatically convert __doc_rst__ to __doc__ (HTML).

    If a class has a __doc_rst__ attribute, this converts it to HTML
    and sets it as the class docstring for VS Code display.
    """
    if hasattr(cls, "__doc_rst__") and cls.__doc_rst__:
        # Convert RST to HTML
        html_doc = rst_to_html(cls.__doc_rst__)
        # Set as the main docstring
        cls.__doc__ = html_doc
    return cls
