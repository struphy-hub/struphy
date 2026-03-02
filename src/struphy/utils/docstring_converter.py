"""Utility to convert RST docstrings to HTML for VS Code display."""

import re


def latex_to_unicode(latex_str: str) -> str:
    """
    Convert LaTeX math expressions to Unicode symbols.
    
    Args:
        latex_str: LaTeX math expression
        
    Returns:
        Unicode representation of the math expression
    """
    # Create a working copy
    result = latex_str
    
    # Bold math symbols (\mathbf)
    bold_letters = {
        'E': '𝐄', 'B': '𝐁', 'A': '𝐀', 'C': '𝐂', 'D': '𝐃', 'F': '𝐅', 
        'G': '𝐆', 'H': '𝐇', 'I': '𝐈', 'J': '𝐉', 'K': '𝐊', 'L': '𝐋',
        'M': '𝐌', 'N': '𝐍', 'O': '𝐎', 'P': '𝐏', 'Q': '𝐐', 'R': '𝐑',
        'S': '𝐒', 'T': '𝐓', 'U': '𝐔', 'V': '𝐕', 'W': '𝐖', 'X': '𝐗',
        'Y': '𝐘', 'Z': '𝐙',
    }
    
    for letter, bold in bold_letters.items():
        result = re.sub(rf'\\mathbf\s*{letter}\b', bold, result)
        result = re.sub(rf'\\mathbf\s*\{{\s*{letter}\s*\}}', bold, result)
    
    # Hat symbols (\hat)
    # Match \hat E or \hat{E}
    def replace_hat(match):
        letter = match.group(1).strip()
        # Combining character for circumflex
        return letter + '\u0302'
    
    result = re.sub(r'\\hat\s*\{([A-Za-z])\}', replace_hat, result)
    result = re.sub(r'\\hat\s+([A-Za-z])\b', replace_hat, result)
    
    # Fractions - handle common patterns
    # \frac{\partial ...}{\partial t} -> ∂.../∂t
    result = re.sub(r'\\frac\{\\partial\s+([^}]+)\}\{\\partial\s+([^}]+)\}', r'∂\1/∂\2', result)
    # \frac{1}{2} -> ½
    result = result.replace(r'\frac{1}{2}', '½')
    # Generic fraction \frac{a}{b} -> a/b
    result = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', result)
    
    # Greek and special symbols
    symbols = {
        r'\nabla': '∇',
        r'\times': '×',
        r'\partial': '∂',
        r'\int': '∫',
        r'\infty': '∞',
        r'\cdot': '·',
        r'\pm': '±',
        r'\mp': '∓',
        r'\leq': '≤',
        r'\geq': '≥',
        r'\neq': '≠',
        r'\approx': '≈',
        r'\equiv': '≡',
        r'\sim': '∼',
        r'\propto': '∝',
        r'\alpha': 'α',
        r'\beta': 'β',
        r'\gamma': 'γ',
        r'\delta': 'δ',
        r'\epsilon': 'ε',
        r'\theta': 'θ',
        r'\lambda': 'λ',
        r'\mu': 'μ',
        r'\nu': 'ν',
        r'\pi': 'π',
        r'\rho': 'ρ',
        r'\sigma': 'σ',
        r'\tau': 'τ',
        r'\phi': 'φ',
        r'\omega': 'ω',
    }
    
    for latex, unicode_sym in symbols.items():
        result = result.replace(latex, unicode_sym)
    
    # Superscripts
    superscripts = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
        '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾',
    }
    
    # Convert ^2, ^{2}, etc.
    def replace_superscript(match):
        content = match.group(1)
        return ''.join(superscripts.get(c, c) for c in content)
    
    result = re.sub(r'\^\{([^}]+)\}', replace_superscript, result)
    result = re.sub(r'\^(\d)', replace_superscript, result)
    
    # Remove remaining LaTeX commands
    result = re.sub(r'\\,', ' ', result)  # thin space
    result = re.sub(r'\\;', ' ', result)  # medium space
    result = re.sub(r'\\quad', ' ', result)
    result = re.sub(r'\\qquad', '  ', result)
    result = re.sub(r'\\left\|', '|', result)
    result = re.sub(r'\\right\|', '|', result)
    result = re.sub(r'\\left', '', result)
    result = re.sub(r'\\right', '', result)
    
    return result.strip()


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

    # Remove code-block sections (.. code-block:: python and following indented content)
    # Extract code blocks first
    code_blocks = []
    
    def save_code_block(match):
        # Extract the indented code content
        code_content = match.group(1)
        # Remove leading spaces (assuming 4-space indent)
        code_lines = code_content.split("\n")
        dedented_lines = []
        for code_line in code_lines:
            if code_line.startswith("    "):
                dedented_lines.append(code_line[4:])
            elif code_line.strip():
                dedented_lines.append(code_line)
            else:
                dedented_lines.append("")
        cleaned_code = "\n".join(dedented_lines).strip()
        code_blocks.append(cleaned_code)
        return f"<!--CODEBLOCK{len(code_blocks) - 1}-->"
    
    # Match .. code-block:: python (or any language) followed by indented content
    html = re.sub(r"\.\. code-block::[^\n]*\n((?:[ \t]+.*\n)*)", save_code_block, html)

    # Extract and convert math blocks
    math_blocks = []
    inline_math_items = []

    def save_math_block(match):
        math_content = match.group(1)
        # Clean up the math (remove leading spaces)
        math_lines = math_content.strip().split("\n")
        latex_str = " ".join(line.strip() for line in math_lines if line.strip())
        # Convert LaTeX to Unicode
        unicode_math = latex_to_unicode(latex_str)
        math_blocks.append(unicode_math)
        # Check if there's a blank line after the math block in the original match
        # If the match ends with newlines, preserve them
        return f"<!--MATHBLOCK{len(math_blocks) - 1}-->"

    def save_inline_math(match):
        latex_str = match.group(1)
        unicode_math = latex_to_unicode(latex_str)
        inline_math_items.append(unicode_math)
        return f"<!--INLINEMATH{len(inline_math_items) - 1}-->"

    # Handle .. math:: blocks (including trailing blank line if present)
    html = re.sub(r"\.\. math::\s*\n\n((?:[ \t]+.*\n)*)\n?", lambda m: save_math_block(m) + "\n", html)

    # Handle inline :math:`...` 
    html = re.sub(r":math:`([^`]+)`", save_inline_math, html)

    # Convert inline code (``code``) BEFORE converting bold
    html = re.sub(r"``([^`]+)``", r"<code>\1</code>", html)

    # Convert :class: references to code
    html = re.sub(r":class:`~?([^`]+)`", r"<code>\1</code>", html)

    # Convert :meth:, :func:, :mod: to code
    html = re.sub(r":(?:meth|func|mod|attr):`~?([^`]+)`", r"<code>\1</code>", html)

    # Now process line by line to handle structure
    lines = html.split("\n")
    result_lines = []
    i = 0
    
    # Track if we're in the first line (summary)
    first_line = True
    in_list = False
    
    while i < len(lines):
        line = lines[i]
        
        # Check for section headers with underlines
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            # Check for RST section markers
            if next_line and all(c in '=-~^"#' for c in next_line.strip()) and len(next_line.strip()) > 0:
                if in_list:
                    result_lines.append("</ul>")
                    result_lines.append("")
                    in_list = False
                level = {"=": 1, "-": 2, "~": 3, "^": 4, '"': 5, "#": 6}.get(next_line.strip()[0], 3)
                result_lines.append("")  # blank line before header
                result_lines.append(f"<h{level}>{line.strip()}</h{level}>")
                first_line = False
                i += 2  # Skip both the title and underline
                continue
        
        # Check for bold section headers (**Text**)
        bold_header_match = re.match(r"^\*\*([^*]+)\*\*\s*$", line.strip())
        if bold_header_match:
            if in_list:
                result_lines.append("</ul>")
                result_lines.append("")
                in_list = False
            result_lines.append("")  # blank line before header
            result_lines.append(f"<h3>{bold_header_match.group(1)}</h3>")
            first_line = False
            i += 1
            continue
        
        # Check for list items
        list_match = re.match(r"^-\s+(.+)$", line)
        if list_match:
            if not in_list:
                result_lines.append("")  # blank line before list
                result_lines.append("<ul>")
                in_list = True
            result_lines.append(f"    <li>{list_match.group(1)}</li>")
            first_line = False
            i += 1
            continue
        
        # Check for numbered list items
        numbered_list_match = re.match(r"^\d+\.\s+(.+)$", line)
        if numbered_list_match:
            if in_list:
                result_lines.append("</ul>")
                result_lines.append("")
                in_list = False
            # Just treat it as a regular list item
            if not in_list:
                result_lines.append("")
                result_lines.append("<ul>")
                in_list = True
            result_lines.append(f"    <li>{numbered_list_match.group(1)}</li>")
            first_line = False
            i += 1
            continue
        
        # Empty line handling
        if not line.strip():
            if in_list:
                result_lines.append("</ul>")
                result_lines.append("")
                in_list = False
            else:
                result_lines.append("")
            first_line = False
            i += 1
            continue
        
        # Regular paragraph text
        if in_list:
            result_lines.append("</ul>")
            result_lines.append("")
            in_list = False
        
        if first_line:
            # First line (summary) - no <p> tags
            result_lines.append(line)
            first_line = False
        else:
            # Multi-line paragraphs - collect all lines until next blank line
            paragraph_lines = [line]
            j = i + 1
            while j < len(lines) and lines[j].strip():
                paragraph_lines.append(lines[j])
                j += 1
            
            # If it's a single line, wrap in <p> tags
            if len(paragraph_lines) == 1:
                result_lines.append(f"<p>{paragraph_lines[0]}</p>")
            else:
                # Check if any line is a math placeholder - if so, wrap each group separately
                has_math_placeholder = any("<!--MATHBLOCK" in l or "<!--INLINEMATH" in l for l in paragraph_lines)
                
                if has_math_placeholder:
                    # Wrap each line in <p> tags separately
                    for para_line in paragraph_lines:
                        result_lines.append(f"<p>{para_line}</p>")
                else:
                    # Multi-line paragraph
                    result_lines.append(f"<p>{paragraph_lines[0]}")
                    for para_line in paragraph_lines[1:]:
                        result_lines.append(para_line)
                    result_lines.append("</p>")
            
            i = j
            continue
        
        i += 1
    
    # Close any open list
    if in_list:
        result_lines.append("</ul>")
    
    html = "\n".join(result_lines)
    
    # Convert bold (**text**) and italic (*text*) within paragraphs
    html = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", html)

    # Restore math blocks as inline code
    for i, unicode_math in enumerate(math_blocks):
        html = html.replace(f"<!--MATHBLOCK{i}-->", f"<code>{unicode_math}</code>")
    
    # Restore inline math as code
    for i, unicode_math in enumerate(inline_math_items):
        html = html.replace(f"<!--INLINEMATH{i}-->", f"<code>{unicode_math}</code>")
    
    # Restore code blocks
    for i, code in enumerate(code_blocks):
        # Escape HTML special characters in code
        code_escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html = html.replace(f"<!--CODEBLOCK{i}-->", f"<pre><code>{code_escaped}</code></pre>")

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


if __name__ == "__main__":
    import argparse
    from struphy.models.utils import get_model_by_name
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name",
        type=str,
        help="Name of the model to convert docstring for",
    )
    args = parser.parse_args()
    model = get_model_by_name(args.model_name)
    auto_convert_docstring(model)
    print(model.__doc__)
