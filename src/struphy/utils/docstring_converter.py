"""Utility to convert RST docstrings to HTML for VS Code display."""

import logging
import re

logger = logging.getLogger("struphy")


def _format_fraction(numerator: str, denominator: str, display_mode: bool = False) -> str:
    """Format a fraction without using plain '/' text."""
    num = numerator.strip()
    den = denominator.strip()
    if display_mode:
        # LaTeX-like stacked fraction for display equations.
        return (
            '<span style="display:inline-flex;flex-direction:column;align-items:center;'
            'vertical-align:middle;line-height:1;margin:0 0.08em;">'
            f'<span style="display:block;padding:0 0.18em;border-bottom:1px solid currentColor;">{num}</span>'
            f'<span style="display:block;padding:0 0.18em;">{den}</span>'
            "</span>"
        )

    # Inline: keep lightweight typography.
    return f"<sup>{num}</sup>⁄<sub>{den}</sub>"


def latex_to_unicode(latex_str: str, display_mode: bool = False) -> str:
    """
    Convert LaTeX math expressions to Unicode symbols.

    Args:
        latex_str: LaTeX math expression

    Returns:
        Unicode representation of the math expression
    """
    # Create a working copy
    result = latex_str

    # FIRST: Handle nested tilde/hat with mathbf using HTML bold tags
    # \tilde{\mathbf{U}} -> <b>Ũ</b> (use regular letter with combining mark + HTML bold)
    # This renders better than mathematical bold + combining mark
    tilde_char = "\u0303"  # combining tilde
    hat_char = "\u0302"  # combining circumflex

    # Match \tilde{\mathbf{X}} or \hat{\mathbf{X}}
    result = re.sub(r"\\tilde\s*\{\\mathbf\s*\{([A-Za-z])\}\}", lambda m: f"<b>{m.group(1)}{tilde_char}</b>", result)
    result = re.sub(r"\\tilde\s*\{\\mathbf\s+([A-Za-z])\}", lambda m: f"<b>{m.group(1)}{tilde_char}</b>", result)
    result = re.sub(r"\\hat\s*\{\\mathbf\s*\{([A-Za-z])\}\}", lambda m: f"<b>{m.group(1)}{hat_char}</b>", result)
    result = re.sub(r"\\hat\s*\{\\mathbf\s+([A-Za-z])\}", lambda m: f"<b>{m.group(1)}{hat_char}</b>", result)

    # Bold math symbols (\mathbf) - for letters WITHOUT combining marks
    bold_letters = {
        # Uppercase
        "A": "𝐀",
        "B": "𝐁",
        "C": "𝐂",
        "D": "𝐃",
        "E": "𝐄",
        "F": "𝐅",
        "G": "𝐆",
        "H": "𝐇",
        "I": "𝐈",
        "J": "𝐉",
        "K": "𝐊",
        "L": "𝐋",
        "M": "𝐌",
        "N": "𝐍",
        "O": "𝐎",
        "P": "𝐏",
        "Q": "𝐐",
        "R": "𝐑",
        "S": "𝐒",
        "T": "𝐓",
        "U": "𝐔",
        "V": "𝐕",
        "W": "𝐖",
        "X": "𝐗",
        "Y": "𝐘",
        "Z": "𝐙",
        # Lowercase
        "a": "𝐚",
        "b": "𝐛",
        "c": "𝐜",
        "d": "𝐝",
        "e": "𝐞",
        "f": "𝐟",
        "g": "𝐠",
        "h": "𝐡",
        "i": "𝐢",
        "j": "𝐣",
        "k": "𝐤",
        "l": "𝐥",
        "m": "𝐦",
        "n": "𝐧",
        "o": "𝐨",
        "p": "𝐩",
        "q": "𝐪",
        "r": "𝐫",
        "s": "𝐬",
        "t": "𝐭",
        "u": "𝐮",
        "v": "𝐯",
        "w": "𝐰",
        "x": "𝐱",
        "y": "𝐲",
        "z": "𝐳",
    }

    for letter, bold in bold_letters.items():
        # Match \mathbf E or \mathbf{E} (without combining marks)
        result = re.sub(rf"\\mathbf\s*{re.escape(letter)}\b", bold, result)
        result = re.sub(rf"\\mathbf\s*\{{\s*{re.escape(letter)}\s*\}}", bold, result)

    # Roman/upright math symbols (\mathrm)
    # For \mathrm, we just remove the command and keep the letter
    result = re.sub(r"\\mathrm\s*\{\s*([A-Za-z0-9]+)\s*\}", r"\1", result)
    result = re.sub(r"\\mathrm\s+([A-Za-z0-9])\b", r"\1", result)

    # Blackboard bold (\mathbb) - common mathematical sets
    mathbb_letters = {
        "A": "𝔸",
        "B": "𝔹",
        "C": "ℂ",
        "D": "𝔻",
        "E": "𝔼",
        "F": "𝔽",
        "G": "𝔾",
        "H": "ℍ",
        "I": "𝕀",
        "J": "𝕁",
        "K": "𝕂",
        "L": "𝕃",
        "M": "𝕄",
        "N": "ℕ",
        "O": "𝕆",
        "P": "ℙ",
        "Q": "ℚ",
        "R": "ℝ",
        "S": "𝕊",
        "T": "𝕋",
        "U": "𝕌",
        "V": "𝕍",
        "W": "𝕎",
        "X": "𝕏",
        "Y": "𝕐",
        "Z": "ℤ",
    }

    for letter, bb in mathbb_letters.items():
        result = re.sub(rf"\\mathbb\s*{re.escape(letter)}\b", bb, result)
        result = re.sub(rf"\\mathbb\s*\{{\s*{re.escape(letter)}\s*\}}", bb, result)

    # Script/calligraphic math symbols (\mathcal)
    mathcal_letters = {
        "A": "𝒜",
        "B": "ℬ",
        "C": "𝒞",
        "D": "𝒟",
        "E": "ℰ",
        "F": "ℱ",
        "G": "𝒢",
        "H": "ℋ",
        "I": "ℐ",
        "J": "𝒥",
        "K": "𝒦",
        "L": "ℒ",
        "M": "ℳ",
        "N": "𝒩",
        "O": "𝒪",
        "P": "𝒫",
        "Q": "𝒬",
        "R": "ℛ",
        "S": "𝒮",
        "T": "𝒯",
        "U": "𝒰",
        "V": "𝒱",
        "W": "𝒲",
        "X": "𝒳",
        "Y": "𝒴",
        "Z": "𝒵",
    }

    for letter, sc in mathcal_letters.items():
        result = re.sub(rf"\\mathcal\s*{re.escape(letter)}\b", sc, result)
        result = re.sub(rf"\\mathcal\s*\{{\s*{re.escape(letter)}\s*\}}", sc, result)

    # Hat symbols (\hat)
    # Match \hat E or \hat{E} or \hat{\mathbf{E}}
    def replace_hat(match):
        content = match.group(1).strip()
        # Combining character for circumflex
        return content + "\u0302"

    result = re.sub(r"\\hat\s*\{([^}]+)\}", replace_hat, result)
    result = re.sub(r"\\hat\s+([A-Za-z])\b", replace_hat, result)

    # Tilde symbols (\tilde)
    # Match \tilde E or \tilde{E} or \tilde{\mathbf{E}}
    def replace_tilde(match):
        content = match.group(1).strip()
        # Combining character for tilde
        return content + "\u0303"

    result = re.sub(r"\\tilde\s*\{([^}]+)\}", replace_tilde, result)
    result = re.sub(r"\\tilde\s+([A-Za-z])\b", replace_tilde, result)

    # Fractions - handle FIRST (before sqrt) to process fractions inside sqrt
    # \frac{\partial ...}{\partial t} -> typographic fraction
    result = re.sub(
        r"\\frac\{\\partial\s+([^}]+)\}\{\\partial\s+([^}]+)\}",
        lambda m: _format_fraction(f"∂{m.group(1)}", f"∂{m.group(2)}", display_mode=display_mode),
        result,
    )

    # Common fractions
    common_fractions = {
        r"\frac{1}{2}": "½",
        r"\frac{1}{3}": "⅓",
        r"\frac{2}{3}": "⅔",
        r"\frac{1}{4}": "¼",
        r"\frac{3}{4}": "¾",
        r"\frac{1}{5}": "⅕",
        r"\frac{2}{5}": "⅖",
        r"\frac{3}{5}": "⅗",
        r"\frac{4}{5}": "⅘",
        r"\frac{1}{6}": "⅙",
        r"\frac{5}{6}": "⅚",
        r"\frac{1}{8}": "⅛",
        r"\frac{3}{8}": "⅜",
        r"\frac{5}{8}": "⅝",
        r"\frac{7}{8}": "⅞",
    }
    if not display_mode:
        for frac, unicode_frac in common_fractions.items():
            result = result.replace(frac, unicode_frac)

    # Generic fraction \frac{a}{b} -> typographic fraction
    # Handle simple nested cases with multiple passes
    for _ in range(3):  # Multiple passes for nested fractions
        result = re.sub(
            r"\\frac\{([^{}]+)\}\{([^{}]+)\}",
            lambda m: _format_fraction(m.group(1), m.group(2), display_mode=display_mode),
            result,
        )

    # Square root (\sqrt) - handle AFTER initial fractions so fractions inside sqrt are processed first
    def replace_sqrt(match):
        content = match.group(1).strip()
        return f"√({content})"

    result = re.sub(r"\\sqrt\s*\{([^}]+)\}", replace_sqrt, result)

    # Process fractions AGAIN to catch fractions created by sqrt conversion
    # (e.g., \frac{a}{\sqrt{b}} becomes \frac{a}{√(b)} which can now be matched)
    for _ in range(3):  # Multiple passes for nested fractions
        result = re.sub(
            r"\\frac\{([^{}]+)\}\{([^{}]+)\}",
            lambda m: _format_fraction(m.group(1), m.group(2), display_mode=display_mode),
            result,
        )

    # Normalize subscript patterns: _\command{...} -> _{\command{...}}
    # Do this BEFORE symbol replacement so we can handle _\mathbb{R}, _\Omega, etc.
    result = re.sub(r"_(\\[a-zA-Z]+\{[^}]+\})", r"_{\1}", result)

    # Normalize superscript patterns: ^\command{...} -> ^{\command{...}}
    result = re.sub(r"\^(\\[a-zA-Z]+\{[^}]+\})", r"^{\1}", result)

    # Greek and special symbols
    symbols = {
        r"\sum": "∑",
        r"\nabla": "∇",
        r"\times": "×",
        r"\partial": "∂",
        r"\int": "∫",
        r"\infty": "∞",
        r"\cdot": "·",
        r"\pm": "±",
        r"\mp": "∓",
        r"\leq": "≤",
        r"\geq": "≥",
        r"\neq": "≠",
        r"\approx": "≈",
        r"\equiv": "≡",
        r"\sim": "∼",
        r"\propto": "∝",
        r"\top": "ᵀ",  # transpose symbol
        r"\in": "∈",
        r"\notin": "∉",
        r"\forall": "∀",
        r"\exists": "∃",
        # Greek letters
        r"\alpha": "α",
        r"\beta": "β",
        r"\gamma": "γ",
        r"\delta": "δ",
        r"\epsilon": "ε",
        r"\varepsilon": "ε",
        r"\theta": "θ",
        r"\vartheta": "ϑ",
        r"\lambda": "λ",
        r"\mu": "μ",
        r"\nu": "ν",
        r"\pi": "π",
        r"\varpi": "ϖ",
        r"\rho": "ρ",
        r"\varrho": "ϱ",
        r"\sigma": "σ",
        r"\varsigma": "ς",
        r"\tau": "τ",
        r"\phi": "φ",
        r"\varphi": "φ",
        r"\omega": "ω",
        r"\zeta": "ζ",
        r"\eta": "η",
        r"\iota": "ι",
        r"\kappa": "κ",
        r"\xi": "ξ",
        r"\omicron": "ο",
        r"\upsilon": "υ",
        r"\chi": "χ",
        r"\psi": "ψ",
        # Capital Greek letters
        r"\Alpha": "Α",
        r"\Beta": "Β",
        r"\Gamma": "Γ",
        r"\Delta": "Δ",
        r"\Epsilon": "Ε",
        r"\Zeta": "Ζ",
        r"\Eta": "Η",
        r"\Theta": "Θ",
        r"\Iota": "Ι",
        r"\Kappa": "Κ",
        r"\Lambda": "Λ",
        r"\Mu": "Μ",
        r"\Nu": "Ν",
        r"\Xi": "Ξ",
        r"\Omicron": "Ο",
        r"\Pi": "Π",
        r"\Rho": "Ρ",
        r"\Sigma": "Σ",
        r"\Tau": "Τ",
        r"\Upsilon": "Υ",
        r"\Phi": "Φ",
        r"\Chi": "Χ",
        r"\Psi": "Ψ",
        r"\Omega": "Ω",
    }

    for latex, unicode_sym in symbols.items():
        result = result.replace(latex, unicode_sym)

    # Subscripts and Superscripts - handle with better heuristics
    subscripts = {
        "0": "₀",
        "1": "₁",
        "2": "₂",
        "3": "₃",
        "4": "₄",
        "5": "₅",
        "6": "₆",
        "7": "₇",
        "8": "₈",
        "9": "₉",
        "+": "₊",
        "-": "₋",
        "=": "₌",
        "(": "₍",
        ")": "₎",
        "a": "ₐ",
        "e": "ₑ",
        "h": "ₕ",
        "i": "ᵢ",
        "j": "ⱼ",
        "k": "ₖ",
        "l": "ₗ",
        "m": "ₘ",
        "n": "ₙ",
        "o": "ₒ",
        "p": "ₚ",
        "r": "ᵣ",
        "s": "ₛ",
        "t": "ₜ",
        "u": "ᵤ",
        "v": "ᵥ",
        "x": "ₓ",
    }

    superscripts = {
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹",
        "+": "⁺",
        "-": "⁻",
        "=": "⁼",
        "(": "⁽",
        ")": "⁾",
        "a": "ᵃ",
        "b": "ᵇ",
        "c": "ᶜ",
        "d": "ᵈ",
        "e": "ᵉ",
        "f": "ᶠ",
        "g": "ᵍ",
        "h": "ʰ",
        "i": "ⁱ",
        "j": "ʲ",
        "k": "ᵏ",
        "l": "ˡ",
        "m": "ᵐ",
        "n": "ⁿ",
        "o": "ᵒ",
        "p": "ᵖ",
        "r": "ʳ",
        "s": "ˢ",
        "t": "ᵗ",
        "u": "ᵘ",
        "v": "ᵛ",
        "w": "ʷ",
        "x": "ˣ",
        "y": "ʸ",
        "z": "ᶻ",
        "A": "ᴬ",
        "B": "ᴮ",
        "D": "ᴰ",
        "E": "ᴱ",
        "G": "ᴳ",
        "H": "ᴴ",
        "I": "ᴵ",
        "J": "ᴶ",
        "K": "ᴷ",
        "L": "ᴸ",
        "M": "ᴹ",
        "N": "ᴺ",
        "O": "ᴼ",
        "P": "ᴾ",
        "R": "ᴿ",
        "T": "ᵀ",
        "U": "ᵁ",
        "V": "ⱽ",
        "W": "ᵂ",
    }

    # Convert _{...} subscripts with smarter handling
    def replace_subscript(match):
        content = match.group(1).strip()
        # Check if all characters can be converted to Unicode subscripts
        converted = "".join(subscripts.get(c, "") for c in content)

        if converted and len(converted) == len(content):
            # All characters have Unicode subscript equivalents
            return converted
        elif len(content) == 1 and content in subscripts:
            # Single character with Unicode equivalent
            return subscripts[content]
        elif len(content) <= 2 and all(c in subscripts for c in content):
            # Short sequence of convertible characters
            return "".join(subscripts[c] for c in content)
        else:
            # Fallback for characters without Unicode subscript glyphs
            # (e.g. uppercase letters like U in E_U)
            return f"<sub>{content}</sub>"

    result = re.sub(r"_\{([^}]+)\}", replace_subscript, result)
    # Handle multi-character unbraced subscripts before single-character ones
    result = re.sub(r"_([A-Za-z0-9]+)(?![A-Za-z0-9])", replace_subscript, result)
    result = re.sub(r"_([A-Za-z0-9])(?![A-Za-z0-9])", replace_subscript, result)

    # Convert ^{...} superscripts with smarter handling
    def replace_superscript(match):
        content = match.group(1).strip()
        # Check if all characters can be converted to Unicode superscripts
        converted = "".join(superscripts.get(c, "") for c in content)

        if converted and len(converted) == len(content):
            # All characters have Unicode superscript equivalents
            return converted
        elif len(content) == 1 and content in superscripts:
            # Single character with Unicode equivalent
            return superscripts[content]
        elif len(content) <= 2 and all(c in superscripts for c in content):
            # Short sequence of convertible characters
            return "".join(superscripts[c] for c in content)
        else:
            # Fallback for characters without Unicode superscript glyphs
            return f"<sup>{content}</sup>"

    result = re.sub(r"\^\{([^}]+)\}", replace_superscript, result)
    # Handle multi-character unbraced superscripts before single-character ones
    result = re.sub(r"\^([A-Za-z0-9]+)(?![A-Za-z0-9])", replace_superscript, result)
    result = re.sub(r"\^([A-Za-z0-9])(?![A-Za-z0-9])", replace_superscript, result)
    # Remove remaining LaTeX commands
    # Preserve spacing intent using Unicode space characters (HTML-safe)
    result = re.sub(r"\\,", chr(0x2009), result)  # thin space
    result = re.sub(r"\\;", chr(0x2005), result)  # medium mathematical space
    result = re.sub(r"\\quad", chr(0x2003), result)  # em space
    result = re.sub(r"\\qquad", chr(0x2003) * 2, result)  # double em space
    result = re.sub(r"\\left\|", "|", result)
    result = re.sub(r"\\right\|", "|", result)
    result = re.sub(r"\\left", "", result)
    result = re.sub(r"\\right", "", result)

    # Remove standalone backslashes (used for spacing in LaTeX)
    # Match backslash followed by space or at end of word
    result = re.sub(r"\\\s+", " ", result)  # backslash followed by space(s)
    result = re.sub(r"\\(?=[^\w\\])", " ", result)  # backslash before non-word char

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

    # Match .. code-block:: python (or any language) followed by optional blank line and indented content
    html = re.sub(r"\.\. code-block::[^\n]*\n(?:\n)?((?:[ \t]+.*\n)*)", save_code_block, html)

    # Extract and convert math blocks
    math_blocks = []
    inline_math_items = []

    def save_math_block(match):
        math_content = match.group(1)
        # Clean up the math but preserve line structure for multiline equations
        math_lines = math_content.strip().split("\n")
        # Remove leading indentation consistently
        cleaned_lines = [line.strip() for line in math_lines if line.strip()]

        # Check if this is a multiline equation (contains & or \\)
        is_multiline = any("&" in line or "\\\\" in line for line in cleaned_lines)

        if is_multiline:
            # Preserve multiline structure and align on '&' (LaTeX align-style).
            aligned_rows = []
            for line in cleaned_lines:
                # Remove trailing \\
                line = re.sub(r"\\\\\s*$", "", line).strip()
                if not line:
                    continue

                if "&" in line:
                    lhs_raw, rhs_raw = line.split("&", 1)
                    lhs = latex_to_unicode(lhs_raw.strip(), display_mode=True)
                    rhs = latex_to_unicode(rhs_raw.strip(), display_mode=True)
                    aligned_rows.append(
                        "<tr>"
                        '<td style="text-align:right;padding-right:0.35em;vertical-align:middle;white-space:nowrap;">'
                        f"{lhs}"
                        "</td>"
                        '<td style="text-align:left;vertical-align:middle;white-space:nowrap;">'
                        f"{rhs}"
                        "</td>"
                        "</tr>"
                    )
                else:
                    expr = latex_to_unicode(line, display_mode=True)
                    aligned_rows.append(
                        "<tr>"
                        '<td colspan="2" style="text-align:center;vertical-align:middle;white-space:nowrap;">'
                        f"{expr}"
                        "</td>"
                        "</tr>"
                    )

            unicode_math = f'<table style="margin:0 auto;border-collapse:collapse;">{"".join(aligned_rows)}</table>'
        else:
            # Single line equation - join all lines
            latex_str = " ".join(cleaned_lines)
            unicode_math = latex_to_unicode(latex_str, display_mode=True)

        math_blocks.append(unicode_math)
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
                # Check if any line is a math BLOCK placeholder (not inline)
                # Inline math placeholders should be kept together with text
                has_math_block = any("<!--MATHBLOCK" in l for l in paragraph_lines)

                if has_math_block:
                    # Wrap each line in <p> tags separately
                    for para_line in paragraph_lines:
                        result_lines.append(f"<p>{para_line}</p>")
                else:
                    # Multi-line paragraph (keep together even if it has inline math)
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

    # Restore math blocks as display equations.
    for i, unicode_math in enumerate(math_blocks):
        if "\n" in unicode_math:
            display_math = "<br/>".join(line for line in unicode_math.split("\n") if line.strip())
            html = html.replace(
                f"<!--MATHBLOCK{i}-->",
                (
                    '<span style="display:block;text-align:center;font-size:1.18em;'
                    'line-height:1.6;margin:0.35em 0;">'
                    f"{display_math}"
                    "</span>"
                ),
            )
        elif unicode_math.lstrip().startswith("<table"):
            html = html.replace(
                f"<!--MATHBLOCK{i}-->",
                (
                    '<span style="display:block;text-align:center;font-size:1.18em;'
                    'line-height:1.6;margin:0.35em 0;">'
                    f"{unicode_math}"
                    "</span>"
                ),
            )
        else:
            html = html.replace(
                f"<!--MATHBLOCK{i}-->",
                (
                    '<span style="display:block;text-align:center;font-size:1.18em;'
                    'line-height:1.6;margin:0.35em 0;">'
                    f"{unicode_math}"
                    "</span>"
                ),
            )

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
    logger.info(model.__doc__)
