# Dictionary containing font styling for plots
font_dict = dict(family="Arial, sans-serif", size=18, color="black")


def format_size(fig, width=1000, height=600):
    """
    Adjust the size of a figure.

    Parameters:
    fig (plotly.graph_objects.Figure): The figure to be formatted.
    width (int, optional): The width of the figure. Default is 800.
    height (int, optional): The height of the figure. Default is 500.
    """
    fig.update_layout(
        width=width,  # figure width
        height=height,  # figure height
    )


def format_font(fig):
    """
    Apply general font formatting to a figure.

    Parameters:
    fig (plotly.graph_objects.Figure): The figure to be formatted.
    """
    fig.update_layout(
        font=font_dict,  # font formatting
        plot_bgcolor="white",  # background color
    )


def format_axes(fig):
    """
    Apply formatting to the x and y axes of a figure.

    Parameters:
    fig (plotly.graph_objects.Figure): The figure to be formatted.
    """
    # y-axis formatting
    fig.update_yaxes(
        showline=True,  # add line at x=0
        linecolor="black",  # line color
        linewidth=2.4,  # line size
        ticks="inside",  # ticks inside axis
        tickfont=font_dict,  # tick label font
        mirror="allticks",  # add ticks to top/right axes
        tickwidth=2.4,  # tick width
        tickcolor="black",  # tick color
        tickformat="none",  # No scientific notation for y-axis
        exponentformat="none",  # No 'e' notation for exponents
    )
    # x-axis formatting
    fig.update_xaxes(
        showline=True,  # add line at y=0
        showticklabels=True,  # show tick labels
        linecolor="black",  # line color
        linewidth=2.4,  # line size
        ticks="inside",  # ticks inside axis
        tickfont=font_dict,  # tick label font
        mirror="allticks",  # add ticks to top/right axes
        tickwidth=2.4,  # tick width
        tickcolor="black",  # tick color
        tickformat="none",  # No scientific notation for x-axis
        exponentformat="none",  # No 'e' notation for exponents
    )


def to_superscript(num):
    """
    Convert a number to its Unicode superscript representation.

    Parameters:
    num (int or str): The number to be converted.

    Returns:
    str: The Unicode superscript representation of the number.
    """
    superscript_map = {
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
        "+": "",
        "-": "",
    }
    return "".join(superscript_map[digit] for digit in str(num))


def format_grid(fig):
    """
    Apply grid formatting to a figure.

    Parameters:
    fig (plotly.graph_objects.Figure): The figure to be formatted.
    """
    fig.update_layout(
        xaxis_showgrid=True,  # Show grid lines on x-axis
        xaxis_gridcolor="black",  # Grid color for x-axis
        xaxis_griddash="dot",  # Grid dash style for x-axis
        yaxis_showgrid=True,  # Show grid lines on y-axis
        yaxis_gridcolor="black",  # Grid color for y-axis
        yaxis_griddash="dot",  # Grid dash style for y-axis
    )
