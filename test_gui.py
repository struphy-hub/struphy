import ast
import base64
import inspect
import io
import os
import re
import tempfile
from typing import get_type_hints

# from docutils.core import publish_parts
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from nicegui import ui
from sympy import plot

import struphy.models as models
from struphy.geometry import domains
from struphy.geometry.domains import Domain
from struphy.models.base import StruphyModel

CARD_SETUP = "p-4 border-2 border-gray-400 rounded-lg shadow-md"
CARD_SETUP_NO_PAD = "border-2 border-gray-400 rounded-lg shadow-md"

# Collect available domains
domain_dict = {}
for name, cls in domains.__dict__.items():
    if isinstance(cls, type) and issubclass(cls, Domain) and cls != Domain:
        domain_dict[name] = cls
domain_names = list(domain_dict.keys())

model_dict = {}
for name, cls in models.__dict__.items():
    if isinstance(cls, type) and issubclass(cls, StruphyModel) and cls != StruphyModel:
        model_dict[name] = cls
model_names = list(model_dict.keys())

# Globals
model_name = "Maxwell"
domain_name = "Cuboid"
matplotlib_ui = None
plotly_ui = None
param_inputs = {}  # store input fields for parameters


def show_domain_interactive():
    global matplotlib_ui
    global plotly_ui

    # Collect typed params
    params = {}
    for pname, (input_field, annotation) in param_inputs.items():
        value = input_field.value
        # print(f"{pname}: {value} ({annotation})")
        try:
            if annotation is bool:
                params[pname] = bool(value)
            elif annotation is int:
                params[pname] = int(value)
            elif annotation is float:
                params[pname] = float(value)
            elif annotation is tuple:
                params[pname] = t = ast.literal_eval(value)
            else:
                params[pname] = value  # fallback to string
        except Exception:
            params[pname] = value  # fallback if conversion fails
        if params[pname] == "None":
            params[pname] = None
    # print(f"Running simulation with {params}")
    # Create domain instance
    domain: Domain = domain_dict[domain_name](**params)

    # fig_plotly = domain.show3D_interactive()
    # fig_sidetopview = domain.show_plotly()
    fig_domain = domain.show_combined_plotly()
    # plotly_ui = None

    if plotly_ui is None:
        with (
            ui.card()
            .classes("p-0 border-2 border-gray-400 rounded-lg shadow-md")
            .style("width: 90vw; max-width: 100vw; margin: 0;")
        ):
            # Plotly figure fills the card completely
            plotly_ui = ui.plotly(fig_domain).classes("w-full h-[300px]; margin: 10")
    else:
        plotly_ui.figure = fig_domain
        plotly_ui.update()


def show_domain():
    global matplotlib_ui
    global plotly_ui

    # Collect typed params
    params = {}
    for pname, (input_field, annotation) in param_inputs.items():
        value = input_field.value
        # print(f"{pname}: {value} ({annotation})")
        try:
            if annotation is bool:
                params[pname] = bool(value)
            elif annotation is int:
                params[pname] = int(value)
            elif annotation is float:
                params[pname] = float(value)
            elif annotation is tuple:
                params[pname] = t = ast.literal_eval(value)
            else:
                params[pname] = value  # fallback to string
        except Exception:
            params[pname] = value  # fallback if conversion fails
        if params[pname] == "None":
            params[pname] = None
    # print(f"Running simulation with {params}")
    # Create domain instance
    domain: Domain = domain_dict[domain_name](**params)

    fig_matplotlib, ax = domain.show()

    if matplotlib_ui is None:
        with ui.card().classes(CARD_SETUP + " w-full h-full"):
            ui.label("Simulation domain")
            matplotlib_ui = ui.matplotlib(figsize=(12, 6))

    # Always redraw the figure
    fig_matplotlib = matplotlib_ui.figure
    fig_matplotlib.clear()
    domain.show(fig=fig_matplotlib)
    matplotlib_ui.update()


def update_domain(value):
    """Update selected domain and refresh parameter UI"""
    global domain_name, param_inputs
    domain_name = value

    # Clear old parameter inputs
    param_container.clear()
    param_inputs = {}

    # Introspect constructor parameters + type hints
    cls = domain_dict[domain_name]
    sig = inspect.signature(cls.__init__)
    hints = get_type_hints(cls.__init__)

    for pname, param in sig.parameters.items():
        if pname == "self":
            continue

        # Determine default value
        default = "" if param.default is inspect.Parameter.empty else param.default

        # Get type hint (if any)
        annotation = hints.get(pname, str)  # fallback to str

        # Choose input widget depending on type
        with param_container:
            if annotation is bool:
                inp = ui.checkbox(pname, value=bool(default))
            elif annotation in (int, float):
                inp = ui.number(label=pname, value=default if default != "" else 0)
            else:
                inp = ui.input(label=pname, value=str(default))

            # store field and annotation
            param_inputs[pname] = (inp, annotation)


def update_model(value):
    """Update selected domain and refresh parameter UI"""
    global model_name
    model_name = value

    # Clear old parameter inputs
    model_container.clear()
    param_inputs = {}

    # Introspect constructor parameters + type hints
    cls = model_dict[model_name]
    sig = inspect.signature(cls.__init__)
    hints = get_type_hints(cls.__init__)

    with model_container:
        ui.add_css("""
                .nicegui-markdown a {
                    color: orange;
                    text-decoration: none;
                }
                .nicegui-markdown a:hover {
                    color: red;
                    text-decoration: underline;
                }
            """)

        doc = cls.__doc__
        # print(repr(doc))
        doc = doc.replace(":ref:`normalization`:", "Normalization:")
        doc = doc.replace(":ref:`Equations <gempic>`:", "Equations:")
        doc = doc.replace(":ref:`propagators` (called in sequence):", "Propagators:")

        # save docstring to replace lines
        tmp_path = os.path.join(os.getcwd(), "tmp.txt")
        try:
            file = open(tmp_path, "x")
        except FileExistsError:
            file = open(tmp_path, "w")
        file.write(doc)
        file.close()

        with open(tmp_path, "r") as file:
            doc = r""
            for line in file:
                if ":class:`~" in line:
                    # print(repr(line))
                    s1 = line.split("~")[1]
                    # print(repr(s1))
                    s2 = s1.split("`")[0]
                    # print(repr(s2))
                    s3 = s2.removeprefix("struphy.propagators.")
                    # print(repr(s3))
                    if "_fields" in s2:
                        doc += f"""`{s3} <https://struphy.pages.mpcdf.de/struphy/sections/subsections/propagators_fields.html#{s2}>`_
                        """
                    elif "_markers" in s2:
                        doc += f"""`{s3} <https://struphy.pages.mpcdf.de/struphy/sections/subsections/propagators_markers.html#{s2}>`_
                        """
                    elif "_coupling" in s2:
                        doc += f"""`{s3} <https://struphy.pages.mpcdf.de/struphy/sections/subsections/propagators_coupling.html#{s2}>`
                        """
                else:
                    doc += line

        with ui.card().classes("p-6"):
            ui.restructured_text(doc)

        # # Replace refs and classes
        # doc = re.sub(r':ref:`([^`]+)`', r'**\1**', doc)
        # doc = re.sub(r':class:`([^`]+)`', r'`\1`', doc)

        # # Replace math directive with LaTeX blocks
        # doc = re.sub(r'\.\. math::\n\n(.*?)\n\n', r'$\1$\n\n', doc, flags=re.S)

        # # Print plain docstring
        # ui.markdown(doc, extras=["latex"])

        # html_parts = publish_parts(doc, writer_name='html')
        # html = html_parts['body']


with ui.tabs().classes("w-full") as tabs:
    domain_tab = ui.tab("Domain")
    model_tab = ui.tab("Model")

with ui.tab_panels(tabs, value=domain_tab).classes("w-full"):
    with ui.tab_panel(domain_tab):
        # UI layout
        with ui.card().classes(CARD_SETUP):
            with ui.row():
                # ui.label("Select a domain:")
                ui.select(
                    domain_names,
                    value=domain_name,
                    on_change=lambda e: update_domain(e.value),
                )

        with ui.card().classes(CARD_SETUP):
            param_container = ui.row()  # container for parameter fields

        # Initialize with default domain
        update_domain(domain_name)

        with ui.row():  # .classes("justify-center"):
            ui.button("Show domain", on_click=show_domain)
            ui.button("Show domain (interactive)", on_click=show_domain_interactive)

    with ui.tab_panel(model_tab):
        # ui.label(f'Model tab')
        with ui.card().classes(CARD_SETUP):
            with ui.row():
                # ui.label("Select a domain:")
                ui.select(
                    model_names,
                    value=model_name,
                    on_change=lambda e: update_model(e.value),
                )

        with ui.card().classes(CARD_SETUP):
            model_container = ui.row()  # container for parameter fields

        # Initialize with default domain
        update_model(model_name)

ui.run()
