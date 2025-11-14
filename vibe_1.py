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

import struphy.models.toy as toymodels
from struphy.geometry import domains
from struphy.geometry.domains import Domain
from struphy.models.base import StruphyModel

CARD_SETUP = "p-4 border-2 border-gray-400 rounded-lg shadow-md"

# Collect available domains
code_names = ["Struphy", "DESC", "GVEC", "Tokamaker"]

domain_dict = {}
for name, cls in domains.__dict__.items():
    if isinstance(cls, type) and issubclass(cls, Domain) and cls != Domain:
        domain_dict[name] = cls
domain_names = list(domain_dict.keys())

model_dict = {}
for name, cls in toymodels.__dict__.items():
    if isinstance(cls, type) and issubclass(cls, StruphyModel) and cls != StruphyModel:
        model_dict[name] = cls
model_names = list(model_dict.keys())

# Globals
code_name = "Struphy"
model_name = "Maxwell"
domain_name = "Tokamak"
matplotlib_ui = None
param_inputs = {}  # store input fields for parameters


def run_simulation():
    global matplotlib_ui

    # Collect typed params
    params = {}
    for pname, (input_field, annotation) in param_inputs.items():
        value = input_field.value
        print(f"{pname}: {value} ({annotation})")
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

    if matplotlib_ui is None:
        # Create card + matplotlib once
        with ui.card().classes(CARD_SETUP):
            ui.label("Simulation domain")
            matplotlib_ui = ui.matplotlib(figsize=(12, 6))

    # Always redraw the figure
    fig = matplotlib_ui.figure
    fig.clear()
    domain.show(fig=fig)
    matplotlib_ui.update()


def update_code(value, code_container):
    global code_name
    code_name = value

    # Clear old parameter inputs
    code_container.clear()

    rst = r"""
Tl;dr
=====

| **Struphy provides easy access to partial differential equations (PDEs) in plasma physics.**
| The package combines *performance* (for HPC), *flexibility* (physics features) and *usability* (documentation). 

*Performance* in Struphy is achieved using three building blocks:

* `numpy <https://numpy.org/>`_ (vectorization)
* `mpi4py <https://pypi.org/project/mpi4py/>`_ (parallelization)
* `pyccel <https://github.com/pyccel/pyccel>`_ (compilation)

| Heavy computational kernels are pre-compiled using the Python accelerator `pyccel <https://github.com/pyccel/pyccel>`_,
| which on average shows `better performance <https://github.com/pyccel/pyccel-benchmarks>`_ than *Pythran* or *Numba*.


| *Flexibility* comes through the possibility of applying different *models* to a plasma physics problem.
| Each model can be run on different *geometries* and can load a variety of *MHD equilibria*.


| *Usability* is guaranteed by Struphy's intuitive Python API.
| Moreover, an extensive, maintained documentation is provided. 
| In addition, you can learn Struphy through a series of Jupyter notebook `tutorials <https://struphy.pages.mpcdf.de/struphy/sections/tutorials.html>`_. 
    """

    with code_container:
        with ui.card().classes("p-6"):
            ui.restructured_text(rst)


def update_domain(value, param_container):
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


def update_model(value, model_container):
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


# --- Page 1: Simulation Center ---


@ui.page("/")
def cockpit():
    # 1. Header: Simulation Center
    with ui.header().classes("items-center justify-between"):
        ui.label("Stellignite - Cockpit").classes("text-3xl font-bold")

    # Main content area: Tabs and Panels
    with ui.row().classes("w-full"):
        # 2. Left Column: Tabs
        with ui.column():
            # The tabs container itself
            with ui.tabs().props("vertical").classes("w-full") as tabs:
                code = ui.tab("Code")
                model = ui.tab("Model")
                geometry = ui.tab("Geometry")
                env = ui.tab("Environment")

        # 3. Right Panel: Content Display
        with ui.column():
            # The tab panels container
            with ui.tab_panels(tabs, value=code).classes("w-full h-full p-4 border rounded-lg bg-white shadow"):
                with ui.tab_panel(code):
                    # 3. Panel Content: "Hello World"
                    with ui.row():
                        with ui.column():
                            with ui.card().classes(CARD_SETUP):
                                with ui.row():
                                    # ui.label("Select a domain:")
                                    ui.select(
                                        code_names,
                                        value=code_name,
                                        on_change=lambda e: update_code(e.value, code_container),
                                    )

                        with ui.column():
                            with ui.card().classes(CARD_SETUP):
                                code_container = ui.row()  # container for parameter fields

                    # Initialize with default domain
                    update_code(code_name, code_container)
                with ui.tab_panel(model):
                    # 3. Panel Content: "Hello World"
                    with ui.row():
                        with ui.column():
                            with ui.card().classes(CARD_SETUP):
                                with ui.row():
                                    # ui.label("Select a domain:")
                                    ui.select(
                                        model_names,
                                        value=model_name,
                                        on_change=lambda e: update_model(e.value, model_container),
                                    )

                        with ui.column():
                            with ui.card().classes(CARD_SETUP):
                                model_container = ui.row()  # container for parameter fields

                    # Initialize with default domain
                    update_model(model_name, model_container)
                with ui.tab_panel(geometry):
                    with ui.row():
                        with ui.column():
                            # UI layout
                            with ui.card().classes(CARD_SETUP):
                                with ui.row():
                                    # ui.label("Select a domain:")
                                    ui.select(
                                        domain_names,
                                        value=domain_name,
                                        on_change=lambda e: update_domain(e.value),
                                    )
                        with ui.column():
                            with ui.card().classes(CARD_SETUP):
                                param_container = ui.column()  # container for parameter fields

                        # Initialize with default domain
                        update_domain(domain_name, param_container)

                        with ui.column():  # .classes("justify-center"):
                            ui.button("Show domain", on_click=run_simulation)

    # 4. Bottom Right Corner: Checkout Button
    with ui.footer(fixed=False):
        with ui.row().classes("w-full justify-end"):
            # Button directs to the checkout page
            ui.button("Checkout", on_click=lambda: ui.navigate.to("/checkout")).classes("text-lg px-8 py-2").props(
                "color=green-600"
            )


# --- Page 2: Checkout Page ---


@ui.page("/checkout")
def checkout_page():
    # Header: Checkout (same as button name)
    with ui.header().classes("items-center justify-start"):
        ui.button("Back to Cockpit", on_click=lambda: ui.navigate.to("/"))
        ui.label("Checkout").classes("text-3xl font-bold ml-4")

    # # Main content area: Text elements with tab names
    # with ui.column().classes('w-full max-w-xl mx-auto p-6'):
    #     ui.label('Contents from Simulation Center Tabs:').classes('text-2xl font-semibold mb-4')

    #     # Display text elements with the same names as the left column tabs
    #     for name in tab_names:
    #         ui.label(name).classes('text-lg font-medium p-2 border-b border-gray-300')

    #     ui.label('Review and confirm your settings.').classes('mt-6 text-gray-600 italic')


# --- Run the App ---
if __name__ in {"__main__", "__mp_main__"}:
    ui.run()
