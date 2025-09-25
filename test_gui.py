import ast
import base64
import inspect
import io
import tempfile
from typing import get_type_hints
# from docutils.core import publish_parts
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from nicegui import ui
import re
import struphy.models.toy as toymodels

from struphy.geometry import domains
from struphy.geometry.domains import Domain
from struphy.models.base import StruphyModel

CARD_SETUP = "p-4 border-2 border-gray-400 rounded-lg shadow-md"

# Collect available domains
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
model_name = "Vlasov"
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
        
        ui.add_css('''
                .nicegui-markdown a {
                    color: orange;
                    text-decoration: none;
                }
                .nicegui-markdown a:hover {
                    color: red;
                    text-decoration: underline;
                }
            ''')
        
        doc = cls.__doc__

        # Replace refs and classes
        doc = re.sub(r':ref:`([^`]+)`', r'**\1**', doc)
        doc = re.sub(r':class:`([^`]+)`', r'`\1`', doc)

        # Replace math directive with LaTeX blocks
        doc = re.sub(r'\.\. math::\n\n(.*?)\n\n', r'$\1$\n\n', doc, flags=re.S)

        with ui.card().classes('p-6'):
            ui.markdown(doc, extras=["latex"])
        
        # Print plain docstring
        # ui.markdown(doc, extras=["latex"])
        
        # html_parts = publish_parts(cls.__doc__, writer_name='html')
        # html = html_parts['body']
        # with ui.card().classes('p-6'):
        #     ui.html(html)


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

        with ui.row():#.classes("justify-center"):
            ui.button("Show domain", on_click=run_simulation)

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
