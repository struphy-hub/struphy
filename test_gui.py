import ast
import base64
import inspect
import io
import tempfile
from typing import get_type_hints

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from nicegui import ui

from struphy.geometry import domains
from struphy.geometry.domains import Domain

# Collect available domains
domain_dict = {}
for name, cls in domains.__dict__.items():
    if isinstance(cls, type) and issubclass(cls, Domain) and cls != Domain:
        domain_dict[name] = cls
domain_names = list(domain_dict.keys())

# Globals
domain_name = "Tokamak"
fig_image = None
param_inputs = {}  # store input fields for parameters


def run_simulation():
    global fig_image

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
    print(f"Running simulation with {params}")
    # Create domain instance
    domain = domain_dict[domain_name](**params)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        fig, ax = domain.show(save_dir=tmp.name)

        buf = io.BytesIO()
        FigureCanvas(fig).print_png(buf)
        buf.seek(0)

    img_base64 = base64.b64encode(buf.getvalue()).decode("ascii")
    img_src = f"data:image/png;base64,{img_base64}"

    if fig_image is None:
        fig_image = ui.image(img_src)
    else:
        fig_image.source = img_src


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


# UI layout
with ui.row():
    ui.label("Select a domain:")
    ui.select(
        domain_names, value=domain_name, on_change=lambda e: update_domain(e.value)
    )

param_container = ui.row()  # container for parameter fields

# Initialize with default domain
update_domain(domain_name)

ui.button("Show domain", on_click=run_simulation)

ui.run()
