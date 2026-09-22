# Segmented electrode channel

This example visualizes the new single-patch `SegmentedElectrodeChannel` mapping:
three longitudinal electrodes on each shaped channel wall, separated by finite gaps.
The central aperture is a slit with 2.25 mm half-height, narrower than the 5 mm
domain half-height (10 mm full channel height).

Create the 2D mapped-grid figure:

```sh
MPLBACKEND=Agg .venv/bin/python examples/IonOpticsElectrostatic/segmented_electrode_channel/visualize_domain.py
```

Open the interactive extruded-domain view (requires a working PyVista display):

```sh
.venv/bin/python examples/IonOpticsElectrostatic/segmented_electrode_channel/visualize_domain.py --show-3d
```

The segment voltages are in normalized potential units. In an ion-optics simulation,
convert SI volts with `IonOpticsUnits.potential()` before passing them to the segments.

Run the zero-current 5 keV proton-beam example:

```sh
MPLBACKEND=Agg .venv/bin/python examples/IonOpticsElectrostatic/segmented_electrode_channel/vacuum_aperture_beam.py
```

It solves the vacuum potential from the electrode segments, traces 31 proton rays,
and writes a potential/trajectory figure. It does not include space charge.

Run the two-stage accelerator with two apertures (0 → -5 → -10 kV):

```sh
MPLBACKEND=Agg .venv/bin/python examples/IonOpticsElectrostatic/segmented_electrode_channel/double_aperture_accelerator.py
```

It launches 2 keV protons and plots their nominal 10 keV electrostatic energy gain,
plus a separate mapped-grid figure used by the field solve.

`segmented_verification.py` additionally compares rays with an independent
high-order SciPy integration of the finite-element field. The accompanying
pytest regression checks the mapped aperture dimensions, every electrode
voltage, boundary losses, energy gain, and mapped-coordinate particle push.
