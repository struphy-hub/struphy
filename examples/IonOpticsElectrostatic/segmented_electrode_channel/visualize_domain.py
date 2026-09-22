"""Visualize the single-patch, multi-electrode channel mapping.

Run from the repository root:

    MPLBACKEND=Agg .venv/bin/python examples/IonOpticsElectrostatic/segmented_electrode_channel/visualize_domain.py

For the interactive PyVista view, add ``--show-3d``.
"""

import argparse
from pathlib import Path

from struphy.geometry.domains import ElectrodeSegment, SegmentedElectrodeChannel


def build_domain():
    """Return a shaped slit channel with a central, half-radius aperture."""
    length = 80e-3
    domain_radius = 5e-3
    aperture_radius = 2.25e-3
    return SegmentedElectrodeChannel(
        length=length,
        width=2e-3,
        # The centre aperture is 2 * aperture_radius = 4.5 mm wide, narrower
        # than the 10 mm full channel height. The short transition sections
        # make the boundary a smooth, meshable single-patch constriction.
        lower_profile=(
            (0.0, 28e-3, 32e-3, 48e-3, 52e-3, length),
            (-domain_radius, -domain_radius, -aperture_radius, -aperture_radius, -domain_radius, -domain_radius),
        ),
        upper_profile=(
            (0.0, 28e-3, 32e-3, 48e-3, 52e-3, length),
            (domain_radius, domain_radius, aperture_radius, aperture_radius, domain_radius, domain_radius),
        ),
        segments=(
            ElectrodeSegment("lower", 0.0, 28e-3, 0.0, "entrance"),
            ElectrodeSegment("lower", 31e-3, 49e-3, -5.0, "focus"),
            ElectrodeSegment("lower", 52e-3, length, 0.0, "exit"),
            ElectrodeSegment("upper", 0.0, 28e-3, 0.0, "entrance"),
            ElectrodeSegment("upper", 31e-3, 49e-3, -5.0, "focus"),
            ElectrodeSegment("upper", 52e-3, length, 0.0, "exit"),
        ),
        num_elements=(32, 12),
        degree=(3, 3),
    )


def main(show_3d=False):
    domain = build_domain()
    output = Path(__file__).with_name("segmented_electrode_channel_mapping.png")

    # 2D mapped-grid view, saved so the default command works headlessly.
    domain.show(show_control_pts=True, save_dir=output)
    print(f"Wrote {output}")

    if show_3d:
        # Interactive PyVista view of the extruded vacuum domain.
        domain.show_3d(nx=80, ny=32, nz=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-3d", action="store_true", help="Open the interactive PyVista view.")
    args = parser.parse_args()
    main(show_3d=args.show_3d)
