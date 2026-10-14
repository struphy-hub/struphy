"""Assign the box to each particle and fill the box array, see :meth:`~struphy.pic.base.Particles.put_particles_in_boxes`."""

from struphy.pic.sorting_kernels import (
    assign_box_to_each_particle,
    assign_particles_to_boxes,
    count_particles_in_boxes,
)


def box_particles(
    markers: "float[:,:]",
    holes: "bool[:]",
    nx: "int",
    ny: "int",
    nz: "int",
    domain_array: "float[:]",
    boxes: "int[:,:]",
    next_index: "int[:]",
    box_index: "int" = -2,
) -> "int":
    """Write the box of each particle into column ``box_index`` of the markers, count the particles per box into
    ``next_index`` and, if every box fits into a row of ``boxes`` (the last column stays -1 as terminator), write the
    particle rows into ``boxes``.

    Returns the largest number of particles in a box; if it is ``>= boxes.shape[1]``, ``boxes`` is left untouched and
    the caller must raise.
    """
    assign_box_to_each_particle(markers, holes, nx, ny, nz, domain_array, box_index)
    max_in_box = count_particles_in_boxes(markers, holes, next_index, box_index)
    if max_in_box < boxes.shape[1]:
        assign_particles_to_boxes(markers, holes, boxes, next_index, box_index)
    return max_in_box
