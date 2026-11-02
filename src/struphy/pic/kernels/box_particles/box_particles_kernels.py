"""Assign the box to each particle and fill the box array, see :meth:`~struphy.pic.base.Particles.put_particles_in_boxes`."""

# module import (not names): the module may still be an older compiled version until `struphy compile` runs
import struphy.pic.sorting_kernels as sorting_kernels


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
    """Write the box of each particle into row ``box_index`` of the markers, count the particles per box into
    ``next_index`` and, if every box fits into a row of ``boxes`` (the last column stays -1 as terminator), write the
    particle indices into ``boxes``.

    Returns the largest number of particles in a box; if it is ``>= boxes.shape[1]``, ``boxes`` is left untouched and
    the caller must raise.
    """
    sorting_kernels.assign_box_to_each_particle(markers, holes, nx, ny, nz, domain_array, box_index)
    max_in_box = sorting_kernels.count_particles_in_boxes(markers, holes, next_index, box_index)
    if max_in_box < boxes.shape[1]:
        sorting_kernels.assign_particles_to_boxes(markers, holes, boxes, next_index, box_index)
    return max_in_box
