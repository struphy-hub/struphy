"""Fill the box array from the box column of the markers, see :meth:`~struphy.pic.base.Particles.put_particles_in_boxes`."""

# module import (not names): the module may still be an older compiled version until `struphy compile` runs
import struphy.pic.sorting_kernels as sorting_kernels


def fill_boxes(
    markers: "float[:,:]",
    holes: "bool[:]",
    boxes: "int[:,:]",
    next_index: "int[:]",
    box_index: "int" = -2,
) -> "int":
    """Count the particles per box (box number in column ``box_index`` of the markers) into ``next_index`` and, if
    every box fits into a row of ``boxes`` (the last column stays -1 as terminator), write the particle rows into
    ``boxes``.

    Returns the largest number of particles in a box; if it is ``>= boxes.shape[1]``, ``boxes`` is left untouched and
    the caller must raise.
    """
    max_in_box = sorting_kernels.count_particles_in_boxes(markers, holes, next_index, box_index)
    if max_in_box < boxes.shape[1]:
        sorting_kernels.assign_particles_to_boxes(markers, holes, boxes, next_index, box_index)
    return max_in_box
