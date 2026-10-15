"""Assign boxes, sort by box and rebuild the box array in one call, see :meth:`~struphy.pic.base.Particles.do_sort`."""

# module import (not names): the module may still be an older compiled version until `struphy compile` runs
import struphy.pic.sorting_kernels as sorting_kernels


def box_and_sort_particles(
    markers: "float[:,:]",
    holes: "bool[:]",
    nx: "int",
    ny: "int",
    nz: "int",
    domain_array: "float[:]",
    boxes: "int[:,:]",
    next_index: "int[:]",
    cumul_next_index: "int[:]",
    swap_line_1: "float[:]",
    swap_line_2: "float[:]",
    box_index: "int" = -2,
) -> "int":
    """``box_particles`` followed by ``sort_and_fill_boxes``: write the box of each particle into column ``box_index``,
    count the particles per box, reorder the marker rows by box, then update ``holes`` and fill ``boxes`` for the
    sorted rows. On the CuPy backend the markers cross between device and host once in each direction.

    Returns the largest number of particles in a box; if it is ``>= boxes.shape[1]``, nothing is sorted, ``boxes`` is
    left untouched and the caller must raise.
    """
    sorting_kernels.assign_box_to_each_particle(markers, holes, nx, ny, nz, domain_array, box_index)
    max_in_box = sorting_kernels.count_particles_in_boxes(markers, holes, next_index, box_index)
    if max_in_box < boxes.shape[1]:
        sorting_kernels.sort_boxed_particles(
            markers,
            swap_line_1,
            swap_line_2,
            next_index.shape[0],
            next_index,
            cumul_next_index,
            box_index,
        )
        sorting_kernels.find_holes(markers, holes)
        # the number of particles per box is unchanged, so they still fit into the box array
        sorting_kernels.assign_particles_to_boxes(markers, holes, boxes, next_index, box_index)
    return max_in_box
