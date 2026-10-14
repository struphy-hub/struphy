"""Sort the boxed particles and rebuild the box array, see :meth:`~struphy.pic.base.Particles.do_sort`."""

# module import (not names): the module may still be an older compiled version until `struphy compile` runs
import struphy.pic.sorting_kernels as sorting_kernels


def sort_and_fill_boxes(
    markers: "float[:,:]",
    holes: "bool[:]",
    boxes: "int[:,:]",
    next_index: "int[:]",
    cumul_next_index: "int[:]",
    swap_line_1: "float[:]",
    swap_line_2: "float[:]",
    box_index: "int" = -2,
):
    """Reorder the marker rows by box, then update ``holes`` and rebuild ``boxes`` for the new rows, so that the box
    array stays valid after the sort.

    ``next_index`` must hold the number of particles per box, as left by ``box_particles`` or ``fill_boxes``. The
    swap lines are scratch buffers.
    """
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
