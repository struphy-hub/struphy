from psydac.ddm.mpi import MockComm
from psydac.ddm.mpi import mpi as MPI

from struphy.utils.arrays import xp as np


class CloneConfig:
    """
    Manages the configuration for clone-based parallel processing using MPI.

    This class organizes MPI processes into “clones” by splitting the global communicator
    into sub_comm (for intra-clone communication) and inter_comm (for
    cross-clone communication).

    It also provides a method for getting the Np for the current clone, this is needed
    for injecting the correct number of particles in each clone.
    """

    def __init__(
        self,
        comm: MPI.Intracomm,
        params: None,
        num_clones=1,
    ):
        """
        Initialize a CloneConfig instance.

        Parameters:
            comm : (MPI.Intracomm)
                The MPI communicator covering all processes.
            params : StruphyParameters
                Struphy simulation parameters.
            num_clones : int, optional
                The number of clones to create. The total number of MPI ranks must be divisible by this number.
        """

        self._params = params
        self._num_clones = num_clones

        self._sub_comm = None
        self._inter_comm = None

        self._species_list = None

        self._clone_rank = 0
        self._clone_id = 0

        if comm is not None:
            assert isinstance(comm, MPI.Intracomm)
            rank = comm.Get_rank()
            size = comm.Get_size()

            # Ensure the total number of ranks is divisible by the number of clones
            if size % num_clones != 0:
                if rank == 0:
                    print(
                        f"Total number of ranks ({size}) is not divisible by the number of clones ({num_clones}).",
                    )
                MPI.COMM_WORLD.Abort()  # MPI abort instead of sys.exit(1)

            # Determine the color and rank within each clone
            ranks_per_clone = size // num_clones
            clone_color = rank // ranks_per_clone

            # Create a sub-communicator for each clone
            self._sub_comm = comm.Split(clone_color, rank)
            self._clone_rank = self.sub_comm.Get_rank()

            # Create an inter-clone communicator for cross-clone communication
            self._inter_comm = comm.Split(self.clone_rank, rank)
            self._clone_id = self.inter_comm.Get_rank()

    def get_Np_clone(self, Np, clone_id=None):
        """
        Distribute the total number of particles among clones.

        Given the total particle count (Np), this method calculates how many particles
        should be allocated to a specific clone. The distribution is even, with any remainder
        distributed to the first few clones.

        Parameters:
            Np : int
                Total number of particles to be distributed.
            clone_id : int, optional
                The identifier of the clone. If None, the current clone's ID is used.

        Returns:
            int: The number of particles to inject for the specified clone.
        """

        if clone_id is None:
            clone_id = self.clone_id

        # Calculate the base value and remainder
        base_value = Np // self.num_clones
        remainder = Np % self.num_clones

        Np_clone = base_value

        if clone_id < remainder:
            Np_clone += 1

        return Np_clone

    def get_Np_global(self, species_name):
        """
        Return the total particle count (Np).

        If Np is not explicitly set in the params,
        then Np is calculated based on ppc or ppb.

        Parameters:
            species_name: str
                Name of the particle species

        Returns:
            int: The number of particles.
        """
        species = self.params["kinetic"][species_name]
        markers = species["markers"]

        if "Np" in markers:
            return markers["Np"]
        elif "ppc" in markers:
            n_cells = np.prod(self.params["grid"]["Nel"], dtype=int)
            return int(markers["ppc"] * n_cells)
        elif "ppb" in markers:
            n_boxes = np.prod(species["boxes_per_dim"], dtype=int) * self.num_clones
            return int(markers["ppb"] * n_boxes)

    def print_clone_config(self):
        """Print a table summarizing the clone configuration."""
        if isinstance(MPI.COMM_WORLD, MockComm):
            comm_world = None
            rank = 0
            size = 1
        else:
            comm_world = MPI.COMM_WORLD
            rank = comm_world.Get_rank()
            size = comm_world.Get_size()

        ranks_per_clone = size // self.num_clones
        clone_color = rank // ranks_per_clone

        # Gather information from all ranks to the rank 0 process
        if comm_world is None:
            clone_info = [(rank, clone_color, self.clone_rank, self.clone_id)]
        else:
            clone_info = comm_world.gather(
                (rank, clone_color, self.clone_rank, self.clone_id),
                root=0,
            )

        if rank == 0:
            print(f"\nNumber of clones: {self.num_clones}")
            # Generate an ASCII table for each clone
            message = ""
            for clone in range(self.num_clones):
                message += f"Clone {clone}:\n"
                message += "comm.Get_rank() | sub_comm.Get_rank() | inter_comm.Get_rank()\n"
                message += "-" * 66 + "\n"
                for entry in clone_info:
                    if entry[1] == clone:
                        message += f"{entry[0]:15} | {entry[2]:19} | {entry[3]:21}\n"
            print(message)

    def print_particle_config(self):
        """Print the particle configuration for each clone."""
        if self.params is None:
            if MPI.COMM_WORLD.Get_rank() == 0:
                print("No params in clone_config")
            return
        else:
            _skip = False
            if "kinetic" not in self.params:
                _skip = True
            else:
                for name, species in self.params["kinetic"].items():
                    if ("Np" not in species["markers"]) and ("ppc" not in species["markers"]):
                        _skip = True

            if _skip:
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print("No kinetic parameters")
                    return

            assert "grid" in self.params

            marker_keys = ["Np", "ppc"]

            species_list = list([sp for sp in self.params["kinetic"].keys()])
            column_sums = {species_name: {marker_key: 0 for marker_key in marker_keys} for species_name in species_list}

            # Prepare breakline
            breakline = "-" * (6 + 30 * len(species_list) * len(marker_keys)) + "\n"

            # Prepare the header
            header = "Particle injection by clone:\n"
            header += "Clone  "
            for species_name in species_list:
                for marker_key in marker_keys:
                    column_name = f"{marker_key} ({species_name})"
                    header += f"| {column_name:30} "
            header += "\n"

            # Prepare the data rows
            rows = ""
            for species_name in species_list:
                for i_clone in range(self.num_clones):
                    row = f"{i_clone:6} "
                    # Np = self.params["kinetic"][species_name]["markers"]["Np"]
                    Np = self.get_Np_global(species_name)
                    n_cells_clone = np.prod(self.params["grid"]["Nel"])

                    Np_clone = self.get_Np_clone(Np, clone_id=i_clone)
                    ppc_clone = Np_clone / n_cells_clone

                    row += f"| {str(Np_clone):30} "
                    row += f"| {str(ppc_clone):30} "

                    column_sums[species_name]["Np"] += Np_clone
                    column_sums[species_name]["ppc"] += ppc_clone

                    rows += row + "\n"

            # Prepare the sum row
            sum_row = "Sum    "
            for species_name in species_list:
                for marker_key in marker_keys:
                    sum_value = column_sums[species_name][marker_key]
                    if marker_key in self.params["kinetic"][species_name]["markers"].keys():
                        params_value = self.params["kinetic"][species_name]["markers"][marker_key]
                        if params_value is not None:
                            assert sum_value == params_value, f"{sum_value = } and {params_value = }"
                    sum_row += f"| {str(sum_value):30} "

            # Print the final message
            message = header + breakline + rows + breakline + sum_row
            if MPI.COMM_WORLD.Get_rank() == 0:
                print(message)

    def free(self):
        """Free the MPI communicators associated with this clone configuration."""
        self.sub_comm.Free()
        self.inter_comm.Free()

    @property
    def params(self):
        """Get the simulation parameters."""
        return self._params

    @property
    def num_clones(self):
        """Get the number of domain clones."""
        return self._num_clones

    @property
    def sub_comm(self):
        """Get the sub-communicator (for communication within the current clone)."""
        return self._sub_comm

    @property
    def inter_comm(self):
        """Get the inter-communicator for cross-clone communication."""
        return self._inter_comm

    @property
    def clone_rank(self):
        """Get the rank of the process within its clone's sub_comm."""
        return self._clone_rank

    @property
    def clone_id(self):
        """Get the clone identifier."""
        return self._clone_id
