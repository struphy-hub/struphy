from mpi4py import MPI
import numpy as np

class CloneConfig:
    """Class for managing the Clone configuration."""

    def __init__(self,
                 comm: MPI.Intracomm,
                 params=None,
                 num_clones=1,
                 ):
        self._params = params
        self._num_clones = num_clones
        
        self._sub_comm = None
        self._inter_comm = None

        self._species_list = None
        
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
                MPI.COMM_WORLD.Abort()  # Proper MPI abort instead of exit()

            # Determine the color and rank within each clone
            ranks_per_clone = size // num_clones
            clone_color = rank // ranks_per_clone

            # Create a sub-communicator for each clone
            self._sub_comm = comm.Split(clone_color, rank)
            local_rank = self.sub_comm.Get_rank()

            # Create an inter-clone communicator for cross-clone communication
            self._inter_comm = comm.Split(local_rank, rank)

    def get_Np_clone(self, Np, clone_id = None):
        if clone_id is None:
            clone_id = self.clone_id

        # Calculate the base value and remainder
        base_value = Np // self.num_clones
        remainder = Np % self.num_clones
        
        Np_clone = base_value
        
        if clone_id < remainder:
            Np_clone += 1
        
        return Np_clone

    def print_clone_config(self):
        comm_world = MPI.COMM_WORLD
        rank = comm_world.Get_rank()
        size = comm_world.Get_size()
        
        ranks_per_clone = size // self.num_clones
        clone_color = rank // ranks_per_clone

        # Gather information from all ranks to the rank 0 process
        clone_info = comm_world.gather(
            (rank, clone_color, self.clone_rank, self.clone_id),
            root=0,
        )

        if comm_world.Get_rank() == 0:
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
        if self.params is None:
            if MPI.COMM_WORLD.Get_rank() == 0:
                print("No params in clone_config")
        else:

            assert "kinetic" in self.params
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
                    Np = self.params["kinetic"][species_name]["markers"]["Np"]
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
                        assert sum_value == params_value, f"{sum_value = } and {params_value = }"
                    sum_row += f"| {str(sum_value):30} "

            # Print the final message
            message = header + breakline + rows + breakline + sum_row
            if MPI.COMM_WORLD.Get_rank() == 0:
                print(message)

    def free(self):
        self.sub_comm.Free()
        self.inter_comm.Free()

    @property
    def params(self):
        return self._params

    @property
    def num_clones(self):
        return self._num_clones

    @property
    def sub_comm(self):
        return self._sub_comm

    @property
    def inter_comm(self):
        return self._inter_comm
    
    @property
    def clone_rank(self):
        return self.sub_comm.Get_rank()

    @property
    def clone_id(self):
        return self.inter_comm.Get_rank()