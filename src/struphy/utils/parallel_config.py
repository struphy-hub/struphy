from mpi4py import MPI

class ParallelConfig:
    """Class for managing the MPI communicators"""

    def __init__(self, params=None, comm=None, num_clones=1):
        self._params = params
        self._comm = comm
        self._num_clones = num_clones
        
        self._sub_comm = None
        self._inter_comm = None

        if comm is not None:
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

            self._clone_num_particles = []
            # Process kinetic parameters if present
            if "kinetic" in params and "grid" in params:
                for i_clone in range(self.num_clones):
                    data = {'clone':{}, 'global':{}}
                    for species_name, species_data in params["kinetic"].items():
                        markers = species_data.get("markers")

                        # Calculate the base value and remainder
                        base_value = markers["Np"] // num_clones
                        remainder = markers["Np"] % num_clones

                        # Distribute the values
                        new_Np = [base_value] * num_clones
                        for i in range(remainder):
                            new_Np[i] += 1

                        # Calculate the values to the current clone
                        clone_Np = new_Np[self._inter_comm.Get_rank()]
                        clone_ppc = clone_Np / np.prod(params["grid"]["Nel"])

                        data['clone'][species_name] = {"Np": clone_Np, "ppc": clone_ppc}
                        data['global'][species_name] = {"Np": markers["Np"], "ppc": markers["ppc"]}
                    self._clone_num_particles.append(data)
    
    def get_clone_Np(self, species):
        return self.clone_num_particles[self.inter_comm.Get_rank()]['clone'][species]['Np']
    def get_clone_ppc(self, species):
        return self.clone_num_particles[self.inter_comm.Get_rank()]['clone'][species]['ppc']
    
    def get_global_Np(self, species):
        return self.clone_num_particles[self.inter_comm.Get_rank()]['global'][species]['Np']

    def get_global_ppc(self, species):
        return self.clone_num_particles[self.inter_comm.Get_rank()]['global'][species]['ppc']

    def print_clone_config(self):
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()
        ranks_per_clone = size // self.num_clones
        clone_color = rank // ranks_per_clone

        # Gather information from all ranks to the rank 0 process
        clone_info = self.comm.gather(
            (rank, clone_color, self.sub_comm.Get_rank(), self.inter_comm.Get_rank()),
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
        rank = self.comm.Get_rank()
        # If the current process is the root, compile and print the message
        if rank == 0 and self.num_clones > 1:
            
            
            marker_keys = ["Np", "ppc"]
            species_list = list(self.clone_num_particles[0]['clone'].keys())
            column_sums = {species_name: {marker_key: 0 for marker_key in marker_keys} for species_name in species_list}

            # Prepare breakline
            breakline = "-" * (6 + 30 * len(species_list) * len(marker_keys)) + "\n"

            # Prepare the header
            header = "Particle counting:\n"
            header += "Clone  "
            for species_name in species_list:
                for marker_key in marker_keys:
                    column_name = f"{marker_key} ({species_name})"
                    header += f"| {column_name:30} "
            header += "\n"

            # Prepare the data rows
            rows = ""
            for i_clone, clone_data in enumerate(self.clone_num_particles):
                print(i_clone, clone_data)
                row = f"{i_clone:6} "
                for species_name in species_list:
                    for marker_key in marker_keys:
                        value = clone_data['clone'][species_name][marker_key]
                        row += f"| {str(value):30} "
                        if value is not None:
                            column_sums[species_name][marker_key] += value
                        else:
                            column_sums[species_name][marker_key] = None
                rows += row + "\n"
            
            # Prepare the sum row
            sum_row = "Sum    "
            for species_name in species_list:
                for marker_key in marker_keys:
                    sum_value = column_sums[species_name][marker_key]
                    params_value = self.params["kinetic"][species_name]["markers"][marker_key]
                    assert sum_value == params_value, f"{sum_value = } and {params_value = }"
                    sum_row += f"| {str(sum_value):30} "

            # Print the final message
            message = header + breakline + rows + breakline + sum_row
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
    def comm(self):
        return self._comm

    @property
    def sub_comm(self):
        return self._sub_comm

    @property
    def inter_comm(self):
        return self._inter_comm

    # @property
    # def clone_num_particles(self):
    #     return self._clone_num_particles