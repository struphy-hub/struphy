import logging
import os

import h5py
import numpy as np

logger = logging.getLogger("struphy")


class DataContainer:
    """
    Creates/opens a hdf5 file for data ouput (each process locally).

    Parameters
    ----------
    path_out : str
        Path to hdf5 data files.

    file_name : str, optional
        Name of hdf5 file.

    comm : MPI communicator
    """

    def __init__(self, path_out, file_name=None, comm=None):
        # set name of hdf5 file
        if comm is None:
            self._rank = None
            _affix = "_proc0"
        else:
            self._rank = comm.Get_rank()
            _affix = "_proc" + str(self._rank)

        if file_name is None:
            self._file_name = "data" + _affix + ".hdf5"
        else:
            if file_name.find(".hdf5") == -1:
                self._file_name = file_name + ".hdf5"
            else:
                self._file_name = file_name

        # file path
        self._file_path = os.path.join(path_out, "data/", self._file_name)

        # check if file already exists
        file_exists = os.path.exists(self.file_path)

        # dictionary with pairs (dataset key : object to save)
        self._dset_dict = {}

        # get dataset keys if file already exists and set None objects; time series are
        # chunked (see add_data), static datasets such as kinetic backgrounds are not
        if file_exists:
            dataset_keys = []

            with h5py.File(self.file_path, "a") as file:
                file.visit(
                    lambda key: (
                        dataset_keys.append(key)
                        if isinstance(file[key], h5py.Dataset) and file[key].chunks is not None
                        else None
                    ),
                )

            for key in dataset_keys:
                self._dset_dict[key] = None

    @property
    def file_name(self):
        """The hdf5 file name."""
        return self._file_name

    @property
    def file_path(self):
        """The absolute path to the hdf5 file."""
        return self._file_path

    @property
    def dset_dict(self):
        """Dictionary with dataset keys and the objects saved to them."""
        return self._dset_dict

    @staticmethod
    def _as_numpy_array(val):
        """Return a NumPy view/copy suitable for h5py writes."""
        if isinstance(val, np.ndarray):
            return val

        get = getattr(val, "get", None)
        if callable(get) and "cupy" in val.__class__.__module__:
            return get()

        return np.asarray(val)

    def add_data(self, data_dict):
        """
        Add data object to be saved during simulation.

        Parameters
        ----------
        data_dict : dict
            Name-object pairs to save during time stepping, e.g. {key : val}. key must be a string and val must be an array of fixed shape. Scalar values (floats) must therefore be passed as 1d arrays of size 1.
        """

        for key, val in data_dict.items():
            val_np = self._as_numpy_array(val)
            assert isinstance(val_np, np.ndarray)

            # if dataset already exists, check for compatibility with given array
            if key in self._dset_dict:
                with h5py.File(self.file_path, "a") as file:
                    dataset_shape = file[key].shape

                # scalar values are saved as 1d arrays of size 1
                if len(dataset_shape) == 1:
                    assert val_np.ndim == 1, "for scalar quantities, a 1d array with a single entry must used!"
                    assert val_np.size == 1, "for scalar quantities, a 1d array with a single entry must used!"

                # other values
                else:
                    assert dataset_shape[1:] == val_np.shape

            # create new dataset otherwise and save array
            else:
                with h5py.File(self.file_path, "a") as file:
                    # scalar values are saved as 1d arrays of size 1
                    if val_np.size == 1:
                        assert val_np.ndim == 1
                        file.create_dataset(key, (1,), maxshape=(None,), dtype=val_np.dtype, chunks=True)
                        file[key][0] = val_np[0]
                    else:
                        file.create_dataset(
                            key,
                            (1,) + val_np.shape,
                            maxshape=(None,) + val_np.shape,
                            dtype=val_np.dtype,
                            chunks=True,
                        )
                        file[key][0] = val_np

            # keep a reference, so the object is alive and current when it is saved
            self._dset_dict[key] = val

    def save_data(self, keys=None):
        """
        Save data objects to hdf5 file.

        Parameters
        ----------
        keys : list
            Keys to the data objects specified when using "add_data". Default saves all specified data objects.
        """
        if keys is None:
            keys = self._dset_dict
        with h5py.File(self.file_path, "a") as file:
            for key in keys:
                val = self._dset_dict[key]
                if val is None:
                    raise KeyError(f"Dataset {key!r} exists in {self.file_path} but no data was added for it")
                file[key].resize(file[key].shape[0] + 1, axis=0)
                file[key][-1] = self._as_numpy_array(val)

    def info(self):
        """Print info of data sets to screen."""

        for key in self._dset_dict:
            with h5py.File(self.file_path, "a") as file:
                logger.info(f"\nData set name: {key}")
                logger.info(f"Shape: {file[key].shape}")
                logger.info("Attributes:")
                for attr, val in file[key].attrs.items():
                    logger.info(f"{attr} {val}")
