import gc
import os

import h5py
import numpy as np
import pytest

from struphy.io.output_handling import DataContainer


def container(tmp_path):
    os.makedirs(tmp_path / "data", exist_ok=True)
    return DataContainer(str(tmp_path))


def test_saves_the_current_values_of_the_added_objects(tmp_path):
    data = container(tmp_path)
    value = np.zeros(1)
    field = np.zeros((2, 3))
    data.add_data({"scalar/energy": value, "feec/field": field})
    for step in (1, 2):
        value[0] = step
        field[:] = step
        data.save_data()
    with h5py.File(data.file_path) as file:
        np.testing.assert_array_equal(file["scalar/energy"][:], [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(file["feec/field"][:, 0, 0], [0.0, 1.0, 2.0])


def test_an_object_added_as_a_temporary_is_kept_alive(tmp_path):
    data = container(tmp_path)
    data.add_data({"scalar/value": np.full(1, 7.0)})
    gc.collect()
    np.full(1, -1.0)  # would reuse the freed memory if the container did not hold a reference
    data.save_data()
    with h5py.File(data.file_path) as file:
        np.testing.assert_array_equal(file["scalar/value"][:], [7.0, 7.0])


def test_a_restart_names_a_dataset_that_was_not_added_again(tmp_path):
    first = container(tmp_path)
    first.add_data({"scalar/energy": np.zeros(1), "scalar/dropped": np.zeros(1)})

    restarted = DataContainer(str(tmp_path))
    restarted.add_data({"scalar/energy": np.ones(1)})
    restarted.save_data(keys=["scalar/energy"])
    with pytest.raises(KeyError, match="scalar/dropped"):
        restarted.save_data()
