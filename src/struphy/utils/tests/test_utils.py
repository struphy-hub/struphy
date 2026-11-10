import numpy as np
import yaml

from struphy.utils.utils import MyDumper


def test_yaml_dumper_represents_numpy_values():
    data = {
        "float": np.float64(0.0032310501257103936),
        "integer": np.int64(3),
        "boolean": np.bool_(True),
        "array": np.array([1.0, 2.0]),
    }

    dumped = yaml.dump(data, Dumper=MyDumper, sort_keys=False)

    assert yaml.safe_load(dumped) == {
        "float": 0.0032310501257103936,
        "integer": 3,
        "boolean": True,
        "array": [1.0, 2.0],
    }
