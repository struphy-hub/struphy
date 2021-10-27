import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Allow `json` to serialize `numpy` types.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        #     return float(obj)
        # elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
        #     return int(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)

        return json.JSONEncoder.default(self, obj)