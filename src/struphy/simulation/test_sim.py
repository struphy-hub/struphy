from struphy import Simulation
from struphy.models.utils import get_models

def test_to_from_dict():
    models = get_models(model_type="Toy")
    for model_type in models:
        print(f"Testing model: {model_type}")
        sim1 = Simulation(model = model_type())
        dct = sim1.to_dict()
        sim2 = Simulation.from_dict(dct)

        print(f"sim1: {sim1}")
        print(f"sim2: {sim2}")

        assert sim1 == sim2