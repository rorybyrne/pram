from pathlib import Path
from typing import List, Optional, Tuple

import yaml
from torch import Tensor

from pram import ParameterisedModel, ParameterisedTensor


def test_tensor_parameterisation_should_work():
    class Network(ParameterisedModel):
        weights: ParameterisedTensor
        size: Tuple[int, int]

    data = {"size": (3, 2), "weights": {"mean": 5.0, "std": 0.5}}
    net = Network.model_validate(data)

    assert net.params == data
    assert isinstance(net.weights, Tensor)


def test_complex_model_structure(tmp_path: Path):
    class Layer(ParameterisedModel):
        w_in: ParameterisedTensor
        w_rec: Optional[ParameterisedTensor] = None

    class Network(ParameterisedModel):
        layers: List[Layer]

    class Rig(ParameterisedModel):
        network: Network

    spec_path = Path(__file__).parent / "spec.yaml"
    with spec_path.open("r") as fp:
        data = yaml.safe_load(fp)

    rig = Rig.model_validate(data)

    assert rig.params == data
    assert isinstance(rig.network.layers[0].w_in, Tensor)

    d = tmp_path / "rig"
    d.mkdir()
    model_data = d / "rig.yaml"
    params_data = d / "rig_params.yaml"

    rig.save(model_data)
    with params_data.open("r") as fp:
        loaded_params = yaml.safe_load(fp)

    assert loaded_params == rig.params
