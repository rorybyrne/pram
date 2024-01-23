from pathlib import Path
from typing import Annotated, Optional, Tuple

import yaml
from pydantic import BeforeValidator, ConfigDict
from torch import Tensor

from pram import ParameterisedModel, ParameterisedTensor


def test_tensor_parameterisation_should_work():
    class Network(ParameterisedModel):
        weights: ParameterisedTensor

    data = {"weights": {"mean": 5.0, "std": 0.5, "size": [3, 2]}}
    net = Network.model_validate(data)

    assert net.params == data
    assert isinstance(net.weights, Tensor)


def test_complex_model_structure(tmp_path: Path):
    class NeuronParameters(ParameterisedModel):
        size: Tuple[int]
        a: ParameterisedTensor
        v_th: ParameterisedTensor

    class Neuron(ParameterisedModel):
        kind: str
        parameters: NeuronParameters

    class Synapse(ParameterisedModel):
        weights: ParameterisedTensor
        tau: int

    class Population(ParameterisedModel):
        neuron: Neuron
        syn_input: Synapse
        syn_rec: Optional[Synapse] = None
        parity_ratio: float

    class Network(ParameterisedModel):
        hidden: Population
        output: Population

    class Rig(ParameterisedModel):
        network: Network

    spec_path = Path(__file__).parent / "spec2.yaml"
    with spec_path.open("r") as fp:
        data = yaml.safe_load(fp)

    rig = Rig.from_parameters(file=spec_path)

    assert rig.params == data
    assert isinstance(rig.network.hidden.syn_input.weights, Tensor)

    d = tmp_path / "rig"
    d.mkdir()
    model_data = d / "rig.yaml"
    params_data = d / "rig_params.yaml"

    rig.save(model_data)
    with params_data.open("r") as fp:
        loaded_params = yaml.safe_load(fp)

    assert loaded_params == rig.params
