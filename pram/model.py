import logging
from pathlib import Path
from typing import Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

logger = logging.getLogger(__name__)


class ParameterisedModel(BaseModel):
    """Extended Pydantic model that store parameters used to construct fields."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    params: Optional[dict] = None

    ### Save/Load ###############################

    def save_parameters(self, loc: Path, format: Literal["json", "yaml"] = "yaml"):
        if not loc.parent.exists():
            raise FileNotFoundError(loc.parent)

        with loc.open("w", encoding="utf8") as fp:
            if format == "yaml":
                yaml.dump(self.params, fp, default_flow_style=False)
            else:
                fp.write(str(self.params))

    def save(self, loc: Path, with_parameters: bool = True):
        if not loc.parent.exists():
            raise FileNotFoundError(loc.parent)
        loc.write_text(str(self.model_dump()))

        if with_parameters:
            stem = loc.stem
            params_loc = (loc.parent / f"{stem}_params").with_suffix(".yaml")
            self.save_parameters(params_loc, format="yaml")

    ### Private #################################

    @model_validator(mode="before")
    @classmethod
    def _store_params(cls, data: dict):
        _data = data.copy()
        if "params" not in _data:
            _data["params"] = data.copy()

        return _data
