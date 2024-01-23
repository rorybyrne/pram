import logging
from copy import deepcopy
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

logger = logging.getLogger(__name__)


class ParameterisedModel(BaseModel):
    """Extended Pydantic model that store parameters used to construct fields."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    params: Optional[dict] = None

    ### Save/Load ###############################

    @classmethod
    def from_parameters(cls, *, file: Optional[Path] = None) -> "ParameterisedModel":
        if file is not None:
            if not file.exists():
                raise FileNotFoundError(file)
            params = yaml.safe_load(file.read_text())
        else:
            raise NotImplementedError()

        context = params
        return cls.model_validate(params, context=context)

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

    # @field_validator("*", mode="before")
    # @classmethod
    # def inject_params(cls, data: Any, info: ValidationInfo):
    #     if info.field_name == "params":
    #         return data
    #
    #     assert info.field_name is not None, "TODO: msg"
    #     field = cls.model_fields.get(info.field_name, None)
    #     if field is None:
    #         raise RuntimeError("Field is None")
    #     if field.annotation is None:
    #         raise RuntimeError("Annotation is None")
    #
    #     origin = get_origin(field.annotation)
    #     # print(f"Field: {info.field_name}")
    #     # print(f"Origin: {origin}")
    #     params = deepcopy(info.data["params"])
    #     # print(params)
    #
    #     if origin == list:
    #         # we may need to inject params into the list children...
    #         type_args = get_args(field.annotation)
    #         if len(type_args) == 0:
    #             raise ValueError("There should be one type argument to List[...]")
    #         type_arg = type_args[0]
    #         if issubclass(type_arg, ParameterisedModel):
    #             data = [{"params": params, **item} for item in data if "params" not in item]
    #     elif origin is Annotated:
    #         pass
    #     elif is_optional(field.annotation):
    #         type_arg = get_args(field.annotation)[0]
    #         if get_origin(type_arg) is not Annotated and issubclass(type_arg, ParameterisedModel):
    #             data["params"] = params
    #     elif issubclass(field.annotation, ParameterisedModel):
    #         data["params"] = params
    #
    #     return data

    @model_validator(mode="before")
    @classmethod
    def _store_params(cls, data: dict):
        """...

        When processing `Population`, I've got everything in `data`. How can I pass that
            into all of the `ParameterisedTensor` fields?
        """
        if "params" not in data:
            _data = deepcopy(data)
            _data["params"] = data.copy()
            data = _data

        return data
