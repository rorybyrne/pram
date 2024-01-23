from typing import Type, Union

from typing_extensions import get_args, get_origin


def is_optional(type_: Type):
    """Check whether the field is typing.Optional"""
    return get_origin(type_) is Union and type(None) in get_args(type_)
