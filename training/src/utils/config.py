"""Utility module for managing configuration settings in a structured and immutable way."""
from __future__ import annotations
import copy
import logging
from pprint import pformat
from typing import Any, Optional, Iterable
import os
import yaml
import collections.abc


def override(d, u):
    if not d:
        return u
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = override(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class ImmutabilityError(Exception):
    """Exception raised for errors linked to the modification of immutable config object.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="This object is immutable and cannot be edited"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"


class ImmutableDict(dict):
    _lock = False

    def __init__(self, current_dict: Optional[dict] = None) -> None:
        if current_dict is not None:
            self._set_config(current_dict)

    def __setattr__(self, __name: str, __value: Any) -> None:
        if self._lock:
            raise ImmutabilityError("You cannot modify a read only dict")
        super().__setattr__(__name, __value)

    def __setitem__(self, __name: str, __value: Any) -> None:
        if self._lock:
            raise ImmutabilityError("You cannot modify a read only dict")
        super().__setitem__(__name, __value)

    def _set_config(self, current_dict: dict) -> None:
        for k, v in current_dict.items():
            if type(v) is dict:
                local_immutable_dict = ImmutableDict(v)
                self.__setattr__(k, local_immutable_dict)
                self.__setitem__(k, local_immutable_dict)
            elif type(v) is list or type(v) is tuple or type(v) is set:
                local_tuple = build_iterable(v)
                self.__setattr__(k, local_tuple)
                self.__setitem__(k, local_tuple)
            else:
                self.__setattr__(k, v)
                self.__setitem__(k, v)
        self._lock = True

    def set_nested_value(self, keys: list[str, int], value):
        """
        Set a value in a nested dict.
        Keys is a list of keys to access the value to set. it can be both string or int.
        When it is an integer, it is used to access a tuple element of immutable dict.
        """
        # TODO: handle when self.__getattribute__(keys[0]) is a tuple of immutable dict
        if len(keys) > 1:
            key, keys = keys[0], keys[1:]

            if isinstance(keys[0], str):
                to_set: ImmutableDict = self.__getattribute__(key)
            else:
                index, keys = keys[0], keys[1:]
                to_set: ImmutableDict = self.__getattribute__(key)[index]
            to_set.set_nested_value(keys, value)
        else:
            self.set_param(keys[0], value)

    def set_param(self, name: str, value: Any):
        if not hasattr(self, name):
            raise AttributeError(
                f"You try to set an unknown attribute: {name}. Use 'add_param' instead to add a new param to your config."
            )
        logging.warning(
            f"Changing param '{name}' from '{getattr(self, name)}' to '{value}'"
        )
        super().__setattr__(name, value)
        super().__setitem__(name, value)

    def add_param(self, name: str, value: Any):
        logging.warning(f"Adding new param '{name}' with value = '{value}'")
        super().__setattr__(name, value)
        super().__setitem__(name, value)

    def to_dict(self):
        def handle_list_tuple_set(list_obj):
            output = []
            for x in list_obj:
                if isinstance(x, list) or isinstance(x, tuple) or isinstance(x, set):
                    element = handle_list_tuple_set(x)
                elif isinstance(x, ImmutableDict):
                    element = x.to_dict()
                else:
                    element = x
                output.append(element)
            return output

        current_dict = {}
        for k in list(self.keys()):
            if isinstance(self[k], ImmutableDict):
                current_dict[k] = self[k].to_dict()
                continue
            if (
                isinstance(self[k], list)
                or isinstance(self[k], tuple)
                or isinstance(self[k], set)
            ):
                current_dict[k] = handle_list_tuple_set(self[k])
                continue
            current_dict[k] = self[k]
        return current_dict

    def to_yaml(self, file: str) -> None:
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, "w") as f:
            logging.info(f"Writing config into file {file}")
            yaml.safe_dump(self.to_dict(), f, indent=4, sort_keys=False)

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.to_dict())


def build_iterable(current_iterable: Iterable) -> tuple:
    tmp_list: list = []
    for v in current_iterable:
        if type(v) is dict:
            tmp_list.append(ImmutableDict(v))
        elif type(v) is list or type(v) is tuple or type(v) is set:
            tmp_list.append(build_iterable(v))
        else:
            tmp_list.append(v)
    return tuple(tmp_list)


class Config(ImmutableDict):
    # _instance: Optional["Config"] = None

    def __new__(class_, *args, **kwargs) -> "Config":
        class_._instance: Optional["Config"] = None
        if not isinstance(class_._instance, class_):
            class_._instance = ImmutableDict.__new__(class_, *args, **kwargs)
        return class_._instance

    def __str__(
        self,
    ) -> str:
        output = f"{pformat(self, sort_dicts=False, indent=4)}"
        return output

    def set_config(
        self,
        yaml_conf_path: str | list[str] | None = None,
        list_dict: dict | list[dict] | None = None,
        additional_parameters: dict | None = {},
        yaml_conf_path_override: str | None = None,
    ) -> None:
        global_conf: dict = {}
        if yaml_conf_path is not None:
            logging.info(f"Setting config with file(s) {yaml_conf_path}")
            if isinstance(yaml_conf_path, str):
                yaml_conf_path = [yaml_conf_path]

            for conf_path in yaml_conf_path:
                logging.info(f"Loading file {conf_path} and updating config")
                with open(conf_path) as f:
                    cfg: dict = yaml.safe_load(f)
                    global_conf.update(cfg)
        if list_dict is not None:
            if isinstance(list_dict, dict):
                list_dict = [list_dict]
            for cfg in list_dict:
                global_conf.update(cfg)
        if additional_parameters:
            assert isinstance(additional_parameters, dict)
            logging.info(
                f"Updating config with additional parameters {pformat(additional_parameters, sort_dicts=False, indent=4)}"
            )
            global_conf.update(
                {"additional_parameters_passed_inside_code": additional_parameters}
            )
        if yaml_conf_path_override is not None:
            logging.warning(
                f"Overriding default config with config {yaml_conf_path_override}"
            )
            with open(yaml_conf_path_override) as f:
                cfg: dict = yaml.safe_load(f)
            global_conf = override(global_conf, cfg)
        self._set_config(global_conf)

    def __getattribute__(self, __name: str) -> Config:
        return super().__getattribute__(__name)


def get_config() -> Config:
    return Config()


def reset_config() -> Any:
    Config._instance = None
