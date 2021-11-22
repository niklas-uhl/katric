import subprocess
import logging
import os
from pathlib import Path
import yaml


class InputGraph:
    def __init__(self, name):
        self.name = name

    def args(self):
        raise NotImplementedError()


class FileInputGraph(InputGraph):
    def __init__(self, name, path, format='metis'):
        self.name = name
        self.path = path
        self.format = format

    def args(self):
        file_args = [str(self.path), "--input-format", self.format]
        return file_args

    def __repr__(self):
        return f"FileInputGraph({self.name, self.path, self.format})"


class GenInputGraph(InputGraph):
    pass


def load_inputs_from_yaml(yaml_path):
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
    inputs = {}
    for rec in data["graphs"]:
        name = rec['name']
        path = Path(yaml_path).parent / rec['path']
        format = rec['format']
        inputs[name] = FileInputGraph(name, path, format)
    if "includes" in data:
        for rec in data["includes"]:
            inputs |= load_inputs_from_yaml(Path(yaml_path).parent / rec)
    return inputs


class ExperimentSuite:
    def __init__(self,
                 name: str,
                 PEs: list[int] = [],
                 inputs=[],
                 configs: list[dict] = [],
                 tasks_per_node = None,
                 time_limit = None):
        self.name = name
        self.PEs = PEs
        self.inputs = inputs
        self.configs = configs
        self.tasks_per_node = tasks_per_node
        self.time_limit = time_limit

    def load_inputs(self, input_dict):
        inputs_new = []
        for name in self.inputs:
            input = input_dict.get(name)
            if not input:
                logging.warn(f"Could not load input for {name}")
            inputs_new.append(input)
        self.inputs = inputs_new

    def __repr__(self):
        return f"ExperimentSuite({self.name}, {self.PEs}, {self.inputs}, {self.configs}, {self.time_limit})"


def load_suite_from_yaml(path):
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    configs = []
    if type(data["config"]) == list:
        for config in data["config"]:
            configs = configs + explode(config)
    else:
        configs = explode(data["config"])
    return ExperimentSuite(data["name"], data["ncores"], data["graphs"],
                           configs, tasks_per_node=data.get("tasks_per_node"), time_limit=data.get("time_limit"))


def explode(config):
    configs = []
    for flag, value in config.items():
        if type(value) == list:
            for arg in value:
                exploded = config.copy()
                exploded[flag] = arg
                exp = explode(exploded)
                configs = configs + exp
            break
    if not configs:
        return [config]
    return configs


def cetric_command(ncores, input, **kwargs):
    script_path = os.path.dirname(__file__)
    build_dir = Path(
        os.environ.get("BUILD_DIR", os.path.join(script_path,
                                                 "../build/apps")))
    app = build_dir / "cetric"
    command = [str(app)]
    if input:
        if isinstance(input, InputGraph):
            command = command + input.args()
        else:
            command.append(str(input))
    flags = []
    for flag, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                flags.append("--" + flag)
        else:
            flags.append("--" + flag)
            flags.append(str(value))
    command = command + flags
    return command
