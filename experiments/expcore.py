import subprocess
import logging
import os
from pathlib import Path
import yaml
import sys
import math


class InputGraph:
    def __init__(self, name, triangles=None):
        self.name = name
        self.triangles = triangles

    def args(self, p):
        raise NotImplementedError()


class FileInputGraph(InputGraph):
    def __init__(self, name, path, format='metis', triangles=None):
        self.name = name
        self.path = path
        self.format = format
        self.triangles = triangles

    def args(self, p):
        file_args = [str(self.path), "--input-format", self.format]
        return file_args

    def exists(self):
        if self.format == "metis":
            return self.path.exists()
        elif self.format == "binary":
            root = self.path.parent
            first_out = root / (self.path.stem + ".first_out")
            head = root / (self.path.stem + ".head")
            return first_out.exists() and head.exists()

    def __repr__(self):
        return f"FileInputGraph({self.name, self.triangles, self.path, self.format})"


class GenInputGraph(InputGraph):

    parameter_list = {
        "rhg": ["avg_degree", "gamma"],
        "rgg": ["radius"],
        "rdg_2d": [],
        "rdg_3d": [],
    }

    def __init__(self, generator, **kwargs):
        if generator not in GenInputGraph.parameter_list.keys():
            raise ValueError(f"Generator {generator} is not supported.")
        self.generator = generator
        self.params = kwargs
        self.scale_weak = self.params.get("scale_weak", False);
        required_parameters = set(
            GenInputGraph.parameter_list[self.generator] + ['n'])
        if not required_parameters.issubset(self.params.keys()):
            raise ValueError(
                f"Generator {self.generator} requires the following parameters: {required_parameters}"
            )

    def args(self, p):
        arg_list = ["--gen"]
        if self.generator == 'rgg':
            arg_list.append('rgg_2d')
        elif self.generator == 'rhg':
            arg_list.append('rhg')
        elif self.generator == 'rdg_2d':
            arg_list.append('rgg_2d')
        elif self.generator == 'rdg_3d':
            arg_list.append('rgg_3d')
        arg_list.append("--gen_n")
        if self.scale_weak:
            if not math.log2(p).is_integer():
                sys.exit("Number of PEs must be a power of two")
            scaled_n = self.params["n"] + int(math.log2(p));
            arg_list.append(str(scaled_n))
            arg_list.append("--gen_scale_weak")
        else:
            arg_list.append(str(self.params["n"]))
        if self.generator == 'rgg':
            arg_list.append("--gen_r_coeff")
            arg_list.append(str(self.params["radius"]))
        elif self.generator == 'rhg':
            arg_list.append("--gen_gamma")
            arg_list.append(str(self.params["gamma"]))
            arg_list.append("--gen_d")
            arg_list.append(str(self.params["avg_degree"]))
            arg_list.append("--rhg-fix")
        else:
            arg_list.append("--rhg-fix")
        return arg_list

    @property
    def name(self):
        n = self.params["n"]
        name = f"{self.generator.upper()}(2**{n}"
        for key in GenInputGraph.parameter_list[self.generator]:
            val = self.params[key]
            name += f", {val}"
        name += ")"
        if self.scale_weak:
            name += "_weak"
        return name

def load_inputs_from_yaml(yaml_path):
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
    inputs = {}
    for rec in data["graphs"]:
        name = rec['name']
        path = Path(yaml_path).parent / rec['path']
        format = rec['format']
        triangles = rec.get('triangles')
        graph = FileInputGraph(name, path, format, triangles)
        if not graph.exists():
            logging.warn(f"Could not load graph {graph.name}")
        else:
            inputs[graph.name] = graph
    if "includes" in data:
        for rec in data["includes"]:
            inputs.update(load_inputs_from_yaml(Path(yaml_path).parent / rec))
    return inputs


class ExperimentSuite:
    def __init__(self,
                 name: str,
                 cores=[],
                 threads_per_rank=[1],
                 inputs=[],
                 configs=[],
                 tasks_per_node=None,
                 time_limit=None,
                 input_time_limit={}):
        self.name = name
        self.cores = cores
        self.threads_per_rank = threads_per_rank
        self.inputs = inputs
        self.configs = configs
        self.tasks_per_node = tasks_per_node
        self.time_limit = time_limit
        self.input_time_limit = input_time_limit

    def set_input_time_limit(self, input_name, time_limit):
        self.input_time_limit[input_name] = time_limit

    def get_input_time_limit(self, input_name):
        return self.input_time_limit.get(input_name, self.time_limit)

    def load_inputs(self, input_dict):
        inputs_new = []
        for graph in self.inputs:
            if isinstance(graph, str):
                input = input_dict.get(graph)
                if not input:
                    logging.warn(f"Could not load input for {graph}")
                inputs_new.append(input)
            else:
                inputs_new.append(graph)
        self.inputs = inputs_new

    def __repr__(self):
        return f"ExperimentSuite({self.name}, {self.cores}, {self.inputs}, {self.configs}, {self.time_limit}, {self.input_time_limit})"


def load_suite_from_yaml(path):
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    configs = []
    if type(data["config"]) == list:
        for config in data["config"]:
            configs = configs + explode(config)
    else:
        configs = explode(data["config"])
    inputs = []
    time_limits = {}
    for graph in data["graphs"]:
        if type(graph) == str:
            inputs.append(graph)
        else:
            if "name" in graph:
                inputs.append(graph["name"])
            elif "generator" in graph:
                generator = graph.pop("generator")
                inputs.append(GenInputGraph(generator, **graph))
            time_limit = graph.get("time_limit")
            if time_limit:
                time_limits[graph["name"]] = time_limit
    return ExperimentSuite(data["name"],
                           data["ncores"],
                           data.get("threads_per_rank", [1]),
                           inputs,
                           configs,
                           tasks_per_node=data.get("tasks_per_node"),
                           time_limit=data.get("time_limit"),
                           input_time_limit=time_limits)


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


def cetric_command(input, mpi_ranks, threads_per_rank, **kwargs):
    script_path = os.path.dirname(__file__)
    build_dir = Path(
        os.environ.get("BUILD_DIR", os.path.join(script_path, "../build/")))
    app = build_dir / "apps" / "cetric"
    command = [str(app)]
    if input:
        if isinstance(input, InputGraph):
            command = command + input.args(mpi_ranks)
        else:
            command.append(str(input))
    flags = ["--num-threads", str(threads_per_rank)]
    for flag, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                flags.append("--" + flag)
        else:
            flags.append("--" + flag)
            flags.append(str(value))
    command = command + flags
    return command
