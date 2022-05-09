import yaml
import logging
import argparse
import os
from pathlib import Path
import importlib
import subprocess

run_experiments = importlib.import_module("run-experiments")
load_inputs = run_experiments.load_inputs
from expcore import load_suite_from_yaml, params_to_flags


class FileInputGraph:
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('suite')
    default_inputs = os.environ.get("INPUT_DESCRIPTIONS", default=[])
    if default_inputs:
        default_inputs = default_inputs.split(":")
    default_inputs.append(
        Path(os.path.dirname(__file__)) / ".." / "examples" / 'examples.yaml')
    parser.add_argument('-i', '--input-descriptions', nargs='*', default=[])
    args = parser.parse_args()
    inputs = load_inputs(args.input_descriptions + default_inputs)
    suite = load_suite_from_yaml(args.suite)
    with open(args.suite, "r") as file:
        data = yaml.safe_load(file)
    exec_path = Path(args.suite).absolute().parent / data["binary"]
    suite.load_inputs(inputs)
    for input in suite.inputs:
        if input is None:
            continue
        for threads in suite.cores:
            for config in suite.configs:
                cmd = [str(exec_path)] \
                    + input.args(threads) \
                    + ["--num_threads", str(threads)] \
                    + params_to_flags(config)
                ret = subprocess.run(cmd)

if __name__ == "__main__":
    main()
