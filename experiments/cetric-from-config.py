import expcore
import argparse
from pathlib import Path
import os, json

def load_inputs(input_descriptions):
    inputs = {}
    for path in input_descriptions:
        inputs |= expcore.load_inputs_from_yaml(path)
    return inputs

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('suite')

    default_inputs = os.environ.get("INPUT_DESCRIPTIONS", default = [])
    if default_inputs:
        default_inputs = default_inputs.split(":")
    default_inputs.append(
        Path(os.path.dirname(__file__)) / ".." / "examples" / 'examples.yaml')
    parser.add_argument('-i',
                        '--input-descriptions',
                        nargs='*',
                        default = []
                        )

    parser.add_argument('input')
    parser.add_argument('config', type=int)

    default_output_dir = os.environ.get("OUTPUT_DIR",
                                        Path(os.getcwd()) / "output")
    parser.add_argument('-o', '--output-dir', default=default_output_dir)

    args = parser.parse_args()
    inputs = load_inputs(args.input_descriptions + default_inputs)

    config_file = Path(args.output_dir) / args.suite / 'config.json'
    with open(config_file) as config:
        configs = json.load(config)
    config = configs[args.config]
    cmd = expcore.cetric_command(inputs[args.input], **config)
    print(" ".join(cmd))

if __name__ == "__main__":
    main()
