from runners import *
import expcore
import argparse, os, sys
from pathlib import Path


def load_suites(suite_files, search_paths):
    suites = {}
    for path in suite_files:
        suite = expcore.load_suite_from_yaml(path)
        suites[suite.name] = suite
    for path in search_paths:
        for file in os.listdir(path):
            if file.endswith('.suite.yaml'):
                suite = expcore.load_suite_from_yaml(os.path.join(path, file))
                suites[suite.name] = suite
    return suites


def load_inputs(input_descriptions):
    inputs = {}
    for path in input_descriptions:
        inputs.update(expcore.load_inputs_from_yaml(path))
    return inputs


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('suite', nargs='*')

    default_search_dirs = os.environ.get("SUITE_SEARCH_PATH",
                                         default=os.getcwd()).split(":")
    default_search_dirs.append(os.path.dirname(__file__))
    parser.add_argument('-d',
                        '--search-dirs',
                        nargs='*',
                        default=default_search_dirs)
    parser.add_argument('-s', '--suite-files', default=[])

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

    default_output_dir = os.environ.get("OUTPUT_DIR",
                                        Path(os.getcwd()) / "output")
    parser.add_argument('-o', '--output-dir', default=default_output_dir)

    parser.add_argument('-l', '--list', action='store_true')
    parser.add_argument('-v', '--verify', action='store_true')

    parser.add_argument('-g', '--list-graphs', action='store_true')

    parser.add_argument('-j', '--job-output-dir', default="jobs")

    default_machine_type = os.environ.get("MACHINE", 'shared')
    parser.add_argument('-m', '--machine', choices=['shared', 'supermuc'], default = default_machine_type)
    parser.add_argument('--tasks-per-node', default = os.environ.get("TASKS_PER_NODE", 48), type=int)
    parser.add_argument('-t', '--time-limit', default = os.environ.get("TIME_LIMIT", 20), type=int)

    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()
    print(args.search_dirs)
    suites = load_suites(args.suite_files, args.search_dirs)
    inputs = load_inputs(args.input_descriptions + default_inputs)
    for suite in suites.values():
        suite.load_inputs(inputs)

    if args.list:
        for name in suites.keys():
            print(name)
        sys.exit(0)
    if args.list_graphs:
        for name in inputs.keys():
            print(name)
        sys.exit(0)

    if not args.suite:
        args.suite = suites.keys()

    if args.machine == 'shared':
        runner = SharedMemoryRunner(args.output_dir, verify_results=args.verify)
    else:
        runner = SBatchRunner(args.output_dir, args.job_output_dir, args.tasks_per_node, args.time_limit, args.test)
    for suitename in args.suite:
        suite = suites.get(suitename)
        if suite:
            runner.execute(suite)

    if args.machine == 'shared':
        print(f"Summary: {runner.failed} jobs failed, {runner.incorrect} jobs returned an incorrect result.")

if __name__ == "__main__":
    main()
