from expcore import ExperimentSuite, explode, FileInputGraph, GenInputGraph
import expcore
from pathlib import Path
import subprocess, sys, json, os
import math as m
from string import Template
import time


class SharedMemoryRunner:
    def __init__(self, output_directory, verify_results = False):
        self.output_directory = Path(output_directory)
        self.verify_results = verify_results
        self.failed = 0
        self.incorrect = 0

    def execute(self, experiment_suite: ExperimentSuite):
        print(f"Running suite {experiment_suite.name} ...")
        output_path = self.output_directory / experiment_suite.name
        output_path.mkdir(exist_ok=True, parents=True)
        with open(output_path / "config.json", 'w') as file:
            json.dump(experiment_suite.configs, file, indent=4)
        for i, config in enumerate(experiment_suite.configs):
            for input in experiment_suite.inputs:
                for ncores in experiment_suite.PEs:
                    config['json-output'] = 'stdout'
                    if isinstance(input, expcore.InputGraph):
                        input_name = input.name
                    else:
                        input_name = str(input)
                    log_path = output_path / f"{input_name}-np{ncores}-c{i}-log.txt"
                    err_path = output_path / f"{input_name}-np{ncores}-c{i}-err.txt"
                    mpiexec = os.environ.get("MPI_EXEC", "mpiexec")
                    cmd = mpiexec.split(" ")
                    cmd += ["-np", str(ncores)]
                    cmd += expcore.cetric_command(input, **config)
                    print(
                        f"Running config {i} on {input_name} using {ncores} cores ... ",
                        end='')
                    sys.stdout.flush()
                    print(" ".join(cmd))
                    with open(log_path, 'w') as log_file:
                        with open(err_path, 'w') as err_file:
                            ret = subprocess.run(cmd,
                                                 stdout=log_file,
                                                 stderr=err_file)
                    if ret.returncode == 0:
                        if self.verify_results and input.triangles:
                            print('finished.', end='')
                            with open(log_path) as output:
                                triangles = int(json.load(output)["stats"]["counted_triangles"])
                            if triangles == input.triangles:
                                print(' correct.')
                            else:
                                self.incorrect += 1
                                print(' incorrect.')
                        else:
                            print('finished.')
                    else:
                        self.failed += 1
                        print('failed.')
        print(f"Finished suite {experiment_suite.name}.")


def get_queue(cores, tasks_per_node):
    nodes = required_nodes(cores, tasks_per_node)
    if nodes <= 16:
        return "micro"
    else:
        return "general"


def required_nodes(cores, tasks_per_node):
    return int(max(int(m.ceil(float(cores) / tasks_per_node)), 1))


class SBatchRunner:
    def __init__(self, output_directory, job_output_directory, tasks_per_node,
                 time_limit, use_test_partition = False):
        self.output_directory = Path(output_directory)
        self.job_output_directory = Path(job_output_directory)
        self.tasks_per_node = tasks_per_node
        self.time_limit = time_limit
        self.use_test_partition = use_test_partition

    def execute(self, experiment_suite: ExperimentSuite):
        project = os.environ["PROJECT"]
        output_path = self.output_directory / experiment_suite.name
        output_path.mkdir(exist_ok=True, parents=True)
        with open(output_path / "config.json", 'w') as file:
            json.dump(experiment_suite.configs, file, indent=4)
        njobs = 0
        for i, config in enumerate(experiment_suite.configs):
            for input in experiment_suite.inputs:
                for ncores in experiment_suite.PEs:
                    config['json-output'] = 'stdout'
                    if isinstance(input, expcore.InputGraph):
                        input_name = input.name
                    else:
                        input_name = str(input)
                    log_path = output_path / f"{input_name}-np{ncores}-c{i}-log.txt"
                    err_path = output_path / f"{input_name}-np{ncores}-c{i}-err.txt"
                    cmd = expcore.cetric_command(input, **config)
                    jobname = f"{experiment_suite.name}-{input_name}-np{ncores}-c{i}"
                    script_path = Path(os.path.dirname(__file__))
                    with open(script_path /
                              "sbatch-template.txt") as template_file:
                        template = template_file.read()
                    template = Template(template)
                    if experiment_suite.tasks_per_node:
                        tasks_per_node = experiment_suite.tasks_per_node
                    else:
                        tasks_per_node = self.tasks_per_node
                    subs = {}
                    subs["nodes"] = required_nodes(ncores, tasks_per_node)
                    subs["p"] = ncores
                    subs["output_log"] = str(log_path)
                    subs["error_log"] = str(err_path)
                    subs["job_name"] = jobname
                    if self.use_test_partition:
                        subs["job_queue"] = "test"
                    else:
                        subs["job_queue"] = get_queue(ncores, tasks_per_node)
                    time_limit = experiment_suite.get_input_time_limit(
                        input.name)
                    if not time_limit:
                        time_limit = self.time_limit
                    subs["time_string"] = time.strftime(
                        "%H:%M:%S", time.gmtime(time_limit * 60))
                    subs["account"] = project
                    subs["cmd"] = ' '.join(cmd)
                    job_script = template.substitute(subs)
                    job_file = self.job_output_directory / jobname
                    with open(job_file, "w") as job:
                        job.write(job_script)
                    njobs += 1
        print(f"Created {njobs} job files in directory {self.job_output_directory}.")
