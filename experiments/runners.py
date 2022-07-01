from expcore import ExperimentSuite, explode, FileInputGraph, GenInputGraph
import expcore
from pathlib import Path
import subprocess, sys, json, os
import math as m
from string import Template
import time
import slugify


class SharedMemoryRunner:
    def __init__(self, output_directory, verify_results=False):
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
                for ncores in experiment_suite.cores:
                    print(experiment_suite.threads_per_rank)
                    for threads in experiment_suite.threads_per_rank:
                        mpi_ranks = ncores // threads
                        config['json-output'] = 'stdout'
                        if isinstance(input, expcore.InputGraph):
                            input_name = input.name
                        else:
                            input_name = str(input)
                        log_path = output_path / f"{input_name}-np{mpi_ranks}-t{threads}-c{i}-log.txt"
                        err_path = output_path / f"{input_name}-np{mpi_ranks}-t{threads}-c{i}-err.txt"
                        mpiexec = os.environ.get("MPI_EXEC", "mpiexec")
                        cmd = mpiexec.split(" ")
                        cmd += ["-np", str(mpi_ranks)]
                        cmd += expcore.cetric_command(input, mpi_ranks,
                                                      threads, **config)
                        print(
                            f"Running config {i} on {input_name} using {mpi_ranks} ranks and {threads} threads per rank ... ",
                            end='')
                        print(cmd)
                        sys.stdout.flush()
                        with open(log_path, 'w') as log_file:
                            with open(err_path, 'w') as err_file:
                                ret = subprocess.run(cmd,
                                                     stdout=log_file,
                                                     stderr=err_file)
                        if ret.returncode == 0:
                            if self.verify_results and input.triangles:
                                print('finished.', end='')
                                with open(log_path) as output:
                                    triangles = int(
                                        json.load(output)["stats"][0]
                                        ["counted_triangles"])
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
    def __init__(self,
                 output_directory,
                 job_output_directory,
                 tasks_per_node,
                 time_limit,
                 use_test_partition=False):
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
        script_path = Path(os.path.dirname(__file__))
        with open(script_path / "sbatch-template.txt") as template_file:
            template = template_file.read()
        template = Template(template)
        with open(script_path / "command-template.txt") as template_file:
            command_template = template_file.read()
        command_template = Template(command_template)
        njobs = 0
        for input in experiment_suite.inputs:
            if isinstance(input, expcore.InputGraph):
                input_name = input.name
            else:
                input_name = str(input)
            for ncores in experiment_suite.cores:
                if experiment_suite.tasks_per_node:
                    tasks_per_node = experiment_suite.tasks_per_node
                else:
                    tasks_per_node = self.tasks_per_node
                aggregate_jobname = f"{experiment_suite.name}-{input_name}-cores{ncores}"
                log_path = output_path / f"{input_name}-cores{ncores}-log.txt"
                subs = {}
                subs["nodes"] = required_nodes(ncores, tasks_per_node)
                subs["output_log"] = str(log_path)
                subs["job_name"] = aggregate_jobname
                if self.use_test_partition:
                    subs["job_queue"] = "test"
                else:
                    subs["job_queue"] = get_queue(ncores, tasks_per_node)
                subs["account"] = project
                time_limit = 0
                commands = []
                for threads_per_rank in experiment_suite.threads_per_rank:
                    mpi_ranks = ncores // threads_per_rank
                    ranks_per_node = tasks_per_node // threads_per_rank
                    jobname = f"{experiment_suite.name}-{input_name}-np{mpi_ranks}-t{threads_per_rank}"
                    for i, config in enumerate(experiment_suite.configs):
                        json_path = output_path / f"{input_name}-np{mpi_ranks}-t{threads_per_rank}-log-c{i}.json"
                        config['json-output'] = str(json_path)
                        job_time_limit = experiment_suite.get_input_time_limit(
                            input.name)
                        if not job_time_limit:
                            job_time_limit = self.time_limit
                        time_limit += job_time_limit
                        cmd = expcore.cetric_command(input, ncores,
                                                     threads_per_rank,
                                                     **config)
                        config_jobname = jobname + "-c" + str(i)
                        cmd_string = command_template.substitute(
                            cmd=" ".join(cmd),
                            jobname=config_jobname,
                            mpi_ranks=mpi_ranks,
                            threads_per_rank=threads_per_rank,
                            ranks_per_node=ranks_per_node,
                            timeout=job_time_limit * 60)
                        commands.append(cmd_string)
                subs["commands"] = '\n'.join(commands)
                subs["time_string"] = time.strftime(
                    "%H:%M:%S", time.gmtime(time_limit * 60))
                job_script = template.substitute(subs)
                job_file = self.job_output_directory / aggregate_jobname
                with open(job_file, "w") as job:
                    job.write(job_script)
                njobs += 1
        print(
            f"Created {njobs} job files in directory {self.job_output_directory}."
        )
