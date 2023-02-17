# How to run our experiments

We use a custom experiment runner, which requires python.
Install the dependencies listed in `requirements.txt`.

An experiment is described using a YAML file which consists of the following:
- processor configurations used
- input graphs (from files or generated)
- parameter combinations to run

See [`example.suite.yaml`](./example.suite.yaml) for an example.

To run a experiment suite, run 
```sh
python run-experiments.py <suite-name>[...] --machine <machine-type> --search-dirs <search-dirs...>
```

- `<suite-name>` is the name of the suite to run (you can pass multiple)
- `<machine-type>` is any of `shared` or `supermuc`
    - `shared` runs the experiments directly as subprocesses
    - `supermuc` generates SLURM job files for SuperMUC-NG (you can adapt [`sbatch-template.txt`](./sbatch-template.txt) to your system)
- `<search-dirs>` directories to search for `suite.yaml` files. By default this includes only the current working directory. 

The runner uses symbolic names to refer to graphs on disk. A graph collection can be specified using a YAML file. 
See [`../examples/examples.yaml`](../examples/examples.yaml) for an example. 
This graph collection is loaded by default, use the `--input-descriptions` flag to pass additional ones.

# Reproducibility
The suites for rerunning the experiments from our publication are located in the `paper-suites` directory.
You have to add it explicitely using `--search-dirs`.

The real world graphs used are not part of this repository, but their sources are listed in the paper.
Note that we default to using up to 32K cores, so carefully adapt the suite files to your needs.

The gathered metrics are outputted using the human-readable JSON format to 
the directory specified by the `OUTPUT_DIR` environment variable or the `--output-dir` option.

