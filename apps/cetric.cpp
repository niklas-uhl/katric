#include <mpi.h>
#include <CLI/CLI.hpp>
#include <util.h>
#include <config.h>
#include "parse_parameters.h"
#include <io/distributed_graph_io.h>
#include <counters/cetric.h>
#include <nlohmann/json.hpp>
#include <unistd.h>

Config parse_config(int argc, char* argv[], PEID rank, PEID size) {
    (void) size;

    CLI::App app("Parallel Triangle Counter");

    Config conf;

    //conf.hostname = std::getenv("HOST");
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    conf.hostname = hostname;

    app.add_option("input", conf.input_file, "The input graph");

    app.add_option("--input-format", conf.input_format)
        ->transform(CLI::CheckedTransformer(input_types, CLI::ignore_case));

    app.add_option("--iterations", conf.iterations);

    app.add_set("--cost-function", conf.cost_function, {"N", "D", "DH", "DDH", "DH2", "DPD", "IDPD", "D*"});

    app.add_flag("-v", conf.verbosity_level, "verbosity level");

    //app.add_option("--iterations", conf.iterations);
    app.add_flag("--json-output", conf.json_output);

    app.add_flag("--degree-filtering", conf.degree_filtering);

    parse_gen_parameters(app, conf);

    CLI::Option* input_option = app.get_option("input");
    CLI::Option* gen_option = app.get_option("--gen");
    input_option->excludes(gen_option);
    gen_option->excludes(input_option);

    CLI11_PARSE_MPI(app, argc, argv, rank, size);
    if (conf.input_file.empty() && conf.gen.empty()) {
        int retval;
        if (rank == 0) {
            CLI::RequiredError e("Providing an input file via 'input' or using the generator via '--gen'");
            retval = app.exit(e);
        } else {
            retval = 1;
        }
        MPI_Finalize();
        exit(retval);
    }
    conf.rank = rank;
    conf.PEs = size;
    return conf;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    PEID rank;
    PEID size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    DEBUG_BARRIER(rank);

    Config conf = parse_config(argc, argv, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    cetric::profiling::Timer global_time;

    DistributedGraph G;
    cetric::profiling::Statistics stats(rank, size);
    cetric::profiling::Timer timer;
    if (conf.gen == "") {
        G = cetric::read_local_graph(conf.input_file, conf.input_format, rank, size);
    } else {
        G = cetric::gen_local_graph(conf, rank, size);
    }
    stats.local.io_time = timer.elapsed_time();

    
    run_cetric(G, stats, conf, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    stats.global_wall_time = global_time.elapsed_time();

    stats.reduce();
    if (conf.json_output) {
        if (rank == 0) {
            auto output = nlohmann::json(stats);
            output["config"] = conf;
            std::cout << nlohmann::json(output).dump(4) << std::endl;
        }
    }
    return MPI_Finalize();
}
