#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "rnn/rnn_genome.hxx"
#include "time_series/time_series.hxx"

/**
 * Diagnostic for the genome-copy corruption bug (depth-sort tie instability).
 *
 * Loads a genome, calls RNN_Genome::copy() — the exact operation the island
 * strategy performs when stashing the global best — and evaluates BOTH the
 * original and the copy on the same data with their best_parameters.
 *
 * A healthy copy() must produce identical MSE. Any difference means the
 * constructor's node re-sort permuted equal-depth nodes relative to
 * best_parameters (see rnn/rnn_node_interface.hxx sort_RNN_Nodes_by_depth).
 *
 * Exit 0 = copy faithful, exit 1 = copy corrupted the genome.
 */

vector<string> arguments;

vector<vector<vector<double> > > testing_inputs;
vector<vector<vector<double> > > testing_outputs;

int main(int argc, char** argv) {
    arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_id("main");

    string genome_filename;
    get_argument(arguments, "--genome_file", true, genome_filename);
    RNN_Genome* genome = new RNN_Genome(genome_filename);

    vector<string> testing_filenames;
    get_argument_vector(arguments, "--testing_filenames", true, testing_filenames);

    TimeSeriesSets* time_series_sets = TimeSeriesSets::generate_test(
        testing_filenames, genome->get_input_parameter_names(), genome->get_output_parameter_names()
    );

    string normalize_type = genome->get_normalize_type();
    if (normalize_type.compare("min_max") == 0) {
        time_series_sets->normalize_min_max(genome->get_normalize_mins(), genome->get_normalize_maxs());
    } else if (normalize_type.compare("avg_std_dev") == 0) {
        time_series_sets->normalize_avg_std_dev(
            genome->get_normalize_avgs(), genome->get_normalize_std_devs(), genome->get_normalize_mins(),
            genome->get_normalize_maxs()
        );
    }

    int32_t time_offset = 1;
    get_argument(arguments, "--time_offset", true, time_offset);

    time_series_sets->export_test_series(time_offset, testing_inputs, testing_outputs);

    RNN_Genome* copied = genome->copy();

    vector<double> original_parameters = genome->get_best_parameters();
    vector<double> copied_parameters = copied->get_best_parameters();

    double original_mse = genome->get_mse(original_parameters, testing_inputs, testing_outputs);
    double copied_mse = copied->get_mse(copied_parameters, testing_inputs, testing_outputs);

    Log::info("original MSE: %.10lf\n", original_mse);
    Log::info("copy     MSE: %.10lf\n", copied_mse);

    // sorting can change edge iteration order, which perturbs floating-point
    // accumulation slightly -- allow 0.1% before calling the copy corrupted
    double ratio = copied_mse / original_mse;

    Log::release_id("main");

    if (ratio >= 0.999 && ratio <= 1.001) {
        Log::info("COPY OK: copy() preserved the genome (MSE ratio %.6lf)\n", ratio);
        return 0;
    } else {
        Log::info(
            "COPY CORRUPTED: copy() changed the genome's outputs (MSE ratio %.4lf; node order permuted vs "
            "best_parameters)\n",
            ratio
        );
        return 1;
    }
}
