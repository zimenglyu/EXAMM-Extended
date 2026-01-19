#include <chrono>
#include <condition_variable>
using std::condition_variable;

#include <iomanip>
using std::setw;

#include <mutex>
using std::mutex;

#include <string>
using std::string;

#include <thread>
using std::thread;

#include <vector>
using std::vector;

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "rnn/rnn_genome.hxx"
#include "time_series/time_series.hxx"

vector<string> arguments;

vector<vector<vector<double> > > testing_inputs;
vector<vector<vector<double> > > testing_outputs;

int main(int argc, char** argv) {
    arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_id("main");

    string output_directory;
    get_argument(arguments, "--output_directory", true, output_directory);

    string genome_filename;
    get_argument(arguments, "--genome_file", true, genome_filename);
    RNN_Genome* genome = new RNN_Genome(genome_filename);

    vector<string> testing_filenames;
    get_argument_vector(arguments, "--testing_filenames", true, testing_filenames);

    TimeSeriesSets* time_series_sets = TimeSeriesSets::generate_test(
        testing_filenames, genome->get_input_parameter_names(), genome->get_output_parameter_names()
    );
    Log::debug("got time series sets.\n");

    string normalize_type = genome->get_normalize_type();
    if (normalize_type.compare("min_max") == 0) {
        time_series_sets->normalize_min_max(genome->get_normalize_mins(), genome->get_normalize_maxs());
    } else if (normalize_type.compare("avg_std_dev") == 0) {
        time_series_sets->normalize_avg_std_dev(
            genome->get_normalize_avgs(), genome->get_normalize_std_devs(), genome->get_normalize_mins(),
            genome->get_normalize_maxs()
        );
    }

    Log::info("normalized type: %s \n", normalize_type.c_str());

    int32_t time_offset = 1;
    get_argument(arguments, "--time_offset", true, time_offset);

    time_series_sets->export_test_series(time_offset, testing_inputs, testing_outputs);

    // Count total rows loaded for testing
    int32_t total_rows = 0;
    for (int32_t i = 0; i < (int32_t) testing_inputs.size(); i++) {
        if (testing_inputs[i].size() > 0) {
            total_rows += testing_inputs[i][0].size();
        }
    }
    Log::info("loaded %d rows for testing.\n", total_rows);

    vector<double> best_parameters = genome->get_best_parameters();
    Log::info("Parameter count: %zu\n", best_parameters.size());
    
    // Model Latency: End-to-end from input ready → prediction output
    // (includes inference + lightweight preprocessing + output formatting)
    auto model_latency_start = std::chrono::high_resolution_clock::now();
    
    // Inference Time: Model computation only (matrix ops, RNN steps, forward pass)
    // Excludes: data loading, preprocessing, disk I/O
    auto inference_start = std::chrono::high_resolution_clock::now();
    Log::info("MSE: %lf\n", genome->get_mse(best_parameters, testing_inputs, testing_outputs));
    Log::info("MAE: %lf\n", genome->get_mae(best_parameters, testing_inputs, testing_outputs));
    auto inference_end = std::chrono::high_resolution_clock::now();
    
    // Output formatting (part of model latency)
    genome->write_predictions(
        output_directory, testing_filenames, best_parameters, testing_inputs, testing_outputs, time_series_sets
    );
    auto model_latency_end = std::chrono::high_resolution_clock::now();
    
    // Calculate Inference Time (narrow, technical - model computation only)
    auto inference_duration = std::chrono::duration_cast<std::chrono::microseconds>(inference_end - inference_start);
    double inference_seconds = inference_duration.count() / 1000000.0;
    double inference_milliseconds = inference_seconds * 1000.0;
    
    // Calculate per-data-point latency
    double per_data_point_ms = (total_rows > 0) ? (inference_milliseconds / total_rows) : 0.0;
    double per_data_point_us = (total_rows > 0) ? (inference_duration.count() / (double)total_rows) : 0.0;
    
    Log::info("Inference time (entire dataset): %.3f seconds (%.1f ms)\n", inference_seconds, inference_milliseconds);
    Log::info("  Per data point: %.3f ms (%.1f μs)\n", per_data_point_ms, per_data_point_us);
    Log::info("  Throughput: %.1f data points/second\n", (total_rows > 0) ? (total_rows / inference_seconds) : 0.0);
    
    // Calculate Model Latency (end-to-end: input ready → prediction output)
    auto model_latency_duration = std::chrono::duration_cast<std::chrono::microseconds>(model_latency_end - model_latency_start);
    double model_latency_seconds = model_latency_duration.count() / 1000000.0;
    double model_latency_ms = model_latency_seconds * 1000.0;
    double per_data_point_model_latency_ms = (total_rows > 0) ? (model_latency_ms / total_rows) : 0.0;
    
    Log::info("Model latency (entire dataset): %.3f seconds (%.1f ms)\n", model_latency_seconds, model_latency_ms);
    Log::info("  Per data point: %.3f ms\n", per_data_point_model_latency_ms);
    if (Log::at_level(Log::DEBUG)) {
        int32_t length;
        char* byte_array;

        genome->write_to_array(&byte_array, length);

        Log::debug("WROTE TO BYTE ARRAY WITH LENGTH: %d\n", length);

        RNN_Genome* duplicate_genome = new RNN_Genome(byte_array, length);

        vector<double> best_parameters_2 = duplicate_genome->get_best_parameters();
        Log::debug(
            "duplicate MSE: %lf\n", duplicate_genome->get_mse(best_parameters_2, testing_inputs, testing_outputs)
        );
        Log::debug(
            "duplicate MAE: %lf\n", duplicate_genome->get_mae(best_parameters_2, testing_inputs, testing_outputs)
        );
        duplicate_genome->write_predictions(
            output_directory, testing_filenames, best_parameters_2, testing_inputs, testing_outputs, time_series_sets
        );
    }

    Log::release_id("main");
    return 0;
}
