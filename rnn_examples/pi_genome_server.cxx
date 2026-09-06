// Runs on the Raspberry Pi. Listens on a TCP port for genomes streamed by
// examm_mpi (--send_to_pi) and evaluates each one on the local test data the
// same way evaluate_rnn does: MSE/MAE, inference time, throughput, model
// latency and (optionally, --ina219) INA219 power/energy during inference.
//
// Wire format (same as the MPI messages): int32_t length, then that many bytes
// of RNN_Genome::write_to_array() output.

#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
using std::ofstream;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/arguments.hxx"
#include "common/ina219.hxx"
#include "common/log.hxx"
#include "rnn/rnn_genome.hxx"
#include "time_series/time_series.hxx"

vector<string> arguments;

vector<vector<vector<double> > > testing_inputs;
vector<vector<vector<double> > > testing_outputs;

// reads exactly length bytes, returns false when the connection is closed
bool read_all(int fd, char* buffer, int32_t length) {
    int32_t total = 0;
    while (total < length) {
        ssize_t n = recv(fd, buffer + total, length - total, 0);
        if (n <= 0) {
            return false;
        }
        total += n;
    }
    return true;
}

double elapsed_ms(std::chrono::high_resolution_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count();
}

int main(int argc, char** argv) {
    arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_id("main");

    int32_t port;
    get_argument(arguments, "--port", true, port);

    string output_directory;
    get_argument(arguments, "--output_directory", true, output_directory);

    vector<string> testing_filenames;
    get_argument_vector(arguments, "--testing_filenames", true, testing_filenames);

    int32_t time_offset = 1;
    get_argument(arguments, "--time_offset", true, time_offset);

    bool save_genomes = argument_exists(arguments, "--save_genomes");

    bool use_ina219 = argument_exists(arguments, "--ina219");
    string ina219_device = "/dev/i2c-1";
    get_argument(arguments, "--ina219_device", false, ina219_device);

    INA219 ina219;
    INA219Sampler ina219_sampler;
    bool ina219_active = false;
    if (use_ina219) {
        if (ina219.open_device(ina219_device.c_str()) && ina219.configure()) {
            ina219_active = true;
            Log::info("INA219 power monitor enabled on %s\n", ina219_device.c_str());
        } else {
            Log::warning("INA219 requested but could not open %s, continuing without power monitoring\n", ina219_device.c_str());
        }
    }

    ofstream results(output_directory + "/pi_evaluations.csv", std::ios::app);
    results << "generation_id,parameters,rows,mse,mae,inference_ms,per_point_us,throughput_per_s,model_latency_ms,"
            << "ina219_samples,bus_voltage_v_avg,current_ma_avg,power_mw_avg,energy_mj,energy_per_point_mj" << std::endl;

    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    if (::bind(listen_fd, (struct sockaddr*) &addr, sizeof(addr)) < 0 || listen(listen_fd, 1) < 0) {
        Log::fatal("could not listen on port %d\n", port);
        return 1;
    }

    // test data is loaded (and normalized) once, using the first genome received
    TimeSeriesSets* time_series_sets = NULL;
    int32_t total_rows = 0;

    while (true) {
        Log::info("waiting for a connection on port %d\n", port);
        int fd = accept(listen_fd, NULL, NULL);
        if (fd < 0) {
            continue;
        }
        Log::info("connected\n");

        int32_t length;
        while (read_all(fd, (char*) &length, sizeof(int32_t))) {
            vector<char> bytes(length);
            if (!read_all(fd, bytes.data(), length)) {
                break;
            }
            RNN_Genome* genome = new RNN_Genome(bytes.data(), length);
            int32_t generation_id = genome->get_generation_id();
            Log::info("received genome %d (%d bytes)\n", generation_id, length);

            if (time_series_sets == NULL) {
                time_series_sets = TimeSeriesSets::generate_test(
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
                time_series_sets->export_test_series(time_offset, testing_inputs, testing_outputs);
                for (int32_t i = 0; i < (int32_t) testing_inputs.size(); i++) {
                    total_rows += testing_inputs[i][0].size();
                }
                Log::info("loaded %d rows for testing, normalize type: %s\n", total_rows, normalize_type.c_str());
            }

            // predictions for each genome go in their own directory
            string genome_directory = output_directory + "/genome_" + std::to_string(generation_id);
            mkdir(genome_directory.c_str(), 0755);
            if (save_genomes) {
                genome->write_to_file(genome_directory + "/genome_" + std::to_string(generation_id) + ".bin");
            }

            vector<double> best_parameters = genome->get_best_parameters();
            Log::info("parameter count: %zu\n", best_parameters.size());

            // model latency: end-to-end input ready -> predictions written
            // inference time: model computation only (mse + mae passes)
            auto model_latency_start = std::chrono::high_resolution_clock::now();
            auto inference_start = model_latency_start;
            if (ina219_active) {
                ina219_sampler.start(&ina219);
            }
            double mse = genome->get_mse(best_parameters, testing_inputs, testing_outputs);
            double mae = genome->get_mae(best_parameters, testing_inputs, testing_outputs);
            double inference_ms = elapsed_ms(inference_start);
            if (ina219_active) {
                ina219_sampler.stop();
            }
            genome->write_predictions(
                genome_directory, testing_filenames, best_parameters, testing_inputs, testing_outputs, time_series_sets
            );
            double model_latency_ms = elapsed_ms(model_latency_start);

            double per_point_us = inference_ms * 1000.0 / total_rows;
            double throughput = total_rows / (inference_ms / 1000.0);

            Log::info("genome %d: MSE %lf, MAE %lf\n", generation_id, mse, mae);
            Log::info("  inference time: %.1f ms, per data point: %.1f us, throughput: %.1f points/s\n", inference_ms, per_point_us, throughput);
            Log::info("  model latency: %.1f ms, per data point: %.3f ms\n", model_latency_ms, model_latency_ms / total_rows);

            INA219Stats power = {};
            if (ina219_active) {
                power = ina219_sampler.get_stats();
                Log::info("  INA219 (%d samples): bus %.3f V, current avg %.1f mA (min %.1f, max %.1f), power avg %.1f mW (min %.1f, max %.1f)\n",
                    power.sample_count, power.bus_voltage_v_avg, power.current_ma_avg, power.current_ma_min, power.current_ma_max,
                    power.power_mw_avg, power.power_mw_min, power.power_mw_max);
                Log::info("  energy: %.3f mJ, per data point: %.6f mJ\n", power.energy_mj, power.energy_mj / total_rows);
            }

            results << generation_id << "," << best_parameters.size() << "," << total_rows << "," << mse << "," << mae << ","
                    << inference_ms << "," << per_point_us << "," << throughput << "," << model_latency_ms << ","
                    << power.sample_count << "," << power.bus_voltage_v_avg << "," << power.current_ma_avg << ","
                    << power.power_mw_avg << "," << power.energy_mj << "," << power.energy_mj / total_rows << std::endl;

            delete genome;
        }

        Log::info("connection closed\n");
        close(fd);
    }

    if (ina219_active) {
        ina219.close_device();
    }
}
