#include <signal.h>

#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "common/process_arguments.hxx"
#include "pi_data.hxx"
#include "pi_protocol.hxx"
#include "rnn/rnn_genome.hxx"
#include "time_series/time_series.hxx"
#include "weights/weight_update.hxx"

namespace {

using TimeSeries = std::vector<std::vector<std::vector<double>>>;

void train_requests(int fd, const pi_training::Config& config) {
    if (pi_training::fingerprint_files(pi_training::dataset_filenames(config.arguments)) != config.fingerprint) {
        throw std::runtime_error("training or validation datasets do not match the master");
    }

    auto sets = std::unique_ptr<TimeSeriesSets>(TimeSeriesSets::generate_from_arguments(config.arguments));
    TimeSeries training_inputs, training_outputs, validation_inputs, validation_outputs;
    get_train_validation_data(
        config.arguments, sets.get(), training_inputs, training_outputs, validation_inputs, validation_outputs
    );
    sets.reset();
    WeightUpdate update;
    update.generate_from_arguments(config.arguments);
    int32_t timeout_seconds = pi_training::kDefaultTrainingTimeoutSeconds;
    get_argument(config.arguments, "--pi_train_timeout_seconds", false, timeout_seconds);
    pi_training::set_timeout_seconds(fd, timeout_seconds);
    double weight_decay = 0.0;
    get_argument(config.arguments, "--weight_decay", false, weight_decay);
    pi_training::send_frame(fd, pi_training::Message::ready, "");

    std::uint32_t expected_request = 1;
    while (true) {
        const pi_training::Frame frame = pi_training::receive_frame(fd);
        if (frame.type == pi_training::Message::stop && frame.payload.empty()) {
            Log::info("received termination from the master\n");
            pi_training::send_frame(fd, pi_training::Message::ready, "");
            return;
        }
        if (frame.type != pi_training::Message::genome) {
            throw std::runtime_error("expected a training genome");
        }
        pi_training::Reader reader(frame.payload);
        if (reader.u64() != config.session) {
            throw std::runtime_error("unexpected training session");
        }
        const std::uint32_t request = reader.u32();
        if (request != expected_request++) {
            throw std::runtime_error("unexpected genome request number");
        }
        const std::string genome_bytes = reader.rest();
        if (genome_bytes.empty()) {
            throw std::runtime_error("received an empty genome");
        }
        std::istringstream input(genome_bytes, std::ios::in | std::ios::binary);
        auto genome = std::make_unique<RNN_Genome>(input);
        if (!input) {
            throw std::runtime_error("could not deserialize the training genome");
        }
        genome->set_weight_decay(weight_decay);
        genome->set_log_filename("");
        const int32_t id = genome->get_generation_id();
        Log::info("received genome %d\n", id);
        const std::string log_id = "genome_" + std::to_string(id) + "_pi";
        Log::set_id(log_id);
        genome->backpropagate_stochastic(
            training_inputs, training_outputs, validation_inputs, validation_outputs, &update
        );
        Log::release_id(log_id);
        Log::set_id("pi_worker");
        std::ostringstream output(std::ios::out | std::ios::binary);
        genome->write_to_stream(output);
        if (!output) {
            throw std::runtime_error("could not serialize the trained genome");
        }
        pi_training::send_frame(
            fd, pi_training::Message::result, pi_training::request_payload(config.session, request, output.str())
        );
        Log::info(
            "returned genome %d validation MSE %.17g parameter fingerprint %llu\n", id,
            genome->get_best_validation_mse(),
            static_cast<unsigned long long>(pi_training::fingerprint_parameters(genome->get_best_parameters()))
        );
    }
}

}  // namespace

int main(int argc, char** argv) {
    signal(SIGPIPE, SIG_IGN);
    try {
        const std::vector<std::string> arguments(argv, argv + argc);
        Log::initialize(arguments);
        Log::set_id("pi_worker");
        int32_t port = 0;
        get_argument(arguments, "--port", true, port);
        if (port < 1 || port > 65535) {
            throw std::runtime_error("--port must be between 1 and 65535");
        }
        const auto listener = pi_training::listen_local(port);
        Log::info("listening on 127.0.0.1:%d\n", port);
        std::cout << "Pi training worker listening on 127.0.0.1:" << port << std::endl;
        const auto connection = pi_training::accept_connection(listener.get());
        try {
            pi_training::set_timeout_seconds(connection.get(), pi_training::kStartupTimeoutSeconds);
            const pi_training::Frame frame = pi_training::receive_frame(connection.get());
            if (frame.type != pi_training::Message::config) {
                throw std::runtime_error("expected run configuration");
            }
            train_requests(connection.get(), pi_training::parse_config(frame.payload));
        } catch (const std::exception& error) {
            try {
                pi_training::send_frame(connection.get(), pi_training::Message::error, error.what());
            } catch (...) {
            }
            throw;
        }
        Log::release_id("pi_worker");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Pi training worker failed: " << error.what() << std::endl;
        return 1;
    }
}
