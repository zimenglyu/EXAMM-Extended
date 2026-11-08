#ifndef EXAMM_PI_TRAINING_BRIDGE_HXX
#define EXAMM_PI_TRAINING_BRIDGE_HXX

#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pi_data.hxx"
#include "pi_protocol.hxx"
#include "rnn/rnn_genome.hxx"

namespace pi_training {

class Bridge {
   public:
    Bridge(const std::string& host, int port, const std::vector<std::string>& arguments, int timeout_seconds)
        : connection_(connect_to(host, port)), session_(make_session()) {
        set_timeout_seconds(connection_.get(), kStartupTimeoutSeconds);
        const std::uint64_t fingerprint = fingerprint_files(dataset_filenames(arguments));
        send_frame(connection_.get(), Message::config, make_config(arguments, fingerprint, session_));
        expect_ready();
        set_timeout_seconds(connection_.get(), timeout_seconds);
    }

    std::unique_ptr<RNN_Genome> train(RNN_Genome& genome) {
        if (request_ == std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("too many genome requests in one session");
        }
        const std::uint32_t current = ++request_;
        std::ostringstream output(std::ios::out | std::ios::binary);
        genome.write_to_stream(output);
        if (!output) {
            throw std::runtime_error("could not serialize the training genome");
        }
        send_frame(connection_.get(), Message::genome, request_payload(session_, current, output.str()));
        const Frame reply = receive_frame(connection_.get());
        check_error(reply);
        if (reply.type != Message::result) {
            throw std::runtime_error("expected a trained genome");
        }
        Reader reader(reply.payload);
        if (reader.u64() != session_ || reader.u32() != current) {
            throw std::runtime_error("Pi returned a result for a different training request");
        }
        const std::string bytes = reader.rest();
        if (bytes.empty()) {
            throw std::runtime_error("Pi returned an empty genome");
        }
        std::istringstream input(bytes, std::ios::in | std::ios::binary);
        auto result = std::make_unique<RNN_Genome>(input);
        if (!input || result->get_generation_id() != genome.get_generation_id()) {
            throw std::runtime_error("Pi returned the wrong or an incomplete genome");
        }
        result->set_weight_decay(genome.get_weight_decay());
        return result;
    }

    void stop() {
        send_frame(connection_.get(), Message::stop, "");
        expect_ready();
    }

   private:
    static std::uint64_t make_session() {
        std::random_device random;
        return (static_cast<std::uint64_t>(random()) << 32) ^ random();
    }

    static void check_error(const Frame& frame) {
        if (frame.type == Message::error) {
            throw std::runtime_error("Pi training worker: " + frame.payload);
        }
    }

    void expect_ready() {
        const Frame reply = receive_frame(connection_.get());
        check_error(reply);
        if (reply.type != Message::ready || !reply.payload.empty()) {
            throw std::runtime_error("unexpected reply from Pi training worker");
        }
    }

    Socket connection_;
    std::uint64_t session_;
    std::uint32_t request_ = 0;
};

}  // namespace pi_training

#endif
