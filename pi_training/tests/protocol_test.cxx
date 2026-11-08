#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "pi_protocol.hxx"

int main() {
    const std::vector<std::string> arguments = {"examm", "--weight_update", "adagrad"};
    const std::string config = pi_training::make_config(arguments, 123, 456);
    const auto parsed = pi_training::parse_config(config);
    assert(parsed.arguments == arguments);
    assert(parsed.fingerprint == 123);
    assert(parsed.session == 456);
    std::string wrong_format = config;
    wrong_format.at(4) ^= 1;
    bool rejected = false;
    try {
        (void) pi_training::parse_config(wrong_format);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);

    int sockets[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);

    const std::string payload("a\0b", 3);
    pi_training::send_frame(sockets[0], pi_training::Message::genome, payload);
    const auto received = pi_training::receive_frame(sockets[1]);
    assert(received.type == pi_training::Message::genome);
    assert(received.payload == payload);
    close(sockets[0]);
    close(sockets[1]);

    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    const std::uint32_t header[] = {
        htonl(static_cast<std::uint32_t>(pi_training::Message::genome)),
        htonl(pi_training::kMaxFrameBytes + 1),
    };
    assert(write(sockets[0], header, sizeof(header)) == sizeof(header));
    rejected = false;
    try {
        (void) pi_training::receive_frame(sockets[1]);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);
    close(sockets[0]);
    close(sockets[1]);

    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    const std::uint32_t truncated[] = {
        htonl(static_cast<std::uint32_t>(pi_training::Message::genome)),
        htonl(3),
    };
    assert(write(sockets[0], truncated, sizeof(truncated)) == sizeof(truncated));
    assert(write(sockets[0], "x", 1) == 1);
    close(sockets[0]);
    rejected = false;
    try {
        (void) pi_training::receive_frame(sockets[1]);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);
    close(sockets[1]);
}
