#ifndef EXAMM_PI_TRAINING_PROTOCOL_HXX
#define EXAMM_PI_TRAINING_PROTOCOL_HXX

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef EXAMM_PI_SOURCE_FINGERPRINT
#error EXAMM_PI_SOURCE_FINGERPRINT must be set by the project build
#endif

namespace pi_training {

constexpr std::uint32_t kMaxFrameBytes = 64U * 1024U * 1024U;
constexpr std::uint32_t kMaxArguments = 4096;
constexpr int kStartupTimeoutSeconds = 600;
constexpr int kDefaultTrainingTimeoutSeconds = 7200;

enum class Message : std::uint32_t { config = 1, ready = 2, genome = 3, result = 4, stop = 5, error = 6 };

struct Frame {
    Message type;
    std::string payload;
};

class Socket {
   public:
    explicit Socket(int fd = -1) : fd_(fd) {
    }
    ~Socket() {
        if (fd_ >= 0) {
            close(fd_);
        }
    }
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;
    Socket(Socket&& other) noexcept : fd_(other.fd_) {
        other.fd_ = -1;
    }
    int get() const {
        return fd_;
    }

   private:
    int fd_;
};

inline void send_all(int fd, const char* bytes, std::size_t size) {
    while (size != 0) {
        const ssize_t count = send(fd, bytes, size, 0);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            throw std::runtime_error("connection timed out while sending data");
        }
        if (count <= 0) {
            throw std::runtime_error("connection failed while sending data");
        }
        bytes += count;
        size -= static_cast<std::size_t>(count);
    }
}

inline void receive_all(int fd, char* bytes, std::size_t size) {
    while (size != 0) {
        const ssize_t count = recv(fd, bytes, size, 0);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            throw std::runtime_error("connection timed out while waiting for data");
        }
        if (count <= 0) {
            throw std::runtime_error("connection closed before the complete message arrived");
        }
        bytes += count;
        size -= static_cast<std::size_t>(count);
    }
}

inline void send_frame(int fd, Message type, const std::string& payload) {
    if (payload.size() > kMaxFrameBytes) {
        throw std::runtime_error("message exceeds the maximum size");
    }
    const std::uint32_t header[] = {
        htonl(static_cast<std::uint32_t>(type)), htonl(static_cast<std::uint32_t>(payload.size()))
    };
    send_all(fd, reinterpret_cast<const char*>(header), sizeof(header));
    send_all(fd, payload.data(), payload.size());
}

inline Frame receive_frame(int fd) {
    std::uint32_t header[2];
    receive_all(fd, reinterpret_cast<char*>(header), sizeof(header));
    const std::uint32_t type = ntohl(header[0]);
    const std::uint32_t size = ntohl(header[1]);
    if (type < static_cast<std::uint32_t>(Message::config) || type > static_cast<std::uint32_t>(Message::error)) {
        throw std::runtime_error("unknown message type");
    }
    if (size > kMaxFrameBytes) {
        throw std::runtime_error("message exceeds the maximum size");
    }
    std::string payload(size, '\0');
    if (size != 0) {
        receive_all(fd, &payload[0], size);
    }
    return {static_cast<Message>(type), std::move(payload)};
}

inline void set_timeout_seconds(int fd, int seconds) {
    if (seconds <= 0) {
        throw std::runtime_error("training timeout must be positive");
    }
    timeval timeout = {};
    timeout.tv_sec = seconds;
    if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) != 0
        || setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) != 0) {
        throw std::runtime_error("could not set connection timeout");
    }
}

inline Socket connect_to(const std::string& host, int port) {
    addrinfo hints = {};
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* addresses = nullptr;
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &addresses) != 0) {
        throw std::runtime_error("could not resolve Pi training host");
    }
    int connected = -1;
    for (addrinfo* address = addresses; address != nullptr; address = address->ai_next) {
        int fd = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
        if (fd < 0) {
            continue;
        }
        const int flags = fcntl(fd, F_GETFL, 0);
        if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0) {
            close(fd);
            continue;
        }
        int status = connect(fd, address->ai_addr, address->ai_addrlen);
        if (status != 0 && errno == EINPROGRESS) {
            pollfd wait = {fd, POLLOUT, 0};
            do {
                status = poll(&wait, 1, 30000);
            } while (status < 0 && errno == EINTR);
            int socket_error = 0;
            socklen_t error_size = sizeof(socket_error);
            if (status > 0 && getsockopt(fd, SOL_SOCKET, SO_ERROR, &socket_error, &error_size) == 0
                && socket_error == 0) {
                status = 0;
            } else {
                status = -1;
            }
        }
        if (status == 0 && fcntl(fd, F_SETFL, flags) == 0) {
            connected = fd;
            break;
        }
        close(fd);
    }
    freeaddrinfo(addresses);
    if (connected < 0) {
        throw std::runtime_error("could not connect to the Pi training worker");
    }
    const int enabled = 1;
    setsockopt(connected, SOL_SOCKET, SO_KEEPALIVE, &enabled, sizeof(enabled));
    return Socket(connected);
}

inline Socket listen_local(int port) {
    Socket listener(socket(AF_INET, SOCK_STREAM, 0));
    if (listener.get() < 0) {
        throw std::runtime_error("could not create listening socket");
    }
    const int enabled = 1;
    if (setsockopt(listener.get(), SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled)) != 0) {
        throw std::runtime_error("could not configure listening socket");
    }
    sockaddr_in address = {};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(static_cast<std::uint16_t>(port));
    if (::bind(listener.get(), reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0
        || ::listen(listener.get(), 1) != 0) {
        throw std::runtime_error("could not listen on the requested local port");
    }
    return listener;
}

inline Socket accept_connection(int listener) {
    int fd;
    do {
        fd = accept(listener, nullptr, nullptr);
    } while (fd < 0 && errno == EINTR);
    if (fd < 0) {
        throw std::runtime_error("could not accept the training connection");
    }
    const int enabled = 1;
    setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &enabled, sizeof(enabled));
    return Socket(fd);
}

inline void append_u32(std::string& out, std::uint32_t value) {
    value = htonl(value);
    out.append(reinterpret_cast<const char*>(&value), sizeof(value));
}

inline void append_u64(std::string& out, std::uint64_t value) {
    append_u32(out, static_cast<std::uint32_t>(value >> 32));
    append_u32(out, static_cast<std::uint32_t>(value));
}

inline void append_string(std::string& out, const std::string& value) {
    if (value.size() > kMaxFrameBytes) {
        throw std::runtime_error("configuration value is too long");
    }
    append_u32(out, static_cast<std::uint32_t>(value.size()));
    out.append(value);
}

class Reader {
   public:
    explicit Reader(const std::string& data) : data_(data) {
    }
    std::uint32_t u32() {
        if (data_.size() - offset_ < sizeof(std::uint32_t)) {
            throw std::runtime_error("incomplete message");
        }
        std::uint32_t value;
        std::memcpy(&value, data_.data() + offset_, sizeof(value));
        offset_ += sizeof(value);
        return ntohl(value);
    }
    std::uint64_t u64() {
        const std::uint64_t high = u32();
        return (high << 32) | u32();
    }
    std::string string() {
        const std::uint32_t length = u32();
        if (length > data_.size() - offset_) {
            throw std::runtime_error("incomplete configuration value");
        }
        std::string result = data_.substr(offset_, length);
        offset_ += length;
        return result;
    }
    std::string rest() {
        std::string result = data_.substr(offset_);
        offset_ = data_.size();
        return result;
    }
    bool done() const {
        return offset_ == data_.size();
    }

   private:
    const std::string& data_;
    std::size_t offset_ = 0;
};

inline std::string native_signature() {
    if (sizeof(std::int32_t) != 4 || sizeof(double) != 8 || !std::numeric_limits<double>::is_iec559) {
        throw std::runtime_error("this machine has an unsupported genome format");
    }
    const std::int32_t integer = 0x01020304;
    const double number = 1.25;
    const bool no = false, yes = true;
    std::string signature = "EXAMM-PI-TRAIN-1";
    signature += EXAMM_PI_SOURCE_FINGERPRINT;
    signature.push_back(static_cast<char>(sizeof(bool)));
    signature.append(reinterpret_cast<const char*>(&integer), sizeof(integer));
    signature.append(reinterpret_cast<const char*>(&number), sizeof(number));
    signature.append(reinterpret_cast<const char*>(&no), sizeof(no));
    signature.append(reinterpret_cast<const char*>(&yes), sizeof(yes));
    return signature;
}

inline std::uint64_t fingerprint_files(const std::vector<std::string>& filenames) {
    if (filenames.empty()) {
        throw std::runtime_error("no training or validation files were specified");
    }
    std::uint64_t hash = UINT64_C(14695981039346656037);
    const auto update = [&hash](unsigned char byte) { hash = (hash ^ byte) * UINT64_C(1099511628211); };
    for (const std::string& name : filenames) {
        for (unsigned char byte : name) {
            update(byte);
        }
        update(0);
        std::ifstream input(name, std::ios::binary);
        if (!input) {
            throw std::runtime_error("cannot read dataset file: " + name);
        }
        char buffer[16384];
        std::uint64_t size = 0;
        while (input.read(buffer, sizeof(buffer)) || input.gcount() > 0) {
            for (std::streamsize i = 0; i < input.gcount(); ++i) {
                update(static_cast<unsigned char>(buffer[i]));
            }
            size += static_cast<std::uint64_t>(input.gcount());
        }
        if (input.bad()) {
            throw std::runtime_error("error reading dataset file: " + name);
        }
        for (int shift = 0; shift < 64; shift += 8) {
            update(static_cast<unsigned char>(size >> shift));
        }
    }
    return hash;
}

inline std::uint64_t fingerprint_parameters(const std::vector<double>& parameters) {
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (double parameter : parameters) {
        unsigned char bytes[sizeof(parameter)];
        std::memcpy(bytes, &parameter, sizeof(parameter));
        for (unsigned char byte : bytes) {
            hash = (hash ^ byte) * UINT64_C(1099511628211);
        }
    }
    return hash;
}

inline std::string make_config(
    const std::vector<std::string>& arguments, std::uint64_t fingerprint, std::uint64_t session
) {
    if (arguments.size() > kMaxArguments) {
        throw std::runtime_error("too many configuration arguments");
    }
    std::string result;
    append_string(result, native_signature());
    append_u64(result, fingerprint);
    append_u64(result, session);
    append_u32(result, static_cast<std::uint32_t>(arguments.size()));
    for (const auto& argument : arguments) {
        append_string(result, argument);
    }
    return result;
}

struct Config {
    std::uint64_t fingerprint;
    std::uint64_t session;
    std::vector<std::string> arguments;
};

inline Config parse_config(const std::string& payload) {
    Reader reader(payload);
    if (reader.string() != native_signature()) {
        throw std::runtime_error("incompatible genome formats");
    }
    Config config;
    config.fingerprint = reader.u64();
    config.session = reader.u64();
    const std::uint32_t count = reader.u32();
    if (count > kMaxArguments) {
        throw std::runtime_error("too many configuration arguments");
    }
    for (std::uint32_t i = 0; i < count; ++i) {
        config.arguments.push_back(reader.string());
    }
    if (!reader.done()) {
        throw std::runtime_error("unexpected data after configuration");
    }
    return config;
}

inline std::string request_payload(std::uint64_t session, std::uint32_t request, const std::string& genome) {
    std::string result;
    append_u64(result, session);
    append_u32(result, request);
    result.append(genome);
    return result;
}

}  // namespace pi_training

#endif
