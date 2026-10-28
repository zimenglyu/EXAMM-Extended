#ifndef EXAMM_CONTROLLER_CLIENT_HXX
#define EXAMM_CONTROLLER_CLIENT_HXX

// All network work happens on a background thread MPI does wait for controller 
// At most one state message is in flight
// if the master submits another while controller still thinking. it's dropped.

#include <netdb.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>

#include "common/log.hxx"

// Returns string val of "key" in a one-line JSON object or "" if it is missing.
// Handles the escapes json.dumps, anything else becomes '?'.
inline std::string json_string_field(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) {
        return "";
    }
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) {
        return "";
    }
    pos = json.find('"', pos);
    if (pos == std::string::npos) {
        return "";
    }
    std::string out;
    for (size_t i = pos + 1; i < json.size(); i++) {
        char c = json[i];
        if (c == '"') {
            return out;
        }
        if (c == '\\' && i + 1 < json.size()) {
            char n = json[++i];
            switch (n) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                default: out += '?'; break;
            }
        } else {
            out += c;
        }
    }
    return "";  // string never closed
}

class ControllerClient {
   private:
    struct Message {
        std::string line;
        bool expects_reply;
    };

    std::string addr_file;
    int reply_timeout_s;
    int sockfd = -1;
    std::string read_buffer;

    std::mutex m;
    std::condition_variable cv;
    bool stopping = false;
    bool state_busy = false;  // state message is queued or waiting for reply
    std::deque<Message> queue;
    std::string hello_line;  // sent first on every (re)connection so the controller knows the run
    bool decision_ready = false;
    std::string decision_args;
    std::string decision_reason;
    std::thread worker;

    bool is_stopping() {
        std::lock_guard<std::mutex> lock(m);
        return stopping;
    }

    // the controller writes "host:port" into addr_file when starts
    bool read_addr_file(std::string& host, std::string& port) {
        std::ifstream in(addr_file);
        std::string line;
        if (!in || !std::getline(in, line)) {
            return false;
        }
        size_t colon = line.rfind(':');
        if (colon == std::string::npos || colon == 0 || colon + 1 >= line.size()) {
            return false;
        }
        host = line.substr(0, colon);
        port = line.substr(colon + 1);
        while (!port.empty() && (port.back() == '\r' || port.back() == ' ')) {
            port.pop_back();
        }
        return true;
    }

    bool connect_to_controller() {
        std::string host, port;
        if (!read_addr_file(host, port)) {
            Log::warning("controller client: no usable address in %s yet\n", addr_file.c_str());
            return false;
        }
        struct addrinfo hints = {}, *res = NULL;
        hints.ai_socktype = SOCK_STREAM;
        if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0) {
            Log::warning("controller client: cannot resolve %s\n", host.c_str());
            return false;
        }
        for (struct addrinfo* r = res; r != NULL && sockfd < 0; r = r->ai_next) {
            int fd = socket(r->ai_family, r->ai_socktype, r->ai_protocol);
            if (fd >= 0 && connect(fd, r->ai_addr, r->ai_addrlen) == 0) {
                sockfd = fd;
            } else if (fd >= 0) {
                close(fd);
            }
        }
        freeaddrinfo(res);
        if (sockfd < 0) {
            Log::warning("controller client: could not connect to %s:%s\n", host.c_str(), port.c_str());
            return false;
        }
        read_buffer.clear();
        Log::info("controller client: connected to %s:%s\n", host.c_str(), port.c_str());
        std::string hello;
        {
            std::lock_guard<std::mutex> lock(m);
            hello = hello_line;
        }
        if (!hello.empty() && !send_line(hello)) {  // new con starts with the hello
            return false;
        }
        return true;
    }

    void drop_connection() {
        if (sockfd >= 0) {
            close(sockfd);
            sockfd = -1;
        }
        read_buffer.clear();
    }

    bool send_line(const std::string& line) {
        std::string msg = line + "\n";
        size_t total = 0;
        while (total < msg.size()) {
            ssize_t n = send(sockfd, msg.data() + total, msg.size() - total, MSG_NOSIGNAL);
            if (n <= 0) {
                drop_connection();
                return false;
            }
            total += n;
        }
        return true;
    }

    // waits for one full line from the controller. gives up on timeout
    bool read_line(std::string& line) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(reply_timeout_s);
        while (true) {
            size_t nl = read_buffer.find('\n');
            if (nl != std::string::npos) {
                line = read_buffer.substr(0, nl);
                read_buffer.erase(0, nl + 1);
                return true;
            }
            if (is_stopping() || std::chrono::steady_clock::now() > deadline) {
                return false;
            }
            struct pollfd p = {sockfd, POLLIN, 0};
            int ready = poll(&p, 1, 200);  // wake up often so shutdown no waits long
            if (ready < 0) {
                drop_connection();
                return false;
            }
            if (ready > 0) {
                char buf[4096];
                ssize_t n = recv(sockfd, buf, sizeof(buf), 0);
                if (n <= 0) {
                    drop_connection();
                    return false;
                }
                read_buffer.append(buf, n);
            }
        }
    }

    void run() {
        signal(SIGPIPE, SIG_IGN);  // dropped controller con must not kill the master
        Log::set_id("controller_client");
        while (true) {
            Message msg;
            {
                std::unique_lock<std::mutex> lock(m);
                cv.wait(lock, [this] { return stopping || !queue.empty(); });
                if (queue.empty()) {  // stopping and everything has been sent
                    break;
                }
                msg = queue.front();
                queue.pop_front();
            }

            bool sent = (sockfd >= 0 || connect_to_controller()) && send_line(msg.line);
            std::string reply;
            bool got_reply = false;
            if (sent && msg.expects_reply) {
                got_reply = read_line(reply);
                if (!got_reply && sockfd >= 0 && !is_stopping()) {
                    Log::warning("controller client: no decision arrived in %d s, dropping connection\n", reply_timeout_s);
                    drop_connection();  // late reply would answer the wrong window
                }
            }

            std::lock_guard<std::mutex> lock(m);
            if (msg.expects_reply) {
                state_busy = false;
                if (got_reply && json_string_field(reply, "type") == "decision") {
                    decision_args = json_string_field(reply, "args");
                    decision_reason = json_string_field(reply, "reason");
                    decision_ready = true;
                } else if (!sent) {
                    Log::warning("controller client: state message was not delivered\n");
                }
            }
        }
        drop_connection();
        Log::release_id("controller_client");
    }

   public:
    ControllerClient(const std::string& _addr_file, int _reply_timeout_s)
        : addr_file(_addr_file), reply_timeout_s(_reply_timeout_s) {
        worker = std::thread(&ControllerClient::run, this);
    }

    ~ControllerClient() {
        {
            std::lock_guard<std::mutex> lock(m);
            stopping = true;
        }
        cv.notify_all();
        worker.join();
    }

    void set_hello(const std::string& json_line) {
        std::lock_guard<std::mutex> lock(m);
        hello_line = json_line;
    }

    // messages need no reply
    void send_notice(const std::string& json_line) {
        {
            std::lock_guard<std::mutex> lock(m);
            queue.push_back({json_line, false});
        }
        cv.notify_one();
    }

    // returns false if the previous state is still being answered
    bool submit_state(const std::string& json_line) {
        {
            std::lock_guard<std::mutex> lock(m);
            if (state_busy) {
                return false;
            }
            state_busy = true;
            queue.push_back({json_line, true});
        }
        cv.notify_one();
        return true;
    }

    // true once per decision
    bool take_decision(std::string& args, std::string& reason) {
        std::lock_guard<std::mutex> lock(m);
        if (!decision_ready) {
            return false;
        }
        args = decision_args;
        reason = decision_reason;
        decision_ready = false;
        return true;
    }
};

#endif
