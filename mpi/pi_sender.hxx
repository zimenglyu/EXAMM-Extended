#ifndef EXAMM_PI_SENDER_HXX
#define EXAMM_PI_SENDER_HXX

// Optional: streams serialized genomes to a remote TCP listener (pi_genome_server)
// on a background thread so the MPI master is never blocked by the network.
// Wire format matches the MPI genome messages: int32_t length, then the bytes.

#include <netdb.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include "common/log.hxx"

class PiSender {
   private:
    string host;
    string port;
    int sockfd = -1;
    bool stopping = false;
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::vector<char> > queue;
    std::thread worker;

    bool connect_to_pi() {
        struct addrinfo hints = {}, *res = NULL;
        hints.ai_socktype = SOCK_STREAM;
        if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0) {
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
            Log::warning("pi sender could not connect to %s:%s, will retry\n", host.c_str(), port.c_str());
        } else {
            Log::info("pi sender connected to %s:%s\n", host.c_str(), port.c_str());
        }
        return sockfd >= 0;
    }

    bool send_all(const std::vector<char>& msg) {
        size_t total = 0;
        while (total < msg.size()) {
            ssize_t n = send(sockfd, msg.data() + total, msg.size() - total, 0);
            if (n <= 0) {
                close(sockfd);
                sockfd = -1;
                return false;
            }
            total += n;
        }
        return true;
    }

    void run() {
        signal(SIGPIPE, SIG_IGN);  // a dropped pi connection must not kill the master
        Log::set_id("pi_sender");
        while (true) {
            std::vector<char> msg;
            {
                std::unique_lock<std::mutex> lock(m);
                cv.wait(lock, [this] { return stopping || !queue.empty(); });
                if (queue.empty()) {
                    Log::release_id("pi_sender");
                    return;
                }
                msg = queue.front();
            }
            if ((sockfd >= 0 || connect_to_pi()) && send_all(msg)) {
                std::lock_guard<std::mutex> lock(m);
                queue.pop_front();
            } else if (stopping) {
                Log::release_id("pi_sender");
                return;
            } else {
                std::this_thread::sleep_for(std::chrono::seconds(5));
            }
        }
    }

   public:
    PiSender(string _host, int32_t _port) : host(_host), port(std::to_string(_port)) {
        worker = std::thread(&PiSender::run, this);
    }

    ~PiSender() {
        {
            std::lock_guard<std::mutex> lock(m);
            stopping = true;
        }
        cv.notify_all();
        worker.join();
        if (sockfd >= 0) {
            close(sockfd);
        }
    }

    // copies the bytes and returns immediately
    void enqueue(const char* bytes, int32_t length) {
        std::vector<char> msg(sizeof(int32_t) + length);
        memcpy(msg.data(), &length, sizeof(int32_t));
        memcpy(msg.data() + sizeof(int32_t), bytes, length);
        {
            std::lock_guard<std::mutex> lock(m);
            queue.push_back(std::move(msg));
        }
        cv.notify_one();
    }
};

#endif
