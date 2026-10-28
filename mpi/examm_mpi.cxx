#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>
using std::fixed;
using std::setprecision;
using std::setw;

#include <mutex>
using std::mutex;

#include <string>
using std::string;

#include <thread>
using std::thread;

#include <vector>
using std::vector;

#include "common/log.hxx"
#include "common/process_arguments.hxx"
#include "examm/examm.hxx"
#include "mpi.h"
#include "mpi/controller_client.hxx"
#include "mpi/pi_sender.hxx"
#include "rnn/generate_nn.hxx"
#include "time_series/time_series.hxx"
#include "weights/weight_rules.hxx"
#include "weights/weight_update.hxx"

#define WORK_REQUEST_TAG  1
#define GENOME_LENGTH_TAG 2
#define GENOME_TAG        3
#define TERMINATE_TAG     4

mutex examm_mutex;

vector<string> arguments;

EXAMM* examm;
WeightUpdate* weight_update_method;

// optional: only created when --send_to_pi is given. --pi_host/--pi_port
// override these defaults from the command line.
const string DEFAULT_PI_HOST = "192.168.0.70";
const int32_t DEFAULT_PI_PORT = 5555;
PiSender* pi_sender = NULL;
int32_t pi_last_sent_generation_id = -1;

// sends the global best genome to the pi if it changed since the last send
void send_best_genome_to_pi() {
    RNN_Genome* best = examm->get_best_genome();
    if (best == NULL || best->get_generation_id() == pi_last_sent_generation_id) {
        return;
    }
    pi_last_sent_generation_id = best->get_generation_id();

    char* byte_array;
    int32_t length;
    best->write_to_array(&byte_array, length);
    pi_sender->enqueue(byte_array, length);
    free(byte_array);
    Log::info("queued global best genome %d for the pi\n", pi_last_sent_generation_id);
}

// search settings should change and EXAMM applies the answers while it runs
ControllerClient* controller_client = NULL;
int32_t controller_every = 500;       // inserted genomes between progress summaries (--controller_every)
int32_t controller_last_sent_at = 0;  // evaluated-genome count when the last summary was sent
int32_t controller_window = 0;        // num of summaries sent so far
int32_t max_genomes_argument = 0;

string controller_settings_json(IslandSpeciationStrategy* s) {
    double mutation, intra, inter;
    s->get_rates(mutation, intra, inter);
    char buf[256];
    snprintf(
        buf, sizeof(buf),
        "{\"mutation_rate\":%.6f,\"intra_island_crossover_rate\":%.6f,\"inter_island_crossover_rate\":%.6f,"
        "\"num_mutations\":%d}",
        mutation, intra, inter, s->get_num_mutations()
    );
    return string(buf);
}

string controller_state_json(IslandSpeciationStrategy* s, int32_t window) {
    string islands = "[";
    for (int32_t i = 0; i < s->get_number_of_islands(); i++) {
        double best = s->get_island_best_fitness(i);
        char buf[128];
        if (best >= EXAMM_MAX_DOUBLE) {  // empty island has no best genome yet
            snprintf(
                buf, sizeof(buf), "%s{\"id\":%d,\"best_mse\":null,\"genomes\":%d}", i > 0 ? "," : "", i,
                s->get_island_size(i)
            );
        } else {
            snprintf(
                buf, sizeof(buf), "%s{\"id\":%d,\"best_mse\":%.6g,\"genomes\":%d}", i > 0 ? "," : "", i, best,
                s->get_island_size(i)
            );
        }
        islands += buf;
    }
    islands += "]";

    char head[256];
    snprintf(
        head, sizeof(head),
        "{\"type\":\"state\",\"t\":%d,\"inserted_genomes\":%d,\"max_genomes\":%d,\"global_best_mse\":%.6g,"
        "\"settings\":",
        window, s->get_evaluated_genomes(), max_genomes_argument, examm->get_best_fitness()
    );
    return string(head) + controller_settings_json(s) + ",\"islands\":" + islands + "}";
}

// call with examm_mutex held, right after a genome was inserted
void controller_after_insert() {
    IslandSpeciationStrategy* s = examm->get_island_strategy();
    if (s->get_evaluated_genomes() - controller_last_sent_at < controller_every) {
        return;
    }
    // if controller is still answering the last summary, returns false and we try again on the next insert
    if (controller_client->submit_state(controller_state_json(s, controller_window + 1))) {
        controller_window++;
        controller_last_sent_at = s->get_evaluated_genomes();
        Log::info("sent progress summary %d to the controller\n", controller_window);
    }
}

// call with examm_mutex held
void controller_apply_decisions() {
    string args, reason;
    while (controller_client->take_decision(args, reason)) {
        IslandSpeciationStrategy* s = examm->get_island_strategy();
        double mutation, intra, inter;
        s->get_rates(mutation, intra, inter);
        int32_t num_mutations = s->get_num_mutations();

        // args looks like "--mutation_rate 0.7 --num_mutations 2"; missing settings keep their val
        std::istringstream in(args);
        string name, value;
        while (in >> name >> value) {
            try {
                if (name == "--mutation_rate") {
                    mutation = std::stod(value);
                } else if (name == "--intra_island_crossover_rate") {
                    intra = std::stod(value);
                } else if (name == "--inter_island_crossover_rate") {
                    inter = std::stod(value);
                } else if (name == "--num_mutations") {
                    num_mutations = std::stoi(value);
                } else {
                    Log::warning("controller: ignoring unknown setting %s\n", name.c_str());
                }
            } catch (...) {
                Log::warning("controller: could not read '%s' for %s\n", value.c_str(), name.c_str());
            }
        }
        if (!std::isfinite(mutation) || !std::isfinite(intra) || !std::isfinite(inter)) {
            Log::warning("controller: ignoring decision with a non-finite rate: %s\n", args.c_str());
            continue;
        }
        num_mutations = std::min(std::max(num_mutations, 1), 20);  // the controller keeps this small

        s->set_rates(mutation, intra, inter);
        s->set_num_mutations(num_mutations);
        Log::info("applied controller decision: %s (reason: %s)\n", args.c_str(), reason.c_str());
    }
}

bool finished = false;

vector<vector<vector<double> > > training_inputs;
vector<vector<vector<double> > > training_outputs;
vector<vector<vector<double> > > validation_inputs;
vector<vector<vector<double> > > validation_outputs;

// bool random_sequence_length;
// int32_t sequence_length_lower_bound = 30;
// int32_t sequence_length_upper_bound = 100;

void send_work_request(int32_t target) {
    int32_t work_request_message[1];
    work_request_message[0] = 0;
    MPI_Send(work_request_message, 1, MPI_INT, target, WORK_REQUEST_TAG, MPI_COMM_WORLD);
}

void receive_work_request(int32_t source) {
    MPI_Status status;
    int32_t work_request_message[1];
    MPI_Recv(work_request_message, 1, MPI_INT, source, WORK_REQUEST_TAG, MPI_COMM_WORLD, &status);
}

RNN_Genome* receive_genome_from(int32_t source) {
    MPI_Status status;
    int32_t length_message[1];
    MPI_Recv(length_message, 1, MPI_INT, source, GENOME_LENGTH_TAG, MPI_COMM_WORLD, &status);

    int32_t length = length_message[0];

    Log::debug("receiving genome of length: %d from: %d\n", length, source);

    char* genome_str = new char[length + 1];

    Log::debug("receiving genome from: %d\n", source);
    MPI_Recv(genome_str, length, MPI_CHAR, source, GENOME_TAG, MPI_COMM_WORLD, &status);

    genome_str[length] = '\0';

    Log::trace("genome_str:\n%s\n", genome_str);

    RNN_Genome* genome = new RNN_Genome(genome_str, length);

    delete[] genome_str;
    return genome;
}

void send_genome_to(int32_t target, RNN_Genome* genome) {
    char* byte_array;
    int32_t length;

    genome->write_to_array(&byte_array, length);

    Log::debug("sending genome of length: %d to: %d\n", length, target);

    int32_t length_message[1];
    length_message[0] = length;
    MPI_Send(length_message, 1, MPI_INT, target, GENOME_LENGTH_TAG, MPI_COMM_WORLD);

    Log::debug("sending genome to: %d\n", target);
    MPI_Send(byte_array, length, MPI_CHAR, target, GENOME_TAG, MPI_COMM_WORLD);

    free(byte_array);
}

void send_terminate_message(int32_t target) {
    int32_t terminate_message[1];
    terminate_message[0] = 0;
    MPI_Send(terminate_message, 1, MPI_INT, target, TERMINATE_TAG, MPI_COMM_WORLD);
}

void receive_terminate_message(int32_t source) {
    MPI_Status status;
    int32_t terminate_message[1];
    MPI_Recv(terminate_message, 1, MPI_INT, source, TERMINATE_TAG, MPI_COMM_WORLD, &status);
}

void master(int32_t max_rank) {
    // the "main" id will have already been set by the main function so we do not need to re-set it here
    Log::debug("MAX int32_t: %d\n", numeric_limits<int32_t>::max());

    int32_t terminates_sent = 0;

    while (true) {
        // wait for a incoming message
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        int32_t source = status.MPI_SOURCE;
        int32_t tag = status.MPI_TAG;
        Log::debug("probe returned message from: %d with tag: %d\n", source, tag);

        if (controller_client != NULL) {
            examm_mutex.lock();
            controller_apply_decisions();
            examm_mutex.unlock();
        }

        // if the message is a work request, send a genome

        if (tag == WORK_REQUEST_TAG) {
            receive_work_request(source);

            // if (transfer_learning_version.compare("v3") == 0 || transfer_learning_version.compare("v1+v3") == 0) {
            //     seed_stirs = 3;
            // }
            examm_mutex.lock();
            RNN_Genome* genome = examm->generate_genome();
            examm_mutex.unlock();

            if (genome == NULL) {  // search was completed if it returns NULL for an individual
                // send terminate message
                Log::info("terminating worker: %d\n", source);
                send_terminate_message(source);
                terminates_sent++;

                Log::debug("sent: %d terminates of %d\n", terminates_sent, (max_rank - 1));
                if (terminates_sent >= max_rank - 1) {
                    return;
                }

            } else {
                // genome->write_to_file( examm->get_output_directory() + "/before_send_gen_" +
                // to_string(genome->get_generation_id()) );

                // send genome
                Log::debug("sending genome to: %d\n", source);
                send_genome_to(source, genome);

                // delete this genome as it will not be used again
                delete genome;
            }
        } else if (tag == GENOME_LENGTH_TAG) {
            Log::debug("received genome from: %d\n", source);
            RNN_Genome* genome = receive_genome_from(source);

            examm_mutex.lock();
            examm->insert_genome(genome);
            if (pi_sender != NULL) {
                send_best_genome_to_pi();
            }
            if (controller_client != NULL) {
                controller_after_insert();
            }
            examm_mutex.unlock();

            // delete the genome as it won't be used again, a copy was inserted
            delete genome;
            // this genome will be deleted if/when removed from population
        } else {
            Log::fatal("ERROR: received message from %d with unknown tag: %d", source, tag);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}

void worker(int32_t rank) {
    Log::set_id("worker_" + to_string(rank));

    while (true) {
        Log::debug("sending work request!\n");
        send_work_request(0);
        Log::debug("sent work request!\n");

        MPI_Status status;
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int32_t tag = status.MPI_TAG;

        Log::debug("probe received message with tag: %d\n", tag);

        if (tag == TERMINATE_TAG) {
            Log::debug("received terminate tag!\n");
            receive_terminate_message(0);
            break;

        } else if (tag == GENOME_LENGTH_TAG) {
            Log::debug("received genome!\n");
            RNN_Genome* genome = receive_genome_from(0);

            // have each worker write the backproagation to a separate log file
            string log_id = "genome_" + to_string(genome->get_generation_id()) + "_worker_" + to_string(rank);
            Log::set_id(log_id);
            genome->backpropagate_stochastic(
                training_inputs, training_outputs, validation_inputs, validation_outputs, weight_update_method
            );
            Log::release_id(log_id);

            // go back to the worker's log for MPI communication
            Log::set_id("worker_" + to_string(rank));

            send_genome_to(0, genome);

            delete genome;
        } else {
            Log::fatal("ERROR: received message with unknown tag: %d\n", tag);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // release the log file for the worker communication
    Log::release_id("worker_" + to_string(rank));
}

int main(int argc, char** argv) {
    std::cout << "starting up!" << std::endl;
    MPI_Init(&argc, &argv);
    std::cout << "did mpi init!" << std::endl;
    int32_t rank, max_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &max_rank);
    std::cout << "got rank " << rank << " and max rank " << max_rank << std::endl;
    arguments = vector<string>(argv, argv + argc);
    std::cout << "got arguments!" << std::endl;

    Log::initialize(arguments);
    Log::set_rank(rank);
    Log::set_id("main_" + to_string(rank));
    Log::restrict_to_rank(0);
    std::cout << "initailized log!" << std::endl;

    TimeSeriesSets* time_series_sets = NULL;
    time_series_sets = TimeSeriesSets::generate_from_arguments(arguments);
    get_train_validation_data(
        arguments, time_series_sets, training_inputs, training_outputs, validation_inputs, validation_outputs
    );

    weight_update_method = new WeightUpdate();
    weight_update_method->generate_from_arguments(arguments);

    WeightRules* weight_rules = new WeightRules();
    weight_rules->initialize_from_args(arguments);

    RNN_Genome* seed_genome = get_seed_genome(arguments, time_series_sets, weight_rules);

    Log::clear_rank_restriction();

    if (rank == 0) {
        write_time_series_to_file(arguments, time_series_sets);
        examm = generate_examm_from_arguments(arguments, time_series_sets, weight_rules, seed_genome);

        if (argument_exists(arguments, "--send_to_pi")) {
            string pi_host = DEFAULT_PI_HOST;
            int32_t pi_port = DEFAULT_PI_PORT;
            get_argument(arguments, "--pi_host", false, pi_host);
            get_argument(arguments, "--pi_port", false, pi_port);
            pi_sender = new PiSender(pi_host, pi_port);
        }

        if (argument_exists(arguments, "--controller_addr_file")) {
            if (examm->get_island_strategy() == NULL) {
                Log::fatal("--controller_addr_file only works with the island speciation strategy\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            string controller_addr_file;
            get_argument(arguments, "--controller_addr_file", true, controller_addr_file);
            get_argument(arguments, "--controller_every", false, controller_every);
            int32_t controller_timeout = 900;  // seconds to wait for one decision before giving up on it
            get_argument(arguments, "--controller_timeout", false, controller_timeout);
            get_argument(arguments, "--max_genomes", false, max_genomes_argument);

            // the run name is the last part of the output directory, with odd characters replaced
            string output_directory = "examm";
            get_argument(arguments, "--output_directory", false, output_directory);
            while (output_directory.size() > 1 && output_directory.back() == '/') {
                output_directory.pop_back();
            }
            string run_name = output_directory.substr(output_directory.find_last_of('/') + 1);
            for (size_t i = 0; i < run_name.size(); i++) {
                if (!isalnum((unsigned char) run_name[i]) && run_name[i] != '_' && run_name[i] != '-'
                    && run_name[i] != '.') {
                    run_name[i] = '_';
                }
            }

            controller_client = new ControllerClient(controller_addr_file, controller_timeout);
            controller_client->set_hello(
                "{\"type\":\"hello\",\"run\":\"" + run_name + "\",\"settings\":"
                + controller_settings_json(examm->get_island_strategy()) + "}"
            );
            Log::info(
                "controller enabled: address file %s, a summary every %d inserted genomes\n",
                controller_addr_file.c_str(), controller_every
            );
        }

        master(max_rank);

        if (controller_client != NULL) {
            controller_client->send_notice("{\"type\":\"done\",\"t\":" + to_string(controller_window) + "}");
            delete controller_client;  // sends the done message, does not wait for an unanswered summary
        }

        if (pi_sender != NULL) {
            send_best_genome_to_pi();  // make sure the final best is delivered
            delete pi_sender;          // waits for the queue to drain
        }
    } else {
        worker(rank);
    }
    Log::set_id("main_" + to_string(rank));
    finished = true;
    Log::debug("rank %d completed!\n");
    Log::release_id("main_" + to_string(rank));
    MPI_Finalize();

    delete time_series_sets;
    return 0;
}
