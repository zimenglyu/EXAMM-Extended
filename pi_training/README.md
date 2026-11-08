# Training EXAMM candidates on a Raspberry Pi

The Mac runs the EXAMM search with two MPI ranks: a master and a relay. The
relay sends each candidate to a single Pi worker through an SSH tunnel. The Pi
calls the existing `RNN_Genome::backpropagate_stochastic()` function, then sends
the trained genome back. The master uses the returned genome in its population.
Without `--pi_train_host`, the MPI program retains its local training path.

This mode trains one candidate at a time on one Pi. It requires exactly two MPI
ranks and cannot be combined with the separate `--send_to_pi` inference mode.

## Build

Use the same source revision on both machines and the dependencies described in
the repository README. A C++20 compiler and CMake 3.12 or newer are required by
this component. The Mac also needs MPI. Run commands from the repository root.

On the Mac:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target examm_mpi pi_train_server --parallel 2
```

On the Pi:

```sh
cmake -S . -B build-pi -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_DISABLE_FIND_PACKAGE_MPI=TRUE
cmake --build build-pi --target pi_train_server --parallel 1
```

The configuration handshake checks source fingerprints, native genome-format
properties, and dataset fingerprints. Both machines need identical copies of
the relevant source and datasets. Source fingerprints include file contents,
so even comment changes in a hashed source file require matching rebuilds.
The check does not make the native genome serialization universally portable.

Dataset paths supplied by the Mac are also used on the Pi. Use paths relative to
the repository root and start both programs there. Matching data in different
home directories is supported this way; different absolute paths are not remapped.

## Small Mac–Pi run

First, start the Pi worker on the Pi, from its repository root:

```sh
./build-pi/pi_training/pi_train_server \
  --port 24680 \
  --output_directory "$HOME/examm-runs/smoke-pi-01" \
  --std_message_level INFO --file_message_level INFO
```

In a separate Mac terminal, start the SSH tunnel. Replace `PI_USER@PI_HOST`
with the Pi's SSH login:

```sh
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:24680:127.0.0.1:24680 PI_USER@PI_HOST
```

Then, from the Mac's repository root:

```sh
bash pi_training/run_smoke.sh \
  ./build/mpi/examm_mpi 24680 "$HOME/examm-runs/smoke-mac-01"
```

Use a fresh output directory for each program on each run. The fixtures contain
24 rows each. The smoke run uses two training iterations per candidate and
weight decay of `0.015`; these are verification settings, not the coal experiment.
The Pi accepts one connection and exits when that search finishes or fails.
Restart the worker before starting another search. Keep both devices and the
SSH tunnel running for the duration of the search.

## Coal experiment

Restart the Pi worker with a new persistent output directory, keep the tunnel
running, and launch this on the Mac:

```sh
bash pi_training/run_coal.sh \
  ./build/mpi/examm_mpi 24680 "$HOME/examm-runs/coal-mac-01"
```

The launcher retains the data split, inputs, output, optimizer, node types,
10 islands of size 10, five training iterations, and 2,000-genome limit from
`scripts/base_run/coal_mpi.sh`. It uses two MPI ranks and one remote training
worker instead of the example's four MPI ranks. This changes concurrency and
can change the search trajectory; identical results are not expected.

The current launcher saves `all_best_genomes`, rather than the original
`the_best`, to retain best-so-far model files during a run. These files are not
full population checkpoints and do not provide exact search resumption.
The November 2 experiment used `the_best` and ended before completion.

Both coal launchers pass `--sequence_length 50`. The current data loader reads
`--train_sequence_length`, so this flag does not slice training sequences. The
Pi launcher preserves that existing behavior rather than changing the experiment.

## Verification

On the Mac, after building both programs, run:

```sh
python3 pi_training/tests/test_remote_training.py \
  ./build/mpi/examm_mpi ./build/pi_training/pi_train_server
```

This starts the relay and worker locally. It checks unavailable-worker failures,
dataset mismatch rejection, training and population progress, returned metrics
and parameter fingerprints, model-file creation, and shutdown. It exercises the
remote-training code path but does not establish execution on Pi hardware.

The protocol check can be built independently with a test-only source signature.
Keep assertions enabled; do not add `-DNDEBUG`:

```sh
c++ -std=c++20 -I pi_training \
  '-DEXAMM_PI_SOURCE_FINGERPRINT="protocol-test"' \
  pi_training/tests/protocol_test.cxx -o build/pi_protocol_test
./build/pi_protocol_test
```

For a hardware run, the Pi's `genome_*_pi` files record training iterations,
training/validation MSE, and gradient norms. Its `pi_worker` log records returned
genome IDs, validation scores, and fingerprints of the best parameters. Match
these with `Pi training result received` messages in the Mac's `main_0` log.
The Mac writes the population fitness log and saved genome files. A candidate
can train without improving its validation score; EXAMM may retain its initial
weights as the best weights.

## Operational limits

- The listener binds to loopback. Use SSH forwarding; the protocol itself does
  not supply encryption or authentication. Only connect trusted peers and data.
- The default startup socket timeout is 600 seconds; the training socket
  timeout is 7,200 seconds, configurable through `--pi_train_timeout_seconds`.
- A connection or training error aborts the run. There is no automatic retry,
  local fallback, or reconnect/resume mechanism.
- Use persistent output directories. Preserve both machines' logs before
  shutting down; temporary directories can disappear after a restart.

The preserved coal experiment is in
[`results/pi-coal-2026-11-02`](../results/pi-coal-2026-11-02/README.md).
