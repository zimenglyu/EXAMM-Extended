#!/usr/bin/env python3
"""Exercise a complete MPI search with training performed by a Pi server.

Usage: python3 test_remote_training.py EXAMM_MPI PI_TRAIN_SERVER

The small temporary dataset and search settings are used only by this test;
they do not define the settings for the research experiment.
"""

import csv
import math
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional


SEARCH_TIMEOUT = 120
UNAVAILABLE_TIMEOUT = 30
STARTUP_TIMEOUT = 15
SHUTDOWN_TIMEOUT = 15


def write_series(path: Path, offset: float) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["x"])
        for index in range(24):
            writer.writerow([f"{0.3 + offset + 0.2 * math.sin(index / 4):.8f}"])


def make_search_command(
    executable: Path, root: Path, port: int, relative_paths: bool = False
) -> list[str]:
    train_file = root / "train.csv"
    validation_file = root / "validation.csv"
    write_series(train_file, 0.0)
    write_series(validation_file, 0.02)
    return [
        "mpirun", "-np", "2", str(executable.resolve()),
        "--training_filenames", train_file.name if relative_paths else str(train_file),
        "--validation_filenames", validation_file.name if relative_paths else str(validation_file),
        "--input_parameter_names", "x",
        "--output_parameter_names", "x",
        "--time_offset", "1",
        "--number_islands", "1",
        "--island_size", "2",
        "--max_genomes", "6",
        "--bp_iterations", "2",
        "--num_mutations", "2",
        "--weight_update", "adagrad",
        "--weight_decay", "0.015",
        "--eps", "0.000001",
        "--possible_node_types", "simple",
        "--save_genome_option", "the_best",
        "--seed", "1234",
        "--output_directory", str(root / "search-output"),
        "--std_message_level", "INFO",
        "--file_message_level", "INFO",
        "--pi_train_host", "127.0.0.1",
        "--pi_train_port", str(port),
    ]


def read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def text_logs(directory: Path) -> dict[Path, str]:
    if not directory.is_dir():
        return {}
    return {
        path: read_text(path)
        for path in directory.iterdir()
        if path.is_file() and not path.suffix
    }


def recent_output(path: Path, lines: int = 80) -> str:
    return "\n".join(read_text(path).splitlines()[-lines:])


def start_process(
    command: list[str], output: Path, workdir: Optional[Path] = None
) -> subprocess.Popen:
    with output.open("wb") as stream:
        return subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True, cwd=workdir,
        )


def stop_process(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # The MPI launcher may have exited while worker processes remain in its group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def wait_for_process(process: subprocess.Popen, seconds: int, label: str, log: Path) -> int:
    try:
        return process.wait(timeout=seconds)
    except subprocess.TimeoutExpired as error:
        raise AssertionError(
            f"{label} did not finish within {seconds} seconds:\n{recent_output(log)}"
        ) from error


def wait_for_server(process: subprocess.Popen, output_dir: Path, console: Path) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        server_logs = "\n".join(text_logs(output_dir).values()) + read_text(console)
        if "listening" in server_logs.lower():
            return
        if process.poll() is not None:
            raise AssertionError(
                f"Server exited before listening ({process.returncode}):\n"
                f"{recent_output(console)}\n{server_logs}"
            )
        time.sleep(0.05)
    raise AssertionError(f"Server did not report readiness:\n{recent_output(console)}")


def assert_search_progress(output_dir: Path) -> None:
    fitness_file = output_dir / "fitness_log.csv"
    if not fitness_file.is_file():
        raise AssertionError("Search did not produce fitness_log.csv")
    with fitness_file.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, skipinitialspace=True))
    progress = []
    for row in rows:
        try:
            genomes = int(row["Inserted Genomes"])
            epochs = float(row["Total BP Epochs"])
            mse = float(row["Best Val. MSE"])
        except (KeyError, TypeError, ValueError) as error:
            raise AssertionError(f"Invalid fitness-log row: {row}") from error
        if genomes > 0 and epochs > 0 and math.isfinite(mse) and 0 <= mse < 1e100:
            progress.append((genomes, epochs))
    if len(progress) < 2 or any(end <= start for start, end in zip(progress[0], progress[-1])):
        raise AssertionError(
            "Fitness log did not record multiple inserted genomes and increasing "
            "training epochs with finite validation MSE"
        )
    if not any(path.stat().st_size > 0 for path in output_dir.glob("global_best_genome_*.bin")):
        raise AssertionError("Search did not save a non-empty global-best genome")


def assert_remote_results(search_dir: Path, server_dir: Path, console: Path) -> None:
    server_logs = text_logs(server_dir)
    trained_ids = set()
    for path, log in server_logs.items():
        match = re.fullmatch(r"genome_(\d+)_pi", path.name)
        if (match and re.search(r"iteration\s+\d+.*v_mse:", log)
                and "backpropagation completed" in log
                and "weight_decay active: lambda=0.015000" in log):
            trained_ids.add(match.group(1))
    if len(trained_ids) < 2:
        raise AssertionError("Pi server did not log training and validation of multiple genomes")
    master_logs = "\n".join(text_logs(search_dir).values()) + read_text(console)
    received = set(re.findall(r"Pi training result received for genome\s+(\d+)", master_logs))
    if len(received) < 2:
        raise AssertionError("MPI search did not log multiple distinct returned Pi genome IDs")
    if not received.issubset(trained_ids):
        raise AssertionError(
            f"Returned genome IDs {sorted(received)} do not match Pi training logs {sorted(trained_ids)}"
        )
    if "mutating a random genome" not in master_logs:
        raise AssertionError("Search logs did not show mutation beyond the initial seed")
    all_server_logs = "\n".join(server_logs.values())
    pi_received = set(re.findall(r"received genome\s+(\d+)", all_server_logs, re.IGNORECASE))
    pi_returned = set(re.findall(r"returned genome\s+(\d+)", all_server_logs, re.IGNORECASE))
    if not received.issubset(pi_received & pi_returned):
        raise AssertionError("Pi server did not log receiving and returning each reported genome")
    result_fields = r"(\d+) validation MSE ([\d.eE+-]+) parameter fingerprint (\d+)"
    master_results = set(re.findall(r"Pi training result received for genome\s+" + result_fields, master_logs))
    pi_results = set(re.findall(r"returned genome\s+" + result_fields, all_server_logs))
    if len(master_results) < 2 or not master_results.issubset(pi_results):
        raise AssertionError("Returned validation metrics or trained parameters changed in transit")
    if "termination" not in all_server_logs.lower():
        raise AssertionError("Pi server did not log a graceful termination frame")


def run_unavailable_test(executable: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="pi-training-unavailable-") as temporary:
        root = Path(temporary)
        output = root / "mpi.log"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reserved:
            reserved.bind(("127.0.0.1", 0))
            port = reserved.getsockname()[1]
            process = start_process(make_search_command(executable, root, port), output)
            try:
                status = wait_for_process(process, UNAVAILABLE_TIMEOUT, "MPI search", output)
            finally:
                stop_process(process)
        logs = read_text(output) + "\n" + "\n".join(text_logs(root / "search-output").values())
        if status == 0:
            raise AssertionError(
                "MPI search succeeded even though the Pi training server was unavailable; "
                "remote training must not silently fall back to local training"
            )
        if not re.search(r"connect|refused|unreachable|timed?\s*out", logs, re.IGNORECASE):
            raise AssertionError(f"MPI failure did not identify a connection problem:\n{recent_output(output)}")
    print("PASS: an unavailable Pi training server fails the search promptly")


def run_dataset_mismatch_test(executable: Path, server_executable: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="pi-training-mismatch-") as temporary:
        root = Path(temporary)
        master_dir = root / "master"
        server_workdir = root / "server"
        master_dir.mkdir()
        server_workdir.mkdir()
        write_series(server_workdir / "train.csv", 0.15)
        write_series(server_workdir / "validation.csv", 0.02)
        server_dir = root / "server-output"
        server_console = root / "server.log"
        mpi_console = root / "mpi.log"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reserved:
            reserved.bind(("127.0.0.1", 0))
            port = reserved.getsockname()[1]
        server_command = [
            str(server_executable.resolve()), "--port", str(port),
            "--output_directory", str(server_dir),
            "--std_message_level", "INFO", "--file_message_level", "INFO",
        ]
        server = start_process(server_command, server_console, server_workdir)
        try:
            wait_for_server(server, server_dir, server_console)
            command = make_search_command(executable, master_dir, port, relative_paths=True)
            search = start_process(command, mpi_console, master_dir)
            try:
                status = wait_for_process(search, UNAVAILABLE_TIMEOUT, "MPI search", mpi_console)
                server_status = wait_for_process(server, SHUTDOWN_TIMEOUT, "Pi server", server_console)
                if status == 0 or server_status == 0:
                    raise AssertionError("Dataset mismatch did not fail both MPI and the Pi server")
                server_logs = "\n".join(text_logs(server_dir).values()) + read_text(server_console)
                search_logs = "\n".join(text_logs(master_dir / "search-output").values()) + read_text(mpi_console)
                mismatch = "datasets do not match"
                if mismatch not in server_logs.lower() or mismatch not in search_logs.lower():
                    raise AssertionError(
                        "Dataset mismatch was not reported on both sides:\n"
                        f"MPI:\n{recent_output(mpi_console)}\nServer:\n{recent_output(server_console)}"
                    )
                if (re.search(r"received genome\s+\d+|backpropagation completed", server_logs, re.IGNORECASE)
                        or "backpropagation completed" in search_logs.lower()
                        or "Pi training result received for genome" in search_logs):
                    raise AssertionError("A genome was trained or returned despite the dataset mismatch")
                fitness = master_dir / "search-output" / "fitness_log.csv"
                if fitness.is_file():
                    with fitness.open(encoding="utf-8", newline="") as stream:
                        if list(csv.DictReader(stream, skipinitialspace=True)):
                            raise AssertionError("Search inserted a genome despite the dataset mismatch")
            finally:
                stop_process(search)
        finally:
            stop_process(server)
    print("PASS: different datasets with the same filenames stop the search before training or insertion")


def run_remote_test(
    executable: Path, server_executable: Path, separate_directories: bool = False
) -> None:
    with tempfile.TemporaryDirectory(prefix="pi-training-search-") as temporary:
        root = Path(temporary)
        master_dir = root
        server_workdir = None
        if separate_directories:
            master_dir = root / "master"
            server_workdir = root / "server"
            master_dir.mkdir()
            server_workdir.mkdir()
            write_series(server_workdir / "train.csv", 0.0)
            write_series(server_workdir / "validation.csv", 0.02)
        server_dir = root / "server-output"
        server_console = root / "server.log"
        mpi_console = root / "mpi.log"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reserved:
            reserved.bind(("127.0.0.1", 0))
            port = reserved.getsockname()[1]
        server_command = [
            str(server_executable.resolve()), "--port", str(port),
            "--output_directory", str(server_dir),
            "--std_message_level", "INFO", "--file_message_level", "INFO",
        ]
        server = start_process(server_command, server_console, server_workdir)
        try:
            wait_for_server(server, server_dir, server_console)
            command = make_search_command(executable, master_dir, port, separate_directories)
            search = start_process(command, mpi_console, master_dir)
            try:
                status = wait_for_process(search, SEARCH_TIMEOUT, "MPI search", mpi_console)
                if status != 0:
                    raise AssertionError(f"MPI search exited with {status}:\n{recent_output(mpi_console)}")
                server_status = wait_for_process(server, SHUTDOWN_TIMEOUT, "Pi server", server_console)
                if server_status != 0:
                    raise AssertionError(
                        f"Pi server exited with {server_status}:\n{recent_output(server_console)}"
                    )
                assert_remote_results(master_dir / "search-output", server_dir, mpi_console)
                assert_search_progress(master_dir / "search-output")
            finally:
                stop_process(search)
        finally:
            stop_process(server)
    dataset_location = "separate matching datasets" if separate_directories else "shared dataset paths"
    print(
        f"PASS ({dataset_location}): MPI search trained multiple genomes on the Pi, "
        "updated the population, and shut down cleanly"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {Path(sys.argv[0]).name} EXAMM_MPI PI_TRAIN_SERVER", file=sys.stderr)
        raise SystemExit(2)
    try:
        mpi_executable, server_executable = map(Path, sys.argv[1:])
        if not mpi_executable.is_file():
            raise AssertionError(f"MPI executable does not exist: {mpi_executable}")
        run_unavailable_test(mpi_executable)
        if not server_executable.is_file():
            raise AssertionError(f"Pi server executable does not exist: {server_executable}")
        run_dataset_mismatch_test(mpi_executable, server_executable)
        run_remote_test(mpi_executable, server_executable)
        run_remote_test(mpi_executable, server_executable, separate_directories=True)
    except (AssertionError, OSError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from error
