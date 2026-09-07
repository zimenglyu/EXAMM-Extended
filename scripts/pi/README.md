# Streaming genomes to a Raspberry Pi

`examm_mpi --send_to_pi` sends every new global best genome to a
`pi_genome_server` on the pi, which evaluates it on local test data and
writes `test_output/pi_server/pi_evaluations.csv` plus a `genome_<id>/`
directory per genome. Without `--send_to_pi` nothing is sent.

## Laptop -> pi

Pi:
```sh
cd ~/Documents/code/EXAMM-Extended
./scripts/pi/pi_genome_server.sh
```
Laptop:
```sh
./scripts/pi/c172_mpi_to_pi.sh
```
Pi address defaults to `192.168.0.70:5555` (`DEFAULT_PI_HOST` /
`DEFAULT_PI_PORT` in `mpi/examm_mpi.cxx`); override with `--pi_host` / `--pi_port`.

## Anvil -> pi

### Keys (one time)

| key | make it on | put its `.pub` in |
|---|---|---|
| Anvil key (compute node -> login node) | Anvil | Anvil `~/.ssh/authorized_keys` |
| Pi key (pi -> Anvil) | Pi | Anvil `~/.ssh/authorized_keys` |

On Anvil:
```sh
ssh-keygen -t ed25519 -N ""
cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys
```
On the pi:
```sh
ssh-keygen -t ed25519
cat ~/.ssh/id_ed25519.pub      # append this line to ~/.ssh/authorized_keys on Anvil
```

### Build (one time)

Anvil:
```sh
cd ~/code/EXAMM-Extended && git pull
cd build && cmake .. && make examm_mpi
```
Edit `scripts/Anvil/c172_to_pi.sh`: `#SBATCH -A`, `--mail-user`, `EXAMM=`, `LOGIN_NODE`.

Pi:
```sh
cd ~/Documents/code/EXAMM-Extended && git pull
cd build && cmake .. && make pi_genome_server
```

### Run (every time, in this order)

1. Pi, terminal 1:
   ```sh
   ./scripts/pi/pi_genome_server.sh
   ```
2. Pi, terminal 2 (same `loginNN` as `LOGIN_NODE` in the job script):
   ```sh
   ssh -o ServerAliveInterval=60 -R 5555:localhost:5555 x-zlyu2@login03.anvil.rcac.purdue.edu
   ```
3. In that shell:
   ```sh
   cd ~/code/EXAMM-Extended
   sbatch scripts/Anvil/c172_to_pi.sh
   ```
4. Keep both pi terminals open until the job finishes.
