# Streaming genomes to a Raspberry Pi

`examm_mpi --send_to_pi` streams every new global best genome over TCP to a
`pi_genome_server` running on the pi, which evaluates each one on local test
data (MSE/MAE, inference time, throughput, model latency, optional INA219
power/energy). Without `--send_to_pi` nothing is sent and the MPI code is
unchanged.

Wire format is the same as the MPI messages: `int32 length` then the bytes of
`RNN_Genome::write_to_array()`. The master never blocks on the network: a
background thread sends, retries every 5 s if the pi is unreachable, and
queues genomes until it reconnects.

Results on the pi: `test_output/pi_server/pi_evaluations.csv` plus one
`genome_<id>/` directory per genome with predictions and the `.bin`.

---

## Laptop -> pi (same network)

**Pi, terminal 1:**
```sh
cd ~/Documents/code/EXAMM-Extended
cd build && cmake .. && make pi_genome_server && cd ..
./scripts/pi/pi_genome_server.sh
```
**Laptop:**
```sh
./scripts/pi/c172_mpi_to_pi.sh
```
The pi address defaults to `DEFAULT_PI_HOST`/`DEFAULT_PI_PORT` in
`mpi/examm_mpi.cxx` (`192.168.0.70:5555`); override with `--pi_host` /
`--pi_port`. Find the pi's address with `hostname -I` on the pi.

---

## Anvil -> pi

Anvil cannot reach the pi directly (private address, no public IP), so the
pi opens an ssh session to a login node and the job tunnels through it:

```
examm_mpi -> [A] compute:5555 --ssh -L--> [B] login03:5555 --ssh -R--> [C] pi:5555 -> pi_genome_server
```

### One-time setup

**On Anvil (any login node)** - lets compute nodes ssh to login nodes
without a password (home is shared, so one key works on every node):
```sh
ssh-keygen -t ed25519 -N ""
cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys
cd ~/code/EXAMM-Extended && git pull
cd build && cmake .. && make examm_mpi
```
Then edit `scripts/Anvil/c172_to_pi.sh`: `#SBATCH -A`, `--mail-user`,
`EXAMM=` path, and `LOGIN_NODE` (the login node the pi will ssh to).

**On the pi** - so it can log in to Anvil:
```sh
ssh-keygen -t ed25519            # if it has no key yet
cat ~/.ssh/id_ed25519.pub        # add this line to ~/.ssh/authorized_keys on Anvil
ssh x-zlyu2@login03.anvil.rcac.purdue.edu   # should give a shell
```

### Every run (order matters)

Use `tmux` on the pi so the terminals survive.

1. **Pi, terminal 1 - the listener:**
   ```sh
   ./scripts/pi/pi_genome_server.sh
   ```
   wait for `waiting for a connection on port 5555`.

2. **Pi, terminal 2 - the tunnel (this is also your Anvil shell):**
   ```sh
   ssh -o ServerAliveInterval=60 -R 5555:localhost:5555 x-zlyu2@login03.anvil.rcac.purdue.edu
   ```
   Always use the same explicit `loginNN` as `LOGIN_NODE` in the job script,
   not the round-robin `anvil.rcac.purdue.edu`.

3. **In that shell, submit the job:**
   ```sh
   cd ~/code/EXAMM-Extended
   sbatch scripts/Anvil/c172_to_pi.sh
   ```

4. **Keep both pi terminals open** until `squeue -u $USER` shows the job is
   gone. If terminal 2 drops, re-open it; the job reconnects and catches up.

Success looks like: `pi sender connected to 127.0.0.1:5555` in the job's
`.output`, and `connected` / `received genome N` / `genome N: MSE ...` in
terminal 1 on the pi.

### Checking each hop

Stop at the first check that fails.

**C - on the pi:** `ss -tlnp | grep 5555` -> a `pi_genome_server` LISTEN line.

**B - in terminal 2 (login node):**
```sh
hostname                                    # must match LOGIN_NODE
ss -tln | grep 5555                         # 127.0.0.1:5555 LISTEN = reverse tunnel is up
bash -c 'echo > /dev/tcp/127.0.0.1/5555' && echo B_OK   # pi terminal 1 prints connected/closed
```
If `ss` shows nothing: scroll up in terminal 2 for
`Warning: remote port forwarding failed for listen port 5555` - someone else
holds that port on the shared login node. Pick another port (e.g. 55555) in
the `-R`, in `pi_genome_server.sh --port`, and `PI_PORT` in the job script.

**A - from a compute node:**
```sh
srun -A cis251123 -p debug -n 1 -t 5:00 --pty bash
ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new login03 hostname
ssh -N -L 5555:localhost:5555 login03 &
sleep 2; bash -c 'echo > /dev/tcp/127.0.0.1/5555' && echo A_OK
kill %1; exit
```

### What the error messages mean

| where | message | meaning |
|---|---|---|
| pi terminal 2 | `connect_to localhost port 5555: failed` | A and B work; `pi_genome_server` is not running on the pi (C) |
| job `.error` | `channel N: open failed: connect failed: Connection refused` | A works; the pi's reverse tunnel was not up on the login node when the job ran (B) |
| job `.error` | `Permission denied (publickey)` | Anvil key setup (one-time step) |
| job `.error` | `Could not resolve hostname` | `LOGIN_NODE` name is wrong from a compute node; try the full `loginNN.anvil.rcac.purdue.edu` |
| job `.output` | `could not open tunnel to login03` | the compute node's ssh died; see the `.error` line above it |
| job `.output` | `pi sender could not connect ..., will retry` | harmless while the tunnel is being re-established |
| pi terminal 1 | `INA219 requested but could not open /dev/i2c-1` | no sensor wired / I2C disabled; run continues with zeros in the power columns |
