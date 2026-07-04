# TCP-Isolated Worker Runtime Reproducibility

This experiment approximates a containerized master-worker deployment when
nested Docker is unavailable on the rented server.  Each logical worker runs as
an independent TCP service on its own port.  The master serializes row-pair
tasks over sockets, receives streamed row completions, checks first
decodability online, and sends explicit cancellation messages.

On a Docker-enabled local host, the same worker entrypoint can now be launched
as one container per worker with `--use-docker-workers`.  This validates the
container execution path without changing the TCP protocol or scheduler.

## Server Environment

- Server shape observed during the run: 96 CPU cores, about 503 GiB RAM.
- Nested Docker/Podman was unavailable in the provided environment.
- `unshare -n` was not permitted, so network namespaces could not be created.
- The worker entrypoint is still Docker-ready through
  `docker/Dockerfile.network-worker` for hosts that allow container launch.

## Local Docker Container Validation

Build the worker image:

```bash
docker build -f docker/Dockerfile.network-worker \
  -t coded-learning-network-worker:local .
```

Run a quick container-per-worker smoke test:

```bash
python run_network_container_experiment.py --quick \
  --use-docker-workers \
  --base-port 21220 \
  --out local_docker_tcp_smoke \
  --docker-container-prefix coded-local-smoke \
  --strategies sparse_flexible_static rank_aware_sparse_flexible deadline_aware_sparse_flexible
```

The local Docker scaling validation used 8 and 16 container workers with 4 ms
RTT and 100 Mbps transfer delay injected into the same TCP path.  The aggregated
results are in `local_docker_container_diagnostics/container_scaling_summary.csv`.
In the 16-worker setting, rank-aware assignment reduced mean and p95
first-decode latency by 26.7% and 38.4%; deadline-aware assignment reduced p95
latency by 33.0%.

## Commands

```bash
python run_network_container_experiment.py \
  --samples 12000 --features 1200 --density 0.008 \
  --shards 16 --workers 16 --rounds 12 \
  --scenario phase --drift-period 6 \
  --straggler-fraction 0.45 --straggler-slowdown 0.08 \
  --sleep-scale 0.03 --cost-scale 0.006 \
  --seed 11 --base-port 19300 \
  --out network_container_server/w16_seed_11
```

Repeat with seeds `23`, `31`, and `43`, using different base ports.  The
paper run used base ports `19300`, `19400`, `19500`, and `19600`.

```bash
python analyze_network_container_results.py \
  network_container_server/w16_seed_11 \
  network_container_server/w16_seed_23 \
  network_container_server/w16_seed_31 \
  network_container_server/w16_seed_43 \
  --out network_container_diagnostics

python analyze_end_to_end_ci.py --out end_to_end_ci_diagnostics
```

## Main Result

In the 16-worker, 16-shard TCP-isolated runtime, decode-aware and
deadline-aware scheduling reduce p95 first-decode latency by about 55% over
static sparse-flexible placement.  They reduce barrier time-to-loss by 32.7%
and 28.5%, respectively; seed-level bootstrap intervals are 15.2--49.8% and
1.9--48.9%.
