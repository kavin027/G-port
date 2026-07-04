# Direct Multi-Node Port Probe

Date: 2026-05-26

Purpose: determine whether the current rented cloud hosts can support a
non-forwarded master-to-remote-worker TCP experiment.

## Probe Summary

- Previously used short-lived hosts were no longer reachable over SSH.
- The current fresh server is reachable over SSH and can run independent TCP
  worker processes.
- Workers can bind to `0.0.0.0` on the remote host and create ready files.
- Local direct connections to the worker port range were rejected.
- One high port appeared connectable from the local machine, but a raw socket
  read returned an SSH banner from the provider gateway, not the worker
  service.  It is therefore not a valid worker-data path.

## Result

The current environment does not support a routable direct multi-node
deployment.  The strongest completed system paths remain:

1. same-host independent TCP worker services;
2. network-constrained TCP stress with explicit RTT/bandwidth and cancellation;
3. fresh-server direct-service rerun from the submitted artifact package;
4. restricted two-node SSH-forwarded validation.

## Artifact Action

We added `run_direct_remote_network_experiment.py` and
`run_direct_remote_sweep.py` so that, if a future cloud allocation exposes
routable worker ports, the same runtime can run without SSH forwarding.  These
scripts use SSH only to start remote workers and copy problem files; experiment
traffic goes directly from the master to `--worker-host:--remote-base-port`.

The paper should not claim a completed direct multi-node deployment until such
a run succeeds.
