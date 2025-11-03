# distributed-gpu-test

Lightweight, portable example that builds a container (base: nvcr.io/nvidia/pytorch:24.07-py3) and runs a tiny distributed PyTorch/DeepSpeed training job on synthetic data — suitable for GPU cluster acceptance tests (drivers, container runtime, NCCL, SLURM).

What's included:
- Dockerfile — image based on nvcr.io/nvidia/pytorch:24.07-py3.
- train.py — tiny distributed training (synthetic data).
- slurm_train.sbatch — example SLURM submission.
- .github/workflows/distributed-gpu-test-ci.yaml — CI: build → remote SLURM test → push to GHCR.

Runner setup: https://github.com/dashabalashova/distributed-gpu-test/settings/actions/runners/ – in the ./config.sh step the runner is registered with the label docker-host.


Secrets https://github.com/dashabalashova/distributed-gpu-test/settings/secrets/actions:
- CR_PAT
- SLURM_HOST
- SLURM_SSH_KEY
- SLURM_SSH_PORT
- SLURM_USER

To run the container from the registry (instead of the .sqsh file):
1. [configure authentication](https://docs.nebius.com/slurm-soperator/jobs/containers/pyxis-enroot#auth)
2. replace `--container-image="$SLURM_SUBMIT_DIR/distributed-gpu-test.sqsh"` with
`--container-image="ghcr.io/dashabalashova/distributed-gpu-test:v0.1.0"`