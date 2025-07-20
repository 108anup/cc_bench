# CC Bench

Benchmarking congestion control algorithms.

# Setup

- Refer `setup.sh` for installing dependencies.
- Refer `boot.sh` after every boot to set kernel parameters and setup in memory file system for logging. Assumes machine has at least 10GB free memory.

# Using

- Run simple experiment using `python sweep.py -t debug -o ../data/logs/experiment_name`.
- Use `-p` to run experiments in parallel (20 by default).
- Look at the experiment definitions in `sweep.py` to view and change what CCAs
and scenarios to run. Think of `sweep.py` as a file you can directly edit to
"declare" different scenarios. As an example use `python sweep.py -t sweep_bw -o ../data/logs/sweep_bw` to run CCAs on different bandwidths.

# Expected directory setup outside this repo

- `../data/logs` (change in common.py)
- `../data/figs` (change in common.py)
- `../mahimahi-traces` (change in common.py)
- `../ccas/genericCC` (change in run_experiment.sh)
