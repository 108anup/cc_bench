# CC Bench
Benchmarking congestion control algorithms.

# Getting started
- Refer `setup.sh` for installing dependencies.
- Refer `boot.sh` after every boot to set kernel parameters and setup in memory file system for logging. Assumes machine has at least 10GB free memory.

# Expected directory setup outside this repo
- `../data` (change in common.py)
- `../figs` (change in common.py)
- `../mahimahi-traces` (change in common.py)
- `../ccas/genericCC` (change in run_experiment.sh)
