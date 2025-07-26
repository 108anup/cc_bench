#!/usr/bin/env bash

{
set -euo pipefail

# Set kernel parameters (e.g., TCP buffer sizes)
sudo sysctl -p ./sysctl.conf

# Enable BBR and Vegas
sudo modprobe tcp_bbr
sudo modprobe tcp_vegas

# Setup in-memory filesystem for live telemetry
sudo mkdir -p /mnt/ramdisk
sudo mount -t tmpfs -o size=24G tmpfs /mnt/ramdisk

exit 0
}
