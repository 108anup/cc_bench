#!/usr/bin/env bash

{
set -euo pipefail

sudo sysctl -p ./sysctl.conf
sudo modprobe tcp_bbr
sudo modprobe tcp_vegas
sudo mkdir -p /mnt/ramdisk
sudo mount -t tmpfs -o size=10G tmpfs /mnt/ramdisk

exit 0
}
