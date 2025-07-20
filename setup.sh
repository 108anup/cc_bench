#!/usr/bin/env bash

{
set -euo pipefail

# Install mahimahi and dependencies
sudo apt-get install autotools-dev autoconf libtool apache2 apache2-dev \
    protobuf-compiler libprotobuf-dev libssl-dev xcb libxcb-composite0-dev \
    libxcb-present-dev libcairo2-dev libpango1.0-dev gnuplot dnsmasq

cur_dir=$(pwd)
mkdir -p $HOME/opt
cd $HOME/opt
git clone git@github.com:ravinet/mahimahi.git
cd mahimahi
./autogen.sh
./configure
make -j
sudo make install
cd $cur_dir

sudo apt install iperf3 tcpdump

sudo mkdir -p /mnt/ramdisk  # For storing live telemetry

sudo setcap cap_net_raw,cap_net_admin=eip $(which tcpdump)

exit 0
}
