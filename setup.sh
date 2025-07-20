#!/usr/bin/env bash

{
set -euo pipefail

# Install mahimahi and dependencies
sudo apt install -y autotools-dev autoconf libtool apache2 apache2-dev \
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

# Setup other CCAs (copa)
## Comment out copa (genericcc_markovian) in sweep.py if you don't want to setup and run copa.
cd ../ccas/genericCC
sudo apt install -y g++ makepp libboost-dev libprotobuf-dev protobuf-compiler libjemalloc-dev iperf libboost-python-dev
makepp
cd ../cc_bench

# Setup and configure tools
sudo apt install iperf3 tcpdump
sudo setcap cap_net_raw,cap_net_admin=eip $(which tcpdump) # allow tcmpdump to capture packets without sudo

# Setup in-memory filesystem for live telemetry
sudo mkdir -p /mnt/ramdisk

exit 0
}
