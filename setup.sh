#!/usr/bin/env bash

{
set -euo pipefail

SCRIPT=$(realpath "$0")
REPO=$(dirname "$SCRIPT")
GENERICCC="$REPO/../ccas/genericCC"

# Install mahimahi dependencies
sudo apt install -y autotools-dev autoconf libtool apache2 apache2-dev \
  protobuf-c-compiler protobuf-compiler libprotobuf-dev libssl-dev xcb \
  libxcb-composite0-dev libxcb-present-dev libcairo2-dev libpango1.0-dev \
  gnuplot dnsmasq

# Install mahimahi
if ! command -v mm-delay &> /dev/null; then
  cur_dir=$(pwd)
  mkdir -p $HOME/opt
  cd $HOME/opt
  git clone https://github.com/ravinet/mahimahi.git
  cd mahimahi
  ./autogen.sh
  ./configure
  make -j
  sudo make install
  cd $cur_dir
fi

# Setup other CCAs (copa)
## Comment out copa (genericcc_markovian) in sweep.py if you don't want to setup and run copa.
cd $GENERICCC
sudo apt install -y g++ makepp libboost-dev libprotobuf-dev protobuf-compiler libjemalloc-dev iperf libboost-python-dev
makepp
cd $REPO

# Setup and configure tools
sudo apt install iperf3 tcpdump tshark

# Allow tcmpdump to capture packets without sudo
sudo setcap cap_net_raw,cap_net_admin=eip $(which tcpdump) 

exit 0
}
