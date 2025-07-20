#!/bin/bash

killall -g dmesg
killall -g iperf3
killall -g mm-delay -s SIGKILL
killall -g mm-link -s SIGKILL
killall -g sender
killall -g receiver
killall -g tcpdump

ps -aux | grep dmesg
ps -aux | grep iperf3
ps -aux | grep mm-delay
ps -aux | grep mm-link
ps -aux | grep sender
ps -aux | grep receiver
ps -aux | grep tcpdump