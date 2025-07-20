#!/usr/bin/env bash

{
set -euo pipefail

printf '%.0s_' {1..80}
echo ""

echo "tcpdump.sh got inputs:"

echo "exp_tag: $exp_tag"

echo "ramdisk_outdir: $ramdisk_outdir"

echo "flow_tag: $flow_tag"
echo "this_port: $this_port"
echo "sender_cmd: $sender_cmd"
echo "tcpdump_log_path: $tcpdump_log_path"
echo "tcpdump_csv_path: $tcpdump_csv_path"

if [[ $is_genericcc == false ]]; then
  tcpdump -i ingress -s 96 -w "$tcpdump_log_path" "tcp port $this_port" &
  tcpdump_pid=$!
fi

eval "$sender_cmd"

if [[ $is_genericcc == false ]]; then
  kill $tcpdump_pid 2> /dev/null
  tshark -n -T fields -E separator=, -e frame.time_epoch -e tcp.flags \
    -e tcp.srcport -e tcp.analysis.ack_rtt -e frame.len \
    -e tcp.seq -e tcp.ack -r \
    $tcpdump_log_path > $tcpdump_csv_path && rm $tcpdump_log_path
fi

exit 0
}
