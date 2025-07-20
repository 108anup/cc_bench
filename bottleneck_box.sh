#!/usr/bin/env bash

{
set -euo pipefail

printf '%.0s_' {1..80}
echo ""

echo "bottleneck_box.sh got inputs:"
echo "SCRIPT_PATH: $SCRIPT_PATH"

echo "buf_size_bytes: $buf_size_bytes"
echo "cbr_uplink_trace_file: $cbr_uplink_trace_file"
echo "downlink_trace_file: $downlink_trace_file"

echo "log_cbr_uplink: $log_cbr_uplink"

echo "group_outdir: $group_outdir"

echo "Starting CBR bottleneck"
uplink_log_cmd=""
[[ $log_cbr_uplink == true ]] && uplink_log_cmd="--uplink-log=$uplink_log_path"

box_cmd="mm-link $cbr_uplink_trace_file $downlink_trace_file --uplink-queue=droptail \
$uplink_log_cmd --uplink-queue-args="bytes=$buf_size_bytes" -- \
sh -c $SCRIPT_PATH/sender.sh"

echo "Inside bottleneck box box_cmd: $box_cmd"
eval "$box_cmd"

[[ $log_cbr_uplink == true ]] && mv $uplink_log_path $group_outdir

exit 0
}
