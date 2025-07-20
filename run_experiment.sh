#!/usr/bin/env bash

{
# https://stackoverflow.com/questions/2336977/can-a-shell-script-indicate-that-its-lines-be-loaded-into-memory-initially
# https://kvz.io/blog/bash-best-practices.html
set -euo pipefail

SCRIPT=$(realpath "$0")
SCRIPT_PATH=$(dirname "$SCRIPT")

EXPERIMENTS_PATH=$(realpath $SCRIPT_PATH/../)
GENERICCC_PATH=$(realpath $EXPERIMENTS_PATH/ccas/genericCC)
RAMDISK_PATH=/mnt/ramdisk

export SCRIPT_PATH
export EXPERIMENTS_PATH
export GENERICCC_PATH
export RAMDISK_PATH

printf '%.0s=' {1..80}
echo ""

echo "run_experiment.sh got inputs:"
echo "bw_ppms: $bw_ppms"
echo "ow_delay_ms: $ow_delay_ms"
echo "buf_size_bdp: $buf_size_bdp"
echo "n_flows: $n_flows"
echo "cca: $cca"
echo "jitter_shared": $jitter_shared
echo "rtprop_ratio: $rtprop_ratio"

echo "buf_size_bytes: $buf_size_bytes"
echo "exp_tag: $exp_tag"
echo "cca_param_tag: $cca_param_tag"
echo "downlink_trace_file: $downlink_trace_file"
echo "delay_uplink_trace_file: $delay_uplink_trace_file"
echo "cbr_uplink_trace_file: $cbr_uplink_trace_file"
echo "is_genericcc: $is_genericcc"
echo "group_dir: $group_dir"

echo "jitter_type: $jitter_type"
echo "staggered_start: $staggered_start"
echo "different_rtprop: $different_rtprop"
echo "duration_s: $duration_s"
echo "overlap_duration_s: $overlap_duration_s"
echo "iperf_log_interval_s: $iperf_log_interval_s"
echo "log_cbr_uplink: $log_cbr_uplink"
echo "log_jitter_uplink: $log_jitter_uplink"

echo "port: $port"
echo "outdir: $outdir"
echo "log_dmesg: $log_dmesg"

group_outdir=$outdir/$group_dir
mkdir -p $group_outdir

ramdisk_outdir=$RAMDISK_PATH/$group_dir
mkdir -p $ramdisk_outdir

uplink_log_path=$ramdisk_outdir/$exp_tag.log
[[ -f $uplink_log_path ]] && rm $uplink_log_path

uplink_log_path_jitter=$ramdisk_outdir/$exp_tag.jitter_log
[[ -f $uplink_log_path_jitter ]] && rm $uplink_log_path_jitter

dmesg_log_path=$ramdisk_outdir/$exp_tag.dmesg
[[ $log_dmesg == true ]] && [[ -f $dmesg_log_path ]] && rm $dmesg_log_path

genericcc_logfilepath=$ramdisk_outdir/$exp_tag.genericcc
[[ -f $genericcc_logfilepath ]] && rm $genericcc_logfilepath

export group_outdir
export ramdisk_outdir
export uplink_log_path
export genericcc_logfilepath

# Start Server
server_pids=()
if [[ $is_genericcc == true ]]; then
    echo "Using genericCC server"
    $GENERICCC_PATH/receiver $port &
    server_pids+=($!)
else
    echo "Using iperf3 server"
    for i in $(seq 0 $(( n_flows-1 ))); do
        iperf3 -s -p $(( port+i )) &
        server_pids+=($!)
    done
fi
echo "Started servers: ${server_pids[@]}"

if [[ $log_dmesg == true ]]; then
    # https://unix.stackexchange.com/questions/390184/dmesg-read-kernel-buffer-failed-permission-denied
    # sudo causes printing issues, so run dmesg without sudo.
    sudo dmesg --clear # Could do `dmesg --follow-new`, but some versions don't have that.
    dmesg --level info --follow --notime 1> $dmesg_log_path 2>&1 &
    dmesg_pid=$!
    echo "Started dmesg logging with: $dmesg_pid"
fi

# Propagation delay box, then jitter box with inf buffer, then cbr box
echo "Starting delay box and jitter box"
uplink_log_cmd=""
[[ $log_jitter_uplink == true ]] && uplink_log_cmd="--uplink-log=$uplink_log_path_jitter"
jitter_box_cmd="mm-link $delay_uplink_trace_file $downlink_trace_file $uplink_log_cmd -- "
{ [[ $jitter_type == "ideal" ]] || [[ $jitter_shared == false ]]; } && jitter_box_cmd=""
echo "shared jitter_box_cmd: $jitter_box_cmd"

box_cmd="mm-delay $ow_delay_ms $jitter_box_cmd $SCRIPT_PATH/bottleneck_box.sh"
echo "Starting bottleneck box using: $box_cmd"
eval "$box_cmd"

echo "Sleeping"
sleep 5 # iperf3 and mahimahi can gracefully cleanup any sockets etc.

[[ $log_jitter_uplink == true ]] && mv $uplink_log_path_jitter $group_outdir

echo "Killing servers"
for server_pid in "${server_pids[@]}"; do
    kill $server_pid
done

if [[ $log_dmesg == true ]]; then
    echo "Killing dmesg logging"
    kill $dmesg_pid
    mv $dmesg_log_path $group_outdir
fi

[[ $is_genericcc == true ]] && [[ -f $genericcc_logfilepath ]] && mv $genericcc_logfilepath $group_outdir

exit 0
}
