#!/usr/bin/env bash

{
set -euo pipefail

printf '%.0s_' {1..80}
echo ""

echo "sender.sh got inputs:"
echo "SCRIPT_PATH: $SCRIPT_PATH"
echo "GENERICCC_PATH: $GENERICCC_PATH"

echo "ow_delay_ms: $ow_delay_ms"
echo "n_flows: $n_flows"
echo "cca: $cca"
echo "jitter_shared": $jitter_shared
echo "rtprop_ratio: $rtprop_ratio"

echo "exp_tag: $exp_tag"
echo "downlink_trace_file: $downlink_trace_file"
echo "delay_uplink_trace_file: $delay_uplink_trace_file"
echo "is_genericcc: $is_genericcc"

echo "jitter_type: $jitter_type"
echo "staggered_start: $staggered_start"
echo "different_rtprop: $different_rtprop"
echo "duration_s: $duration_s"
echo "overlap_duration_s: $overlap_duration_s"
echo "iperf_log_interval_s: $iperf_log_interval_s"

echo "port: $port"

echo "group_outdir: $group_outdir"
echo "ramdisk_outdir: $ramdisk_outdir"

tcpdump_pids=()
client_pids=()

launch_sender() {
    this_duration_s=$1
    flow_tag=$2
    exp_tag=$3
    this_extra_delay=$4
    port_increment=$5

    this_port=$(( $port + $port_increment ))
    export this_port

    tcpdump_log_path=$ramdisk_outdir/$flow_tag$exp_tag.pcap
    [[ -f $tcpdump_log_path ]] && rm $tcpdump_log_path
    tcpdump_csv_path="${tcpdump_log_path}.csv"
    [[ -f $tcpdump_csv_path ]] && rm $tcpdump_csv_path

    flow_genericcc_logfilepath=$ramdisk_outdir/$flow_tag$exp_tag.genericcc
    [[ -f $flow_genericcc_logfilepath ]] && rm $flow_genericcc_logfilepath

    flow_iperf_log_path=$ramdisk_outdir/$flow_tag$exp_tag.json
    [[ -f $flow_iperf_log_path ]] && rm $flow_iperf_log_path

    if [[ $is_genericcc == true ]]; then
        short_cca=$(echo $cca | sed 's/genericcc_//g')

        cc_params=""
        if [[ $short_cca == "markovian" ]]; then
            # cc_params="delta_conf=do_ss:auto:0.5"
            cc_params="delta_conf=do_ss:constant_delta:0.125"
            if [[ ${COPA_CONST_VELOCITY:-false} == true ]]; then
              cc_params="delta_conf=const_velocity:do_ss:constant_delta:0.125"
            fi
        # elif [[ $short_cca == "slow_conv" ]] || [[ $short_cca == "fast_conv" ]]; then
        else
            cc_params="logfilepath=$flow_genericcc_logfilepath"
        fi
    fi

    iperf_cca=$cca
    if [[ $cca == "bbr3" ]]; then
        # They renamed bbr3 to bbr.
        # Only run this in the appropriate VM.
        HOSTNAME="$(hostname)"
        if [[ $HOSTNAME != "bbrv3-testbed" ]]; then
            echo "Need to run on bbrv3-testbed for using bbr3"
            exit 1
        fi
        iperf_cca="bbr"
    fi

    # Since we launch flows in parallel, this can cause race conditions.
    # For now, just run experiments for different variants of ndd separately.
    # Here we just check that the params are set correctly.
    if [[ $cca == "ndd" ]]; then
      # sudo /home/anupa/Projects/fairCC/ndd-kernel/set_ndd_params.py --reset > /dev/null
      # sudo -E /home/anupa/Projects/fairCC/ndd-kernel/set_ndd_params.py
      # sudo /home/anupa/Projects/fairCC/ndd-kernel/set_ndd_params.py --print
      echo "Temporarily disabling checking of ndd params"
      # sudo /home/anupa/Projects/fairCC/ndd-kernel/set_ndd_params.py --print
      # sudo -E /home/anupa/Projects/fairCC/ndd-kernel/set_ndd_params.py --check
    fi

    # Sender command
    sender_cmd="iperf3 -c $MAHIMAHI_BASE -p $this_port --congestion $iperf_cca "
    sender_cmd+="-t $this_duration_s --json --logfile $flow_iperf_log_path -i $iperf_log_interval_s"

    if [[ $is_genericcc == true ]]; then
        sender_cmd="$GENERICCC_PATH/sender serverip=$MAHIMAHI_BASE serverport=$port "
        sender_cmd+="offduration=0 onduration=${this_duration_s}000 "
        sender_cmd+="cctype=$short_cca $cc_params "
        sender_cmd+="traffic_params=deterministic,num_cycles=1 "
        sender_cmd+="linklog=$tcpdump_csv_path "
    fi

    # Note, uplink and downlink here have been swapped as we want jitter on the
    # ACKs as we want jitter after the bottleneck
    downlink_log_cmd=""
    [[ $log_jitter_uplink == true ]] && downlink_log_cmd="--downlink-log=$uplink_log_path_jitter"
    delay_cmd="mm-delay $this_extra_delay"
    [[ $this_extra_delay -le 0 ]] && delay_cmd=""
    jitter_box_cmd="mm-link $downlink_trace_file $delay_uplink_trace_file $downlink_log_cmd -- "
    { [[ $jitter_type == "ideal" ]] || [[ $jitter_shared == true ]] || [[ $flow_tag != "flow[1]-" ]]; } && jitter_box_cmd=""

    # Exports for tcpdump.sh
    export flow_tag
    export this_port
    export sender_cmd
    export tcpdump_log_path
    export tcpdump_csv_path

    tcpdump_cmd="sh -c '$SCRIPT_PATH/tcpdump.sh'"
    box_cmd="$delay_cmd $jitter_box_cmd $tcpdump_cmd"
    echo "Starting sender using: $box_cmd"
    # https://unix.stackexchange.com/questions/356534/how-to-run-string-with-values-as-a-command-in-bash
    eval "$box_cmd &"
    client_pids+=($!)
}

for i in $(seq 1 $n_flows); do
    echo "Starting flow $i"
    flow_tag="flow[$i]-"

    if [[ $staggered_start == false ]]; then
        if [[ $different_rtprop == false ]]; then
            launch_sender $duration_s $flow_tag $exp_tag 0 $(( $i - 1 ))
        else
            launch_sender $duration_s $flow_tag $exp_tag $(( (rtprop_ratio-1) * ow_delay_ms * (i-1) )) $(( $i - 1 ))
        fi
        sleep 5
    else
        if [[ $different_rtprop == false ]]; then
            launch_sender $(( overlap_duration_s*n_flows )) $flow_tag $exp_tag 0 $(( $i - 1 ))
        else
            launch_sender $(( overlap_duration_s*n_flows )) $flow_tag $exp_tag $(( (rtprop_ratio-1) * ow_delay_ms * (i-1) )) $(( $i - 1 ))
        fi
        echo "Sleeping for $overlap_duration_s seconds"
        sleep $overlap_duration_s
    fi
done

wait "${client_pids[@]}"

for i in $(seq 1 $n_flows); do
    flow_tag="flow[$i]-"
    # for ext in genericcc json pcap pcap.csv; do
    for ext in genericcc json pcap.csv; do
        log_file=$ramdisk_outdir/$flow_tag$exp_tag.$ext
        [[ -f $log_file ]] && mv $log_file $group_outdir
    done
done

exit 0
}
