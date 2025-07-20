import argparse
import os
import random
import math


def smooth(f, start_time, pkts_per_ms, end_time):
    assert pkts_per_ms == math.floor(pkts_per_ms)
    assert start_time == math.floor(start_time)
    assert end_time == math.floor(end_time)

    inter_send_time = 1 / pkts_per_ms
    cur_time = start_time + inter_send_time
    while (cur_time <= end_time):
        floor_cur_time = math.floor(cur_time)
        f.write(f"{floor_cur_time}\n")
        cur_time += inter_send_time


def burst(f, start_time, wait_time, burst_size, pace_time):
    pkts_per_ms = burst_size / pace_time
    smooth(f, start_time + wait_time, pkts_per_ms,
           start_time + wait_time + pace_time)


def aggregation_trace(seed, bw_ppms, ow_delay_ms, jitter_ms, duration, output):
    random.seed(seed)

    rtprop_ms = 2 * ow_delay_ms
    max_burst = bw_ppms * (jitter_ms + rtprop_ms)  # pkts

    def min_wait_time(burst_size):
        return (burst_size / bw_ppms) - rtprop_ms

    total_time = duration * 1e3  # ms

    with open(os.path.join(output), "w") as f:
        current_time = 0  # ms

        smooth(f, current_time, bw_ppms, 2 * rtprop_ms)
        current_time += 2 * rtprop_ms

        while current_time <= total_time:
            # Choose
            this_burst_size = max_burst
            wait_time = min_wait_time(this_burst_size)

            # Clamp
            this_burst_size = min(this_burst_size, max_burst)
            wait_time = max(wait_time, min_wait_time(this_burst_size))
            assert wait_time <= jitter_ms
            pace_time = rtprop_ms

            burst(f, current_time, wait_time, this_burst_size, pace_time)
            current_time = current_time + wait_time + pace_time


def fixed_aggregation(seed, bw_ppms, ow_delay_ms, jitter_ms, jitter_ppms, duration, output):
    random.seed(seed)

    assert jitter_ms * bw_ppms % jitter_ppms == 0

    rtprop_ms = 2 * ow_delay_ms
    max_burst = bw_ppms * jitter_ms
    pace_time = max_burst / jitter_ppms
    wait_time = jitter_ms - pace_time

    total_time = duration * 1e3  # ms

    with open(os.path.join(output), "w") as f:
        current_time = 0  # ms

        smooth(f, current_time, bw_ppms, 2 * rtprop_ms)
        current_time += 2 * rtprop_ms

        while current_time <= total_time:
            burst(f, current_time, wait_time, max_burst, pace_time)
            current_time = current_time + wait_time + pace_time


def all_trace(seed, bw_ppms, ow_delay_ms, jitter_ms, jitter_ppms, duration, output, jitter_type):
    if jitter_type == "aggregation":
        aggregation_trace(
            seed,
            bw_ppms,
            ow_delay_ms,
            jitter_ms,
            duration,
            output,
        )
    if jitter_type == "fixed_aggregation":
        fixed_aggregation(
            seed,
            bw_ppms,
            ow_delay_ms,
            jitter_ms,
            jitter_ppms,
            duration,
            output,
        )
    else:
        raise ValueError(f"Unknown jitter type: {jitter_type}")


def main():
    parser = argparse.ArgumentParser(description="Generate a trace with jitter.")
    parser.add_argument("seed", type=int, default=42, help="Seed.")
    parser.add_argument("bw_ppms", type=int, help="Link speed in packets per ms.")
    parser.add_argument("jitter_ppms", type=int, help="Burst rate for jitter box.")
    parser.add_argument("ow_delay_ms", type=int, help="One way network delay in ms.")
    parser.add_argument("jitter_ms", type=int, help="Maximum jitter in ms.")
    parser.add_argument("jitter_type", type=int, help="Type of jitter.")
    parser.add_argument("duration", type=int, help="Duration of the trace in seconds.")
    parser.add_argument("o", "output", type=str, help="Path to output file.")

    args = parser.parse_args()
    all_trace(
        args.seed,
        args.bw_ppms,
        args.ow_delay_ms,
        args.jitter_ms,
        args.jitter_ppms,
        args.duration,
        args.output,
        args.jitter_type,
    )


if __name__ == "__main__":
    main()
