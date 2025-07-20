import argparse
import json
import math
import os
import pprint
from collections import deque
from typing import Any, Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    CCA_RENAME,
    ENTRY_NUMBER,
    MM_PKT_SIZE,
    ONE_PKT_PER_MS_MBPS_RATE,
    PKT_TO_WIRE,
    S_TO_MS,
    WIRE,
    parse_exp_raw,
    plot_df,
    plot_multi_exp,
    set_output_dir,
    try_except_wrapper,
)

EVENT_NAME = {
    '#': 'LOST_OPPORTUNITY',
    '-': 'DEPARTURE',
    '+': 'ARRIVAL',
    'd': 'DROP',
}

PKT_SIZE = 1504


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)


def parse_exp(exp_tag):
    ret = parse_exp_raw(exp_tag)
    bdp_bytes = MM_PKT_SIZE * ret["bw_ppms"] * 2 * ret["ow_delay_ms"]
    buf_size_bytes = bdp_bytes * ret["buf_size_bdp"]
    ret["bdp_bytes"] = bdp_bytes
    ret["buf_size_bytes"] = int(buf_size_bytes)
    return ret


class MahimahiLog:

    fpath: str
    df: pd.DataFrame
    summary: Dict[str, Any] = {}

    arrival_df: pd.DataFrame
    drop_df: pd.DataFrame
    departure_df: pd.DataFrame
    queue_df: pd.DataFrame

    def __init__(self, fpath, summary_only=False):
        self.fpath = fpath
        self.exp = parse_exp(os.path.basename(fpath).removesuffix('.log'))
        ret = self.check_cache()
        if(not ret or not summary_only):
            self.parse_mahimahi_log()
            self.derive_metrics()
            self.derive_summary_metrics()
        if(not ret):
            self.write_cache()

    def check_cache(self):
        summary_path = self.fpath.removesuffix('.log') + '.summary'
        if (os.path.exists(summary_path)):
            with open(summary_path, 'r') as f:
                jdict = json.load(f)
            self.summary = jdict
            return True
        return False

    def write_cache(self):
        summary_path = self.fpath.removesuffix('.log') + '.summary'
        with open(summary_path, 'w') as f:
            json.dump(self.summary, f, indent=4, sort_keys=True,
                      separators=(', ', ': '), ensure_ascii=False,
                      cls=NumpyEncoder)
        # with open(summary_path, 'w') as f:
        #     json.dump(self.summary, f)

    def parse_mahimahi_log(self):
        with open(self.fpath, 'r') as f:
            lines = f.read().split('\n')

        base_timestamp = None
        queue_size_bytes = None
        records = []

        for line in lines:
            # Parse headers
            if(line.startswith('#')):
                if(line.startswith('# base timestamp: ')):
                    base_timestamp = int(line.removeprefix('# base timestamp: '))
                if(line.startswith('# queue: droptail [bytes=')):
                    queue_size_bytes = \
                            int(line
                            .removeprefix('# queue: droptail [bytes=')
                            .removesuffix(']'))

            elif(line in [' ', '']):
                pass

            else:
                # import ipdb; ipdb.set_trace()
                splits = line.split(' ')
                event = EVENT_NAME[splits[1]]
                assert base_timestamp is not None
                record = {
                    'time': int(splits[0])-base_timestamp,
                    'event': event
                }

                if(event in ['LOST_OPPORTUNITY', 'ARRIVAL', 'DEPARTURE']):
                    record['bytes'] = int(splits[2])
                elif(event == 'DROP'):
                    # record['packets'] = int(splits[2])
                    record['bytes'] = int(splits[3])

                if(event == 'DEPARTURE'):
                    record['ow_delay_ms'] = int(splits[3])

                records.append(record)

        df = pd.DataFrame(records)
        assert base_timestamp is not None
        assert queue_size_bytes is not None

        self.df = df
        start = df['time'].min()
        end = df['time'].max()
        self.summary.update({
            'base_timestamp': base_timestamp,
            'queue_size_bytes': queue_size_bytes,
            'mm_trace_time_ms': end-start,
        })

    def compute_queueing_delay(self):
        """
        Computes queueing delay for each packet based on enqueue (ARRIVAL) and
        dequeue (DEPARTURE) events. The difference between dequeue time and
        enqueue time gives the queueing delay.
        """
        enqueue_times = deque()
        queueing_delay_records = []

        for _, row in self.df.iterrows():
            if row["event"] == "ARRIVAL":
                enqueue_times.append(row["time"])
            elif row["event"] == "DEPARTURE" and enqueue_times:
                enqueue_time = enqueue_times.popleft()
                queueing_delay = row["time"] - enqueue_time

                queueing_delay_records.append(
                    {
                        "dequeue_time_ms": row["time"],
                        "queueing_delay_ms": queueing_delay,
                    }
                )

        self.queueing_delay_df = pd.DataFrame(queueing_delay_records)

    def plot_queueing_delay(self, output_dir):

        if not hasattr(self, "queueing_delay_df"):
            self.compute_queueing_delay()

        if self.queueing_delay_df.empty:
            print("No queueing delay data available to plot.")
            return

        os.makedirs(output_dir, exist_ok=True)
        plot_df(
            self.queueing_delay_df,
            "queueing_delay_ms",
            os.path.join(output_dir, "dequeue_vs_queueing_delay.pdf"),
            xkey="dequeue_time_ms",
            xlabel="Dequeue Time (ms)",
            ylabel="Queueing Delay (ms)",
        )

    def derive_metrics(self):
        df = self.df

        start = df['time'].min()
        end = df['time'].max()

        arrival_df = (df[df['event'] == 'ARRIVAL']
                      .groupby('time')
                      .sum(numeric_only=True)
                      .reindex(range(start, end+1))
                      .fillna(0))
        arrival_df['cum_bytes'] = arrival_df['bytes'].cumsum()

        drop_df = (df[df['event'] == 'DROP']
                   .groupby('time')
                   .sum(numeric_only=True)
                   .reindex(range(start, end+1))
                   .fillna(0))
        drop_df['cum_bytes'] = drop_df['bytes'].cumsum()

        departure_df = (df[df['event'] == 'DEPARTURE']
                        .groupby('time')
                        .sum(numeric_only=True)
                        .reindex(range(start, end+1))
                        .fillna(0))
        departure_df['cum_bytes'] = departure_df['bytes'].cumsum()

        queue_df = (arrival_df - departure_df - drop_df).reset_index()
        queue_df['utilization'] = (100 * queue_df['cum_bytes'] /
                                   self.summary['queue_size_bytes'])

        self.arrival_df = arrival_df.reset_index()
        self.drop_df = drop_df.reset_index()
        self.departure_df = departure_df.reset_index()
        self.queue_df = queue_df

        # self.compute_queueing_delay()
        # qdf = self.queueing_delay_df
        # self.qdf = qdf.groupby("dequeue_time_ms").mean().reset_index()

    def derive_summary_metrics(self):
        delivered_bytes = self.departure_df['cum_bytes'].iloc[-1]
        lost_bytes = self.drop_df['cum_bytes'].iloc[-1]
        lost_pkts = lost_bytes / PKT_SIZE
        Rm = self.exp['ow_delay_ms'] * 2
        num_Rms = self.summary['mm_trace_time_ms'] / Rm
        bdp_pkts = self.exp['ow_delay_ms'] * 2 * self.exp['bw_ppms']  # pkts
        bdp_bytes = bdp_pkts * PKT_SIZE
        throughput_mbps = ((delivered_bytes * 8 * S_TO_MS / self.summary['mm_trace_time_ms']) / 1e6)
        throughput_mbps = throughput_mbps * PKT_TO_WIRE

        SLOW_START_END = math.ceil(10 * math.log(bdp_pkts, 2) * Rm)

        # Skip first 20 seconds of trace (SS)
        lost_bytes_except_ss = self.drop_df['cum_bytes'].iloc[-1] - self.drop_df['cum_bytes'].iloc[SLOW_START_END]
        lost_pkts_except_ss = lost_bytes_except_ss / PKT_SIZE
        num_Rms_except_ss = (self.summary['mm_trace_time_ms'] - SLOW_START_END) / Rm
        delivered_bytes_except_ss = self.departure_df['cum_bytes'].iloc[-1] - self.departure_df['cum_bytes'].iloc[SLOW_START_END]
        mm_throughput_mbps_except_ss = ((delivered_bytes_except_ss * 8 * S_TO_MS / (self.summary['mm_trace_time_ms'] - SLOW_START_END)) / 1e6)
        mm_throughput_mbps_except_ss = mm_throughput_mbps_except_ss * PKT_TO_WIRE
        mm_utilization_except_ss = mm_throughput_mbps_except_ss / (self.exp['bw_ppms'] * ONE_PKT_PER_MS_MBPS_RATE)
        # mm_delay = self.qdf['queueing_delay_ms'].mean()
        # mm_delay_except_ss = self.qdf['queueing_delay_ms'][self.qdf['dequeue_time_ms'] >= SLOW_START_END].mean()

        self.summary.update({
            'lost_bytes': lost_bytes,
            'lost_pkts_per_Rm': lost_pkts / num_Rms,
            'mm_throughput_mbps': throughput_mbps,
            'mm_queue_min': self.queue_df['cum_bytes'].min() / bdp_bytes,
            'mm_queue_max': self.queue_df['cum_bytes'].max() / bdp_bytes,
            'mm_queue_mean': self.queue_df['cum_bytes'].mean() / bdp_bytes,
            'mm_utilization': throughput_mbps / (self.exp['bw_ppms'] * ONE_PKT_PER_MS_MBPS_RATE),
            # 'mm_delay': mm_delay,

            'loss_probability_except_ss': lost_bytes_except_ss / delivered_bytes_except_ss,
            'lost_pkts_per_Rm_except_ss': lost_pkts_except_ss / num_Rms_except_ss,
            'mm_throughput_except_ss': mm_throughput_mbps_except_ss,
            'mm_queue_min_except_ss': self.queue_df['cum_bytes'].iloc[SLOW_START_END:].min() / bdp_bytes,
            'mm_queue_max_except_ss': self.queue_df['cum_bytes'].iloc[SLOW_START_END:].max() / bdp_bytes,
            'mm_queue_mean_except_ss': self.queue_df['cum_bytes'].iloc[SLOW_START_END:].mean() / bdp_bytes,
            'mm_qdel_mean_except_ss': self.queue_df['cum_bytes'].iloc[SLOW_START_END:].mean() * Rm / bdp_bytes,
            'mm_throughput_mbps_except_ss': mm_throughput_mbps_except_ss,
            'mm_utilization_except_ss': mm_utilization_except_ss,
            # 'mm_delay_except_ss': mm_delay_except_ss,
        })


def plot_single_exp(fpath, output_dir):

    ml = MahimahiLog(fpath)

    # import ipdb; ipdb.set_trace()
    arrival_df = ml.arrival_df
    departure_df = ml.departure_df
    queue_df = ml.queue_df
    loss_df = ml.drop_df

    arrival_df['cum_packets'] = arrival_df['cum_bytes'] / PKT_SIZE
    departure_df['cum_packets'] = departure_df['cum_bytes'] / PKT_SIZE
    loss_df['cum_packets'] = loss_df['cum_bytes'] / PKT_SIZE
    arrival_df['packets'] = arrival_df['bytes'] / PKT_SIZE
    departure_df['packets'] = departure_df['bytes'] / PKT_SIZE
    loss_df['packets'] = loss_df['bytes'] / PKT_SIZE

    # import ipdb; ipdb.set_trace()
    win = int(ml.exp['ow_delay_ms'] * 2)
    # arrival_df_windowed = arrival_df.groupby(arrival_df['time'] // win).aggregate({
    #     'time': 'min', 'bytes': 'sum', 'cum_bytes': 'max',
    #     'cum_packets': 'max', 'packets': 'sum'})
    # departure_df_windowed = departure_df.groupby(departure_df['time'] // win).aggregate({
    #     'time': 'min', 'bytes': 'sum', 'cum_bytes': 'max',
    #     'cum_packets': 'max', 'packets': 'sum'})
    # loss_df_windowed = loss_df.groupby(loss_df['time'] // win).aggregate({
    #     'time': 'min', 'bytes': 'sum', 'cum_bytes': 'max',
    #     'cum_packets': 'max', 'packets': 'sum'})
    # arrival_df_windowed['pps'] = arrival_df_windowed['packets'] * S_TO_MS / win
    # departure_df_windowed['pps'] = arrival_df_windowed['packets'] * S_TO_MS / win

    arrival_df_rolling = arrival_df.rolling(win).aggregate({
            'time': 'max', 'bytes': 'sum', 'cum_bytes': 'max',
            'cum_packets': 'max', 'packets': 'sum'})[win:]
    departure_df_rolling = departure_df.rolling(win).aggregate({
            'time': 'max', 'bytes': 'sum', 'cum_bytes': 'max',
            'cum_packets': 'max', 'packets': 'sum'})[win:]
    loss_df_rolling = loss_df.rolling(win).aggregate({
            'time': 'max', 'bytes': 'sum', 'cum_bytes': 'max',
            'cum_packets': 'max', 'packets': 'sum'})[win:]
    arrival_df_rolling['pps'] = arrival_df_rolling['packets'] * S_TO_MS / win
    arrival_df_rolling['mbps'] = arrival_df_rolling['pps'] * ONE_PKT_PER_MS_MBPS_RATE / S_TO_MS
    arrival_df_rolling['secs'] = arrival_df_rolling['time'] / S_TO_MS
    departure_df_rolling['pps'] = departure_df_rolling['packets'] * S_TO_MS / win
    departure_df_rolling['mbps'] = departure_df_rolling['pps'] * ONE_PKT_PER_MS_MBPS_RATE / S_TO_MS
    departure_df_rolling['secs'] = departure_df_rolling['time'] / S_TO_MS

    # For smoothed throughput over time.
    agg_win = 10 * win
    agg_win = 1000
    departure_df_rolling_agg = departure_df.rolling(agg_win).aggregate({
        'time': 'max', 'bytes': 'sum', 'cum_bytes': 'max',
        'cum_packets': 'max', 'packets': 'sum'})[agg_win:]
    departure_df_rolling_agg['pps'] = departure_df_rolling_agg['packets'] * S_TO_MS / agg_win

    os.makedirs(output_dir, exist_ok=True)
    # plot_df(arrival_df_windowed,
    #         'pps', os.path.join(output_dir, 'mm_sending_rate_windowed.pdf'),
    #         xlabel='Time (ms)', ylabel='Packets/sec (arrival)')
    # plot_df(departure_df_windowed,
    #         'pps', os.path.join(output_dir, 'mm_ack_rate_windowed.pdf'),
    #         xlabel='Time (ms)', ylabel='Packets/sec (departure)')
    # plot_df(loss_df_windowed,
    #         'packets', os.path.join(output_dir, 'mm_loss_in_Rm_windowed.pdf'),
    #         xlabel='Time (ms)', ylabel=f'# Pkts lost in 1 Rm. Rm={win}ms')

    plot_df(arrival_df_rolling,
            'mbps', os.path.join(output_dir, 'mm_sending_rate_rolling.pdf'),
            xkey="secs", xlabel='Time (s)', ylabel='Mbps (arrival)')
    plot_df(departure_df_rolling,
            'mbps', os.path.join(output_dir, 'mm_ack_rate_rolling.pdf'),
            xkey="secs", xlabel='Time (s)', ylabel='Mbps (departure)')
    plot_df(loss_df_rolling,
            'packets', os.path.join(output_dir, 'mm_loss_in_Rm_rolling.pdf'),
            xlabel='Time (ms)', ylabel=f'# Pkts lost in 1 Rm. Rm={win}ms')
    plot_df(departure_df_rolling_agg,
            'pps', os.path.join(output_dir, 'mm_smoothed_ack_rate_rolling.pdf'),
            xlabel='Time (ms)', ylabel='Packets/sec')
    # plot_df(arrival_df,
    #         'cum_packets', os.path.join(output_dir, 'mm_arrival.pdf'),
    #         xlabel='Time (ms)', ylabel='Packets (arrival)')
    # plot_df(departure_df,
    #         'cum_packets', os.path.join(output_dir, 'mm_departure.pdf'),
    #         xlabel='Time (ms)', ylabel='Packets (departure)')
    plot_df(queue_df,
            'utilization', os.path.join(output_dir, 'mm_queue.pdf'),
            xlabel='Time (ms)', ylabel='Queue (%)')
    # plot_df(loss_df,
    #         'cum_packets', os.path.join(output_dir, 'mm_loss.pdf'),
    #         xlabel='Time (ms)', ylabel='Packets (loss)')

    # pprint.pprint(ml.summary)
    record = [
        ml.exp["n_flows"],
        ml.summary["mm_throughput_except_ss"],
        ml.summary["mm_qdel_mean_except_ss"],
        # ml.summary["mm_delay_except_ss"],
        ml.summary["loss_probability_except_ss"],
    ]
    print(
        ", ".join([str(x) for x in record])
    )

    # Subsection of the trace.
    # trace_end_ms = queue_df['time'].max()
    # start_ms = trace_end_ms - 5000
    # end_ms = start_ms + 500

    # plot_df(arrival_df[np.logical_and(arrival_df['time'] >= start_ms, arrival_df['time'] <= end_ms)],
    #         'packets', os.path.join(output_dir, 'arrival.pdf'),
    #         xlabel='Time (ms)', ylabel='Packets (arrival)')

    # plot_df(departure_df[np.logical_and(departure_df['time'] >= start_ms, departure_df['time'] <= end_ms)],
    #         'packets', os.path.join(output_dir, 'departure.pdf'),
    #         xlabel='Time (ms)', ylabel='Packets (departure)')

    # plot_df(queue_df[np.logical_and(queue_df['time'] >= start_ms, queue_df['time'] <= end_ms)],
    #         'utilization', os.path.join(output_dir, 'queue.pdf'),
    #         xlabel='Time (ms)', ylabel='Queue (%)')
    # pprint.pprint(ml.summary)


@try_except_wrapper
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', required=True,
        type=str, action='store',
        help='path to mahimahi trace')
    parser.add_argument(
        '-o', '--output', default="",
        type=str, action='store',
        help='path output figure')
    args = parser.parse_args()
    set_output_dir(args)

    if(os.path.isdir(args.input)):
        plot_multi_exp(args.input, args.output, '.log', plot_single_exp)
    else:
        plot_single_exp(args.input, args.output)


if(__name__ == "__main__"):
    main()
