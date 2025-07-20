import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    ONE_PKT_PER_MS_MBPS_RATE,
    S_TO_MS,
    parse_exp_raw,
    plot_df,
    set_output_dir,
    try_except_wrapper,
)


def parse_jdict(fpath):
    with open(fpath, 'r') as f:
        try:
            jdict = json.load(f)
        except json.decoder.JSONDecodeError as e:
            print(f"ERROR: json decode error for file: {fpath}")
            print(e)
            raise e
    return jdict


def parse_iperf_summary(fpath):
    exp = parse_exp_raw(os.path.basename(fpath).removesuffix('.json'))
    jdict = parse_jdict(fpath)

    ret = {
        'min_rtt': jdict['end']['streams'][0]['sender']['min_rtt'],
        'max_rtt': jdict['end']['streams'][0]['sender']['max_rtt'],
        'mean_rtt': jdict['end']['streams'][0]['sender']['mean_rtt'],
        'bits_per_second': jdict['end']['streams'][0]['receiver'][
            'bits_per_second'
        ],
        'retransmits': jdict['end']['streams'][0]['sender']['retransmits'],
        'time_seconds': jdict['end']['streams'][0]['sender']['seconds'],
    }
    ret['mbps'] = ret['bits_per_second'] / 1e6
    Rm = exp['delay'] * 2
    rate = exp['rate']
    num_Rms = ret['time_seconds'] * S_TO_MS / Rm
    ret['retransmits_per_Rm'] = ret['retransmits'] / num_Rms
    ret['utilization'] = ret['bits_per_second'] / (ONE_PKT_PER_MS_MBPS_RATE * rate * 1e6)
    return ret


def parse_iperf_timeseries(fpath):
    jdict = parse_jdict(fpath)

    start_time = jdict['start']['timestamp']['timesecs']

    records = [
        {
            'start': record['streams'][0]['start'],
            'end': record['streams'][0]['end'],
            'seconds': record['streams'][0]['seconds'],
            'bits_per_second': record['streams'][0][
                'bits_per_second'
            ],
            'retransmits': record['streams'][0]['retransmits'],
            'rtt': record['streams'][0]['rtt'],
        }
        for record in jdict['intervals']
    ]
    df = pd.DataFrame(records)
    df["start"] = start_time + df["start"]
    df["end"] = start_time + df["end"]
    df['mbps'] = df['bits_per_second'] / 1e6
    df['interval'] = df['end'] - df['start']
    df = df.sort_values(by='start').astype(np.float64)

    time = pd.to_datetime(df["start"], unit="s")
    df["time"] = time
    df.set_index('time', inplace=True)

    return df


def plot_single_exp(input_file, output_dir):
    df = parse_iperf_timeseries(input_file)

    os.makedirs(output_dir, exist_ok=True)
    plot_df(df, 'retransmits',
            os.path.join(output_dir, 'iperf_retransmits.png'),
            xkey='end', xlabel='Time (s)',
            ylabel='# Retransmits',
            title=os.path.basename(input_file))
    plot_df(df, 'mbps',
            os.path.join(output_dir, 'iperf_throughput.png'),
            xkey='end', xlabel='Time (s)',
            ylabel='Throughput (Mbps)',
            title=os.path.basename(input_file),
            ylim=(0, None))

    # summary = parse_iperf_summary(input_file)
    # print(input_file)
    # print(summary)


def get_dir_without_flow(fpath, ext):
    base = os.path.basename(fpath)
    fname = base.removesuffix(ext)
    # just assume that the first entry is flow
    # remove flow[num]- prefix where num is a number
    dname = fname
    if fname.startswith('flow['):
        dname = fname.split('-', 1)[1]
    return os.path.join(os.path.dirname(fpath), dname)


def plot_multi_exp(input_dir, output_dir, ext=".json"):
    exp_dirs = defaultdict(list)
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if (filename.endswith(ext)):
                fpath = os.path.join(root, filename)
                exp = parse_exp_raw(os.path.basename(fpath).removesuffix(ext))
                exp_dir = get_dir_without_flow(fpath, ext)
                exp_dirs[exp_dir].append(fpath)

    for exp_dir, files in exp_dirs.items():
        rel_path = os.path.relpath(exp_dir, input_dir)
        this_out_dir = os.path.join(output_dir, rel_path)
        os.makedirs(this_out_dir, exist_ok=True)

        df_list = []
        for file in files:
            exp = parse_exp_raw(os.path.basename(file).removesuffix(ext))
            df = parse_iperf_timeseries(file)
            if "flow" in exp:
                df["flow"] = exp["flow"]
            else:
                df["flow"] = 1
            df_list.append(df)
        df = pd.concat(df_list, axis=0).sort_values(by='start')
        df["rtt_ms"] = df["rtt"] / 1e3

        plot_df(
            df,
            "mbps",
            os.path.join(this_out_dir, "iperf_throughput.png"),
            xkey="end",
            xlabel="Time (s)",
            ylabel="Throughput (Mbps)",
            group="flow",
        )
        plot_df(
            df,
            "rtt_ms",
            os.path.join(this_out_dir, "iperf_rtt.png"),
            xkey="end",
            xlabel="Time (s)",
            ylabel="RTT (ms)",
            group="flow",
        )


def plot_parking_lot(input_dir, output_dir):
    input_files = []
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if (filename.endswith('.json')):
                input_files.append(os.path.join(root, filename))
    input_files = sorted(input_files)

    dfs = [parse_iperf_timeseries(f) for f in input_files]
    for df in dfs:
        df['mbps'] = df['bits_per_second'] / 1e6

    exp = parse_exp_raw(os.path.basename(input_files[0]).removesuffix('.json'))
    print_info = [str(int(exp["rate"]) * 12)]

    os.makedirs(output_dir, exist_ok=True)
    n = len(dfs)
    rates = []
    fig, ax = plt.subplots(n, 1, figsize=(6.4, 4.8*1.5), sharex=True, sharey=True)
    for i, df in enumerate(dfs):
        exp = parse_exp_raw(os.path.basename(input_files[i]).removesuffix('.json'))
        summary = parse_iperf_summary(input_files[i])
        title = "flow_id={}, throughput={:.2f} mbps".format(int(exp['port']), summary['mbps'])
        ax[i].set_title(title)
        ax[i].plot(df['end'], df['mbps'])
        ax[i].set_ylabel('Throughput (Mbps)')
        ax[i].grid(True)

        print_info.append("{:.2f}".format(summary['mbps']))
        rates.append(summary['mbps'])

    ax[-1].set_xlabel('Time (s)')
    fig.set_layout_engine("tight")
    fig.savefig(os.path.join(output_dir, 'iperf_parking_lot.png'))
    print_info.append("{:.2f}".format(max(rates) / min(rates)))
    print("\t".join(print_info))


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
    parser.add_argument(
        '--parking-lot', required=False,
        action='store_true',
        help='plot parking lot')
    args = parser.parse_args()
    set_output_dir(args)

    if(os.path.isdir(args.input)):
        plot_multi_exp(args.input, args.output)
        if args.parking_lot:
            plot_parking_lot(args.input, args.output)
    else:
        plot_single_exp(args.input, args.output)


if(__name__ == "__main__"):
    main()
