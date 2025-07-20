import argparse
import os
from collections import defaultdict
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    CCA_RENAME,
    ENTRY_NUMBER,
    parse_literal,
    plot_df,
    set_output_dir,
    try_except_wrapper,
)
from plot_config_light import get_fig_size_paper, get_fig_size_ppt, get_style, get_fig_size_acm_small

ppt = True
ppt = False
style = get_style(use_markers=False, paper=True, use_tex=False)  # paper
get_fig_size = get_fig_size_paper
ext = "pdf"
if ppt:
    style = get_style(use_markers=False, paper=False, use_tex=False)  # ppt
    get_fig_size = get_fig_size_ppt
    ext = "svg"
figsize = get_fig_size()

def parse_tag(tag):
    ret = {}
    """
    [n_flows=1][bw_mbps=100][delay_ms=50][queue_size_bdp=1][cca=vegas]
    """
    for kv in tag.split("]["):
        kv = kv.replace("[", "").replace("]", "")
        key, value = kv.split("=")
        ret[key] = parse_literal(value)
    return ret


def parse_tc_df(input_file: str):
    """
    Header:
    time,bytes,packets,drops,overlimits,requeues,backlog,qlen
    """
    df = pd.read_csv(input_file)

    # window metrics
    # bucket = '100'
    # df['time'] = pd.to_datetime(df['time'], unit='s')
    # df.set_index('time', inplace=True)
    # ddf = df.resample(bucket).last().dropna()
    # ddf.reset_index(inplace=True)
    ddf = df

    ddf["bytes_diff"] = ddf["bytes"].diff()
    ddf["packets_diff"] = ddf["packets"].diff()
    ddf["drops_diff"] = ddf["drops"].diff()
    ddf["time_diff"] = ddf["time"].diff()

    ddf["send_rate_mbps"] = ddf["bytes_diff"] / ddf["time_diff"] * 8 / 1e6
    ddf["loss_prob"] = ddf["drops_diff"] / ddf["packets_diff"]

    return df, ddf


def get_steady_state_bps(df: pd.DataFrame):
    """
    Header is cumulative values:
    time,bytes,packets,drops,overlimits,requeues,backlog,qlen
    """
    # Get average throughput in seconds 240 to 300
    startf, endf = 0.5, 0.9
    startf, endf = 0, 1
    n = len(df)
    start = int(n * startf)
    end = int(n * endf)
    fdf = df.iloc[start:end]
    send_bytes = fdf["bytes"].iloc[-1] - fdf["bytes"].iloc[0]
    send_time = fdf["time"].iloc[-1] - fdf["time"].iloc[0]
    return send_bytes * 8 / send_time


def parse_tc_summary(input_file):
    df, ddf = parse_tc_df(input_file)
    ss_bps = get_steady_state_bps(df)
    return {
        "ss_bps": ss_bps,
        "ss_mbps": ss_bps / 1e6,
    }


@matplotlib.rc_context(rc=style)
def plot_single_exp_combined(exp_dir: str, input_files: list, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df_list = []
    for input_file in input_files:
        df, ddf = parse_tc_df(input_file)
        fname = os.path.basename(input_file).replace(".csv", "")
        if "switch" in fname:
            continue
        else:
            ftags = parse_tag(fname)
            rid = int(ftags["receiver"].removeprefix("hr"))
            ddf["rid"] = f"$f_{rid}$"
            df_list.append(ddf)
    df = pd.concat(df_list, ignore_index=True)
    df.sort_values(by=["rid", "time"], inplace=True)
    figsize = get_fig_size(0.24, 0.32)
    plot_df(
        df,
        "send_rate_mbps",
        os.path.join(output_dir, f"rate-combined.{ext}"),
        group="rid",
        xkey="time",
        xlabel="Time (s)",
        ylabel="Tput (Mbps)",
        figsize=figsize,
        legend_ncol=2,
    )


def plot_multi_exp(input_dir: str, output_dir: str,
                   ext: str, plot_single_exp: Callable):
    experiments = defaultdict(list)
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if (filename.endswith(ext)):
                fpath = os.path.join(root, filename)
                exp_dir = os.path.dirname(fpath)
                experiments[exp_dir].append(fpath)
                # dirpath = fpath.replace(ext, '')
                dirpath = os.path.dirname(fpath)
                rel_path = os.path.relpath(dirpath, input_dir)
                this_out_dir = os.path.join(output_dir, rel_path)
                plot_single_exp(fpath, this_out_dir)

    for exp_dir, input_files in experiments.items():
        rel_path = os.path.relpath(exp_dir, input_dir)
        this_out_dir = os.path.join(output_dir, rel_path)
        plot_single_exp_combined(exp_dir, input_files, this_out_dir)


def summarize_multi_exp(indirs: list, outdir: str, ext: str):
    experiments = defaultdict(list)
    for input_dir in indirs:
        for root, _, files in os.walk(input_dir):
            for filename in files:
                if (filename.endswith(ext)):
                    fpath = os.path.join(root, filename)
                    exp_dir = os.path.dirname(fpath)
                    experiments[exp_dir].append(fpath)

    samples = []
    for exp_dir in experiments.keys():
        print(f"Processing {exp_dir}")
        exp = os.path.basename(exp_dir)
        params = parse_tag(exp)
        if params["cca"] != "astraea":
            continue

        rates = {}
        delays = {}
        # import ipdb; ipdb.set_trace()
        for input_file in experiments[exp_dir]:
            fname = os.path.basename(input_file).replace(".csv", "")
            ftags = parse_tag(fname)

            df, ddf = parse_tc_df(input_file)
            if "switch" in ftags:
                qocc = df["backlog"].mean()
                delay_ms = qocc * 8 / (params["bw_mbps"] * 1e3)
                sid = int(ftags["switch"].removeprefix("s"))
                delays[sid] = delay_ms
            elif "receiver" in ftags:
                rid = int(ftags["receiver"].removeprefix("hr"))
                rates[rid] = ddf["send_rate_mbps"].mean()

        assert len(rates) > 0
        assert len(delays) > 0
        n = len(rates)

        topo = None
        if "n_flows" in params:
            assert len(delays) == 1
            assert len(rates) == params["n_flows"]
            # dumbbell
            topo = "dumbbell"
            for i in range(n):
                sample = {
                    "rate": rates[i],
                    "delay": delays[0],
                    "topology": "dumbbell",
                    "hops": 1,
                }
                sample.update(params)
                samples.append(sample)


        elif "hops" in params:
            assert len(delays) == params["hops"]
            assert len(rates) == params["hops"] + 1
            # parking lot
            topo = "parking_lot"
            for i in range(1, n):
                sample = {
                    "rate": rates[i],
                    "delay": delays[i-1],
                    "topology": "parking_lot",
                    "n_flows": params["hops"] + 1,
                }
                sample.update(params)
                samples.append(sample)

            # long flow
            sample = {
                "rate": rates[0],
                "delay": np.sum(list(delays.values())),
                "topology": "parking_lot",
                "n_flows": params["hops"] + 1,
            }
            sample.update(params)
            samples.append(sample)

        else:
            raise NotImplementedError("Could not identify as dumbbell or parking lot")

        if topo == "dumbbell":
            tput_ratio = max(rates.values()) / min(rates.values())
            print(tput_ratio, params)

    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame(samples)
    df.to_csv(os.path.join(outdir, "rate-delay-samples.csv"), index=False)

    fig, ax = plt.subplots()

    ax.scatter(df["delay"], df["rate"], marker="X")
    ax.set_ylabel("Rate (Mbps)")
    ax.set_xlabel("Delay (ms)")
    ax.grid(True)

    fig.set_layout_engine("tight", pad=0.03)
    fig.savefig(os.path.join(outdir, "rate-delay-samples.pdf"))



@matplotlib.rc_context(rc=style)
def plot_single_exp(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fname = os.path.basename(input_file).replace(".csv", "")
    df, ddf = parse_tc_df(input_file)

    if "switch" in fname:
        plot_df(
            df,
            "backlog",
            os.path.join(output_dir, f"{fname}-queue.{ext}"),
            xkey="time",
            xlabel="Time (s)",
            ylabel="Queue [bytes]",
        )
    else:
        plot_df(
            ddf,
            "send_rate_mbps",
            os.path.join(output_dir, f"{fname}-send-rate.{ext}"),
            xkey="time",
            xlabel="Time (s)",
            ylabel="Sent Rate [Mbps]",
        )

    # plot_df(
    #     df, 'bytes', os.path.join(output_dir, f'{fname}-sent.{ext}'),
    #     xkey='time', xlabel='Time (s)', ylabel='Sent [bytes]',
    # )
    # plot_df(
    #     df, 'drops', os.path.join(output_dir, f'{fname}-loss.{ext}'),
    #     xkey='time', xlabel='Time (s)', ylabel='Drops [pkts]',
    # )
    # plot_df(
    #     ddf, 'loss_prob', os.path.join(output_dir, f'{fname}-loss-prob.{ext}'),
    #     xkey='time', xlabel='Time (s)', ylabel='Loss Probability',
    # )
    # plot_df(
    #     ddf, 'drops_diff', os.path.join(output_dir, f'{fname}-loss.{ext}'),
    #     xkey='time', xlabel='Time (s)', ylabel='Drops [pkts]',
    # )


@matplotlib.rc_context(rc=style)
def summarize_parking_lot(input_dir, output_dir):
    # find all experiment directories
    # parent of all csv files are experiment directories

    exp_dirs = {}
    for root, _, files in os.walk(input_dir):
        if any([f.endswith(".csv") for f in files]):
            exp_dirs[root] = sorted([f for f in files if f.endswith(".csv")])

    master_records = []
    for exp, files in exp_dirs.items():
        records = []
        tags_exp = parse_tag(os.path.basename(exp))
        for file in files:
            fpath = os.path.join(exp, file)
            record = {}
            summary = parse_tc_summary(fpath)
            record.update(summary)
            tags_flow = parse_tag(file.removesuffix(".csv"))
            record.update(tags_flow)
            records.append(record)

        df = pd.DataFrame(records)
        rdf = df[~df["receiver"].isna()].copy()
        rdf["rid"] = rdf["receiver"].apply(lambda x: x.removeprefix("hr")).astype(int)
        rdf = rdf.sort_values(by="rid")
        ratio = rdf["ss_mbps"].iloc[-1] / rdf["ss_mbps"].iloc[0]
        master_record = {
            "ratio": ratio,
        }
        master_record.update(tags_exp)
        master_records.append(master_record)

    mdf = pd.DataFrame(master_records)
    mdf["label"] = "_" + mdf["cca"].replace(CCA_RENAME)

    hmin = mdf["hops"].min()
    hmax = mdf["hops"].max()
    recs = []
    for i in range(hmin, hmax + 1):
        record = {
            "hops": i,
            "ratio": i,
            "label": "$\\texttt{hops}$",
            "cca": "$\\texttt{hops}$",
        }
        recs.append(record)
        record = {
            "hops": i,
            "ratio": i**2,
            "label": "$\\texttt{hops}^2$",
            "cca": "$\\texttt{hops}^2$",
        }
        recs.append(record)
    extra = pd.DataFrame(recs)
    mdf = pd.concat([mdf, extra], ignore_index=True)

    mdf["entry_number"] = mdf["label"].replace(ENTRY_NUMBER).infer_objects(copy=False)
    mdf = mdf.sort_values(by=["bw_mbps", "delay_ms", "entry_number", "cca", "hops"])
    print(mdf)

    figsize = get_fig_size(1, 0.6)
    opath = os.path.join(output_dir, f"parking-lot-summary.{ext}")
    plot_df(
        mdf,
        "ratio",
        opath,
        "hops",
        "Hops",
        "Throughput ratio",
        yscale="log",
        group="label",
        figsize=figsize,
        use_markers=True,
        use_entry=True,
    )


@try_except_wrapper
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        action="extend",
        nargs="+",
        help="path to mahimahi trace",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        action="store",
        help="path output figure",
    )
    parser.add_argument(
        "--parking-lot", required=False, action="store_true", help="plot parking lot"
    )
    args = parser.parse_args()
    if len(args.input) > 1:
        summarize_multi_exp(args.input, args.output, ".csv")
        return
    else:
        args.input = args.input[0]

    set_output_dir(args)

    if os.path.isdir(args.input):
        plot_multi_exp(args.input, args.output, ".csv", plot_single_exp)
        if args.parking_lot:
            summarize_parking_lot(args.input, args.output)
    else:
        plot_single_exp(args.input, args.output)


if __name__ == "__main__":
    main()
