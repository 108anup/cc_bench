import argparse
import io
import multiprocessing as mp
import multiprocessing.dummy
import os
import shlex
import subprocess
from collections import defaultdict
from functools import partial

import matplotlib
import numpy as np
import pandas as pd

from common import (
    CCA_RENAME,
    ENTRY_NUMBER,
    FIGS_PATH,
    LOGS_PATH,
    ONE_PKT_PER_MS_MBPS_RATE,
    TCPFlags,
    parse_exp_raw,
    plot_df,
    set_output_dir,
    try_except_wrapper,
)
from parse_mahimahi import PKT_SIZE
from plot_config_light import get_fig_size_paper, get_fig_size_ppt, get_style

pd.set_option('future.no_silent_downcasting', True)

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


def get_extension(filename):
    assert filename.endswith(("pcap", "pcap.csv"))
    ret = ".pcap"
    if filename.endswith(".pcap.csv"):
        ret = ".pcap.csv"
    return ret


def parse_pcap(fpath):
    ext = get_extension(fpath)
    exp = parse_exp_raw(os.path.basename(fpath).removesuffix(ext))
    if "cca_param_tag" not in exp:
        exp["cca_param_tag"] = ""
    flow = exp.get("flow", 1)

    fpath_or_buffer = fpath
    if ext == ".pcap":
        cmd = (f"tshark -n -T fields -E separator=, -e frame.time_epoch "
            f"-e tcp.flags -e tcp.srcport "
            f"-e tcp.analysis.ack_rtt -e frame.len "
            f"-e tcp.seq -e tcp.ack "
            f"-r {fpath} ")
        cmd = shlex.split(cmd)
        ret = subprocess.run(cmd, capture_output=True)
        fpath_or_buffer = io.StringIO(ret.stdout.decode("utf-8"))

    df = pd.read_csv(
        fpath_or_buffer,
        header=None,
        names=["time_epoch", "flags", "srcport", "rtt", "length", "seq", "ack"],
        converters={'flags': partial(int, base=16)}
    )

    # identify rx srcport
    rx_srcport = None
    for row in df.itertuples():
        flags = row.flags
        assert isinstance(flags, int)
        if (flags & TCPFlags.SYNACK) == TCPFlags.SYNACK:  # SYN-ACK (sent from receiver)
            rx_srcport = row.srcport
            break

    if rx_srcport is None:
        if "genericcc" not in exp["cca"]:
            rx_srcport = df["srcport"].min()
        # ports = set(df["srcport"].unique())
        # assert len(ports) == 2, fpath
        # srcports = df[df["length"] == 1500]["srcport"].unique()
        # if len(srcports) == 1:
        #     tx_srcport = srcports[0]
        #     ports.remove(tx_srcport)
        #     rx_srcport = list(ports)[0]

    if rx_srcport is None:
        print("Warning: no rx_srcport found in", fpath)
        return None, None, None

    assert rx_srcport is not None

    df["flow"] = flow
    df["is_ack"] = df["srcport"] == rx_srcport
    df["time"] = pd.to_datetime(df["time_epoch"], unit="s")

    ack_df = df[df["is_ack"]].dropna().reset_index(drop=True)
    ack_df["rtt_ms"] = ack_df["rtt"] * 1e3  # s to ms
    tx_df = df[~df["is_ack"]].reset_index(drop=True)

    # delete all rows after getting FIN
    # fin_idx = ack_df[ack_df["flags"] & TCPFlags.FIN == TCPFlags.FIN].index
    # fin_idx = fin_idx[0] if len(fin_idx) > 0 else len(ack_df) - 1
    # ack_df = ack_df.loc[:fin_idx]
    # fin_idx = tx_df[tx_df["flags"] & TCPFlags.FIN == TCPFlags.FIN].index
    # fin_idx = fin_idx[0] if len(fin_idx) > 0 else len(tx_df) - 1
    # tx_df = tx_df.loc[:fin_idx]

    # delete all rows after the max seq in both directions
    max_idx = ack_df["ack"].idxmax()
    ack_df = ack_df.loc[:max_idx]
    max_idx = tx_df["seq"].idxmax()
    tx_df = tx_df.loc[:max_idx]

    ack_df.set_index("time", inplace=True)
    tx_df.set_index("time", inplace=True)

    # ideally we want to compute throughput over rtt long intervals. But since
    # we measure this on a middle hop, we do not have the correct rtt.
    tdf = compute_throughput(ack_df, interval_ms=1000, pkt_size=PKT_SIZE, is_ack=True)

    return ack_df, tx_df, tdf


def compute_throughput(df, interval_ms=100, pkt_size=None, is_ack=False):
    df["count"] = 1
    df_resampled = (
        df.drop(columns=["flags", "rtt"])
        .resample(f"{interval_ms}ms")
        .agg(
            {
                "time_epoch": "last",
                "srcport": "last",
                "length": "sum",
                "count": "sum",
                "flow": "last",
                "seq": "max",
                "ack": "max",
            }
        )
    )
    # if pkt_size is not None:
    #     df_resampled["length"] = df_resampled["count"] * pkt_size
    seq = "seq" if not is_ack else "ack"
    df_resampled["length"] = df_resampled[seq] - df_resampled[seq].shift()
    df_resampled["mbps"] = (df_resampled["length"] * 8) / 1e3 / interval_ms
    # import ipdb; ipdb.set_trace()
    return df_resampled


@matplotlib.rc_context(rc=style)
def plot_single_exp(input_file, output_dir):
    figsize = get_fig_size(0.7, 0.7)
    ack_df, tx_df, tdf = parse_pcap(input_file)
    if ack_df is None:
        return
    os.makedirs(output_dir, exist_ok=True)

    plot_df(
        ack_df,
        "rtt_ms",
        os.path.join(output_dir, f"tcpdump_rtt.{ext}"),
        xkey="time_epoch",
        xlabel="Time (s)",
        ylabel="RTT (ms)",
        title=os.path.basename(input_file),
        ylim=(0, None),
        figsize=figsize,
    )

    plot_df(
        tdf,
        "mbps",
        os.path.join(output_dir, f"tcpdump_throughput.{ext}"),
        xkey="time_epoch",
        xlabel="Time (s)",
        ylabel="Throughput (Mbps)",
        title=os.path.basename(input_file),
        ylim=(0, None),
        figsize=figsize,
    )


def get_dir_without_flow(fpath, ext):
    base = os.path.basename(fpath)
    fname = base.removesuffix(ext)
    # just assume that the first entry is flow
    # remove flow[num]- prefix where num is a number
    dname = fname
    if fname.startswith('flow['):
        dname = fname.split('-', 1)[1]
    return os.path.join(os.path.dirname(fpath), dname)


def get_jfi(x):
    """Compute JFI from a list of throughput values"""
    n = len(x)
    assert n > 0
    jfi = (np.sum(x))**2 / (n * np.sum(x**2))
    return jfi


def get_ss_df(df: pd.DataFrame, fpath: str, start=0.6, end=1):
    """Compute steady state df from a dataframe"""
    assert start < end
    n = len(df)
    assert n > 0, fpath
    start_idx = int(n * start)
    end_idx = int(n * end)
    ss_df = df.iloc[start_idx:end_idx]
    return ss_df


@matplotlib.rc_context(rc=style)
def process_exp_dir(input_dir, output_dir, exp_dir, files):
    # threadpool
    pool = multiprocessing.dummy.Pool()
    ret = pool.map(parse_pcap, files)
    # ret = [parse_pcap(f) for f in all_files]
    dfs = {files[i]: ret[i] for i in range(len(files))}
    pool.close()
    pool.join()

    rel_path = os.path.relpath(exp_dir, input_dir)
    this_out_dir = os.path.join(output_dir, rel_path)
    os.makedirs(this_out_dir, exist_ok=True)

    this_exp = parse_exp_raw(os.path.basename(exp_dir))
    this_exp["rtprop_ms"] = 2 * this_exp["ow_delay_ms"]
    this_exp["bw_mbps"] = ONE_PKT_PER_MS_MBPS_RATE * this_exp["bw_ppms"]
    ack_df_list = []
    tx_df_list = []
    tdf_list = []
    ss_xputs = []
    ss_rtts = []
    for i, file in enumerate(files):
        # ack_df, tx_df, tdf = parse_pcap(file)
        ack_df, tx_df, tdf = dfs[file]
        if ack_df is None:
            continue
        ack_df_list.append(ack_df)
        tx_df_list.append(tx_df)
        tdf_list.append(tdf)

    # compute steady state throughput and rtt
    max_start_time = max(tdf["time_epoch"].min() for tdf in tdf_list)
    min_end_time = min(tdf["time_epoch"].max() for tdf in tdf_list)
    # min_start_time = min(tdf["time_epoch"].min() for tdf in tdf_list)
    # rel_start = max_start_time - min_start_time
    # rel_end = min_end_time - min_start_time
    # print(f"Common region: {rel_start:.2f} - {rel_end:.2f} ({rel_end - rel_start:.2f})")
    for i in range(len(tdf_list)):
        this_tdf = tdf_list[i]
        this_tdf = this_tdf[(this_tdf["time_epoch"] >= max_start_time) &
                        (this_tdf["time_epoch"] <= min_end_time)]
        this_ack_df = ack_df_list[i]
        this_ack_df = this_ack_df[(this_ack_df["time_epoch"] >= max_start_time) &
            (this_ack_df["time_epoch"] <= min_end_time)]
        ss_tdf = get_ss_df(this_tdf, files[i])
        # ss_start = ss_tdf["time_epoch"].min() - min_start_time
        # ss_end = ss_tdf["time_epoch"].max() - min_start_time
        # print(f"Flow {i}: {ss_tdf['mbps'].mean():.2f} Mbps, ss_start: {ss_start:.2f}, ss_end: {ss_end:.2f}")
        ss_xputs.append(ss_tdf["mbps"].mean())
        ss_rtts.append(get_ss_df(this_ack_df["rtt_ms"], files[i]).mean())

    ack_df = pd.concat(ack_df_list, axis=0).sort_values(by='time')
    tx_df = pd.concat(tx_df_list, axis=0).sort_values(by='time')
    tdf = pd.concat(tdf_list, axis=0).sort_values(by='time')
    tdf["time_rel"] = tdf["time_epoch"] - tdf["time_epoch"].min()
    ack_df["time_rel"] = ack_df["time_epoch"] - ack_df["time_epoch"].min()

    this_exp["jfi"] = get_jfi(np.array(ss_xputs))
    this_exp["xput_ratio"] = max(ss_xputs) / min(ss_xputs)
    this_exp["rtt"] = np.mean(np.array(ss_rtts))

    fpath = files[0]
    this_ext = get_extension(fpath)
    ftag = os.path.basename(fpath).removesuffix(this_ext)
    exp = parse_exp_raw(ftag)
    label = CCA_RENAME[exp["cca"]]
    figsize = get_fig_size(1, 0.6)
    figsize = get_fig_size(0.49, 0.49)

    plot_df(
        tdf,
        "mbps",
        os.path.join(this_out_dir, f"tcpdump_throughput.{ext}"),
        xkey="time_rel",
        xlabel="Time (s)",
        ylabel="Tput (Mbps)",
        title=label,
        group="flow",
        figsize=figsize,
        legend=False,
    )
    plot_df(
        ack_df,
        "rtt_ms",
        os.path.join(this_out_dir, f"tcpdump_rtt.{ext}"),
        xkey="time_rel",
        xlabel="Time (s)",
        ylabel="RTT (ms)",
        title=label,
        group="flow",
        figsize=figsize,
        legend=False,
    )
    return this_exp


@matplotlib.rc_context(rc=style)
def plot_multi_exp(input_dir, output_dir, exts=(".pcap", ".pcap.csv"), agg=""):
    exp_dirs = defaultdict(list)
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(exts):
                fpath = os.path.join(root, filename)
                this_ext = get_extension(fpath)
                exp_dir = get_dir_without_flow(fpath, this_ext)
                exp_dirs[exp_dir].append(fpath)

    pool = mp.Pool()
    process_exp_dir_partial = partial(process_exp_dir, input_dir, output_dir)
    exps = pool.starmap(process_exp_dir_partial, exp_dirs.items())
    # exps = [process_exp_dir_partial(k, v) for k, v in  exp_dirs.items()]
    pool.close()
    pool.join()


    if agg != "":
        # plot exp summary, group by cca, agg
        agg_xlabel = {
            "n_flows": "Flow count",
            "bw_mbps": "Link capacity (Mbps)",
            "rtprop_ms": "RTprop (ms)",
            "rtprop_ratio": "RTprop ratio",
            "jitter_ms": "Jitter (ms)",
        }
        exp_df = pd.DataFrame(exps)
        exp_df = exp_df[
            ~(
                (exp_df["cca"] == "genericcc_markovian")
                & (
                    (exp_df["cca_param_tag"] == "cv")
                    | (exp_df["cca_param_tag"] == "+cv")
                )
            )
        ]
        exp_df["cca"] = exp_df["cca"] + exp_df["cca_param_tag"]
        exp_df["label"] = exp_df["cca"].replace(CCA_RENAME)
        exp_df["entry_number"] = exp_df["label"].replace(ENTRY_NUMBER).infer_objects(copy=False)
        exp_df = exp_df.sort_values(by=["entry_number", "cca", agg])
        # print(exp_df)
        # import ipdb; ipdb.set_trace()
        figsize = get_fig_size(0.49, 0.49)
        plot_df(
            exp_df,
            "jfi",
            os.path.join(output_dir, f"jfi.{ext}"),
            xkey=agg,
            xlabel=agg_xlabel[agg],
            ylabel="JFI",
            ylim=(0.5, 1.1),
            group="label",
            figsize=figsize,
            use_markers=True,
            legend_ncol=2,
            use_entry=True,
        )

        if agg == "rtprop_ratio":
            exp_df["label"] = "_" + exp_df["label"]
            rtprop_ratios = sorted(list(exp_df[agg].unique()))
            recs = []
            for x in rtprop_ratios:
                recs.append({
                    "rtprop_ratio": x,
                    "label": "RTprop ratio",
                    "xput_ratio": x,
                })
            exp_df = pd.concat([exp_df, pd.DataFrame(recs)], ignore_index=True)

        plot_df(
            exp_df,
            "xput_ratio",
            os.path.join(output_dir, f"xput_ratio.{ext}"),
            xkey=agg,
            xlabel=agg_xlabel[agg],
            ylabel="Tput ratio",
            group="label",
            figsize=figsize,
            use_markers=True,
            yscale="log",
            # legend=False,
            use_entry=True,
        )
        # Remove cubic from RTT plot as it skews the plot
        exp_df = exp_df[~((exp_df["cca"] == "cubic") & (exp_df["buf_size_bdp"] == 100))]
        plot_df(
            exp_df,
            "rtt",
            os.path.join(output_dir, f"rtt.{ext}"),
            xkey=agg,
            xlabel=agg_xlabel[agg],
            ylabel="RTT (ms)",
            group="label",
            figsize=figsize,
            use_markers=True,
            legend=False,
            use_entry=True,
        )


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
        '--agg', default="",
        type=str, action='store',
        help='param to aggregate results by')
    args = parser.parse_args()
    set_output_dir(args)

    if os.path.isdir(args.input):
        plot_multi_exp(args.input, args.output, agg=args.agg)
    else:
        plot_single_exp(args.input, args.output)


if __name__ == "__main__":
    main()
