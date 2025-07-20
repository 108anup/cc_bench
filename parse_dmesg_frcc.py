import argparse
import os

import matplotlib
import pandas as pd

from common import parse_literal, plot_df, plot_multi_exp, set_output_dir, try_except
from plot_config_light import get_fig_size_paper, get_fig_size_ppt, get_style

ppt = True
ppt = False
style = get_style(use_markers=False, paper=True, use_tex=False)  # paper
get_fig_size = get_fig_size_paper
ext = "pdf"
if ppt:
    style = get_style(use_markers=False, paper=False, use_tex=False)  # ppt
    get_fig_size = get_fig_size_ppt
    ext = "svg"


def parse_params(line: str, prefix):
    # Example output from FRCC when logging is enabled:
    # frcc slot_end flow 1 now 72879611239 cwnd 10 pacing 514506 rtt 80054 mss
    # 1448 slot_start 0 slot_end 72879611239 slot_duration 4160134503 probing 0
    # slots_till_now 1 rtprop 59257 round_min_rtt 80054 round_max_rate 124
    # slot_max_qdel 20797 slot_max_rate 124

    record = {}
    params = line.removeprefix(prefix)
    param_list = params.split(' ')
    for param_name, param_val in zip(param_list[::2], param_list[1::2]):
        record[param_name] = parse_literal(param_val)
    return record


class DmesgLog:

    fpath: str
    df: pd.DataFrame
    df_cwnd: pd.DataFrame
    df_probe: pd.DataFrame

    def __init__(self, fpath):
        self.fpath = fpath
        self.parse_dmesg_log()

    def parse_dmesg_log(self):
        with open(self.fpath, 'r') as f:
            lines = f.read().split('\n')
            # Skip first 2 lines, these are sometimes from the last run.
            lines = lines[2:]

        records_slot = []
        records_cwnd = []
        records_cwnd_event = []
        records_probe = []
        records_probe_state = []
        for line in lines:
            if line.startswith("frcc slot_end "):
                record = parse_params(line, "frcc slot_end ")
                records_slot.append(record)
            elif line.startswith("frcc cwnd_update "):
                record = parse_params(line, "frcc cwnd_update ")
                records_cwnd.append(record)
            elif line.startswith("frcc cwnd_event "):
                record = parse_params(line, "frcc cwnd_event ")
                records_cwnd_event.append(record)
            elif line.startswith("frcc probe_state "):
                record = parse_params(line, "frcc probe_state ")
                records_probe_state.append(record)
            elif line.startswith("frcc probe_start "):
                record = parse_params(line, "frcc probe_start ")
                records_probe.append(record)

        df = pd.DataFrame(records_slot)
        df_cwnd = pd.DataFrame(records_cwnd)
        df_cwnd_event = pd.DataFrame(records_cwnd_event)
        df_probe = pd.DataFrame(records_probe)

        start_tstamp = float(df_cwnd_event['now'].min()) # timestamp is in us

        df_cwnd_event['time'] = (df_cwnd_event['now'] - start_tstamp)/1e6
        df_cwnd['time'] = (df_cwnd['now'] - start_tstamp)/1e6

        if len(df_probe) > 0:
            df_probe['time'] = (df_probe['now'] - start_tstamp)/1e6

        df['time'] = (df['now'] - start_tstamp)/1e6 # time is in s
        df['round_min_qdel_ms'] = (df['round_min_rtt_us'] - df['min_rtprop_us'])/1e3 # convert from us to ms
        df['round_min_rtt_ms'] = df['round_min_rtt_us']/1e3 # convert from us to ms
        df['pacing_pps'] = df['pacing'] / df['mss']

        self.df = df
        self.df_cwnd = df_cwnd
        self.df_cwnd_event = df_cwnd_event
        self.df_probe = df_probe

        # df1 = df_cwnd[["flow", "now", "cwnd", "time"]]
        # df2 = df_probe[["flow", "now", "cwnd", "time"]]
        # self.df_all_cwnd_events = pd.concat([df1, df2], axis=0).sort_values(by="time").reset_index()


@matplotlib.rc_context(rc=style)
def plot_single_exp(fpath, output_dir):
    if fpath.endswith('.dmesg'):
        dl = DmesgLog(fpath)
        df = dl.df
    else:
        raise ValueError(f'Unknown file type {fpath}')

    os.makedirs(output_dir, exist_ok=True)
    fdf = df
    cdf = dl.df_cwnd_event
    figsize = get_fig_size(1, 0.6)

    label="FRCC"
    plot_df(
        cdf,
        "cwnd",
        os.path.join(output_dir, f"cca_cwnd.{ext}"),
        xlabel="Time (s)",
        ylabel="cwnd (pkts)",
        title=label,
        group="flow",
        figsize=figsize,
        legend=False,
    )
    plot_df(
        fdf,
        "pacing_pps",
        os.path.join(output_dir, f"cca_rate.{ext}"),
        xlabel="Time (s)",
        ylabel="pacing rate (packets/sec)",
        title=label,
        group="flow",
        figsize=figsize,
        legend=False,
    )
    plot_df(
        fdf,
        "round_min_rtt_ms",
        os.path.join(output_dir, f"cca_round_min_rtt.{ext}"),
        xlabel="Time (s)",
        ylabel="min_rtt (ms)",
        title=label,
        group="flow",
        figsize=figsize,
        legend=False,
    )
    plot_df(
        fdf,
        "round_min_qdel_ms",
        os.path.join(output_dir, f"cca_round_min_qdel.{ext}"),
        xlabel="Time (s)",
        ylabel="min_qdel (ms)",
        title=label,
        group="flow",
        figsize=figsize,
        legend=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', required=True,
        type=str, action='store',
        help='path to dmesg trace')
    parser.add_argument(
        '-o', '--output', default="",
        type=str, action='store',
        help='path output directory')
    args = parser.parse_args()
    set_output_dir(args)

    if os.path.isdir(args.input):
        plot_multi_exp(args.input, args.output, '.dmesg', plot_single_exp)
    else:
        plot_single_exp(args.input, args.output)


if __name__ == "__main__":
    try_except(main)
