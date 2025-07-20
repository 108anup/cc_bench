import ast
import itertools
import operator
import os
from functools import reduce
from typing import Any, Callable, Dict, Iterable, List, Literal

import matplotlib.pyplot as plt

from plot_config_light import colors, linestyles, markers

FILE_PATH = os.path.abspath(os.path.realpath(__file__))  # experiments/cc_bench/common.py
CC_BENCH_PATH = os.path.dirname(FILE_PATH)  # PROJECT_ROOT/src
EXPERIMENTS_PATH = os.path.dirname(CC_BENCH_PATH)
LOGS_PATH = os.path.join(EXPERIMENTS_PATH, "data", "logs")
FIGS_PATH = os.path.join(EXPERIMENTS_PATH, "data", "figs")
TRACE_PATH = os.path.join(EXPERIMENTS_PATH, "mahimahi-traces")
RUN_EXPERIMENT_SH = os.path.join(CC_BENCH_PATH, "run_experiment.sh")

CCA_RENAME = {
    'frcc': 'FRCC',
    'genericcc_markovian': 'Copa',
    'bbr': 'BBRv1',
    'bbr2': 'BBRv2',
    'bbr3': 'BBRv3',
    'cubic': 'Cubic',
    'reno': 'Reno',
    'vegas': 'Vegas',
    'astraea': 'Astraea',
}

ENTRY_NUMBER = {
    '_FRCC': 0,
    '_Copa': 1,
    '_BBRv1': 2,
    '_BBRv3': 3,
    '_Cubic': 4,
    '_Reno': 5,
    '_Vegas': 6,
    '_Astraea': 6,

    'FRCC': 0,
    'Copa': 1,
    'BBRv1': 2,
    'BBRv3': 3,
    'Cubic': 4,
    'Reno': 5,
    'Vegas': 6,
    'Astraea': 6,

    "$\\texttt{hops}$": 7,
    "$\\texttt{hops}^2$": 8,
    "RTprop ratio": 8,
}

colors[ENTRY_NUMBER["$\\texttt{hops}$"]] = 'black'
linestyles[ENTRY_NUMBER["$\\texttt{hops}$"]] = 'solid'
colors[ENTRY_NUMBER["$\\texttt{hops}^2$"]] = 'black'
linestyles[ENTRY_NUMBER["$\\texttt{hops}^2$"]] = 'dashed'

# colors[ENTRY_NUMBER["RTprop ratio"]] = 'black'
# linestyles[ENTRY_NUMBER["RTprop ratio"]] = 'dashed'

# Conversions
S_TO_US = 1e6
S_TO_MS = 1e3
MS_TO_US = 1e3
US_TO_MS = 1e-3
US_TO_S = 1e-6
MS_TO_S = 1e-3
NS_TO_MS = 1e-6
NS_TO_US = 1e-3
NS_TO_PS = 1e3
PS_TO_US = 1e-6
US_TO_NS = 1e3

BYTES_TO_BITS = 8

Gb_TO_b = 1e9
Gb_TO_Mb = 1e3
Mb_TO_b = 1e6
Kb_TO_b = 1e3

GiB_TO_B = 1 << 30
MiB_TO_B = 1 << 20
KiB_TO_B = 1 << 10

B_TO_KiB = 1.0 / KiB_TO_B

# TCP MSS          = 1448 bytes
# Ethernet payload = 1500 bytes (= MSS + 20 [IP] + 32 [TCP])
# https://github.com/zehome/MLVPN/issues/26
# MM_PKT_SIZE      = 1504 bytes (= Ethernet payload + 4 [TUN overhead])
# Ethernet MTU     = 1518 bytes (= Ethernet payload + 18 [Ethernet])
# On the wire      = 1538 bytes (= Ethernet MTU + 20 [Preamble + IPG])

MM_PKT_SIZE = 1504
WIRE = 1538
PKT_TO_WIRE = WIRE / MM_PKT_SIZE
ONE_PKT_PER_MS_MBPS_RATE = WIRE * BYTES_TO_BITS * S_TO_MS / Mb_TO_b


# https://stackoverflow.com/questions/20429674/get-tcp-flags-with-scapy
class TCPFlags:
    FIN = 0x01
    SYN = 0x02
    RST = 0x04
    PSH = 0x08
    ACK = 0x10
    URG = 0x20
    ECE = 0x40
    CWR = 0x80
    SYNACK = SYN | ACK


def set_output_dir(args):
    if args.output == "":
        input_abs = os.path.abspath(args.input)
        if input_abs.startswith(LOGS_PATH):
            args.output = input_abs.replace(LOGS_PATH, FIGS_PATH)
    assert args.output != ""
    os.makedirs(args.output, exist_ok=True)


def plot_df(
    df,
    ykey,
    fpath,
    xkey="time",
    xlabel="",
    ylabel="",
    yscale: Literal["linear", "log", "symlog", "logit"] = "linear",
    xscale: Literal["linear", "log", "symlog", "logit"] = "linear",
    title="",
    ylim=(None, None),
    xlim=(None, None),
    group=None,
    figsize=None,
    legend=True,
    use_markers=False,
    legend_ncol=1,
    use_entry=False,
):
    fig, ax = plt.subplots(figsize=figsize)
    if group is not None and group in df.columns:
        i = 0
        for key, grp in df.groupby(group, sort=False):
            entry_number = i if not use_entry else ENTRY_NUMBER[key]
            marker = None
            if use_markers:
                marker = markers[entry_number]
            color = colors[entry_number]
            linestyle = linestyles[entry_number]
            ax.plot(
                grp[xkey],
                grp[ykey],
                label=key,
                marker=marker,
                color=color,
                linestyle=linestyle,
            )
            i += 1
        if legend:
            ax.legend(ncols=legend_ncol)
    else:
        if use_markers:
            ax.plot(df[xkey], df[ykey])
        else:
            ax.step(df[xkey], df[ykey], where="post")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if title != "":
        ax.set_title(title, y=1.0, pad=-8)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.grid(True)
    fig.tight_layout(pad=0.03)
    fig.savefig(fpath)
    plt.close(fig)


def get_exp_string(info):
    """Convert dict of parameter key/value pairs to string representation.
    Useful for populating figure titles/dashboards, and filenames.

    Examples
    --------
    >>> get_info_string({'over_sub': 1, 'node': 128})
    'over_sub[1]-node[128]'
    """
    return '-'.join([f'{k}[{v}]' for k, v in info.items()])


def parse_exp_raw(exp_tag):
    ret = {}
    for param_tag in exp_tag.split('-'):
        param_name = param_tag.split('[')[0]
        param_val = param_tag.split('[')[1][:-1]
        ret[param_name] = parse_literal(param_val)
    return ret


def plot_multi_exp(input_dir: str, output_dir: str,
                   ext: str, plot_single_exp: Callable):
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if (filename.endswith(ext)):
                fpath = os.path.join(root, filename)
                dirpath = fpath.replace(ext, '')
                rel_path = os.path.relpath(dirpath, input_dir)
                this_out_dir = os.path.join(output_dir, rel_path)
                plot_single_exp(fpath, this_out_dir)



def parse_literal(element: str):
    """Converts string to literal if possible, else returns the string

    Examples
    --------
    >>> parse_literal("1.0")
    1.0
    >>> parse_literal("1")
    1
    >>> type(parse_literal("1"))
    <class 'int'>
    >>> type(parse_literal("1.0"))
    <class 'float'>
    """
    if element == "":
        return element
    try:
        return ast.literal_eval(element)
    except ValueError:
        return element


def try_except(function: Callable):
    """This function is useful for debugging in python. If we call a function
    through `try_except` or decorate a function with the `try_except_wrapper`,
    then we get a python interactive debugger anytime an exception is raised in
    the function.

    Examples
    --------
    >>> def f():
    ...     raise ValueError
    >>> try_except(f)
    Traceback (most recent call last):
    ...
    ipdb>

    >>> @try_except_wrapper
    ... def f():
    ...     raise ValueError
    >>> f()
    Traceback (most recent call last):
    ...
    ipdb>
    """
    try:
        return function()
    except Exception:
        import sys
        import traceback

        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)


def try_except_wrapper(function):
    """See documentation of `try_except`"""

    def func_to_return(*args, **kwargs):
        def func_to_try():
            return function(*args, **kwargs)
        return try_except(func_to_try)
    return func_to_return


class PartialParams(dict):
    """For experiments, we want to run different combination of parameter
    configurations. For instance, we may want to run an experiments such that
    when topology size is 128 nodes, we want to explore over subscription ratios
    of 1 and 8, whereas when topology size is 1024 nodes, we want to explore
    over subscription ratios of 4 and 16.

    This class helps manage and manipulate such configuration/inputs for
    experiments to maintain a list of all possible configurations we want to run
    for an experiment. It helps take cartesian product and union of parameter
    lists. The code example below shows how to setup configurations for the
    above use case.

    Examples
    --------
    >>> l1 = [PartialParams(over_sub=1), PartialParams(over_sub=8)]
    >>> l2 = [PartialParams(nodes=128)]
    >>> pdt1 = PartialParams.product(l1, l2)
    >>> pdt1
    [PartialParams(over_sub=1, nodes=128), PartialParams(over_sub=8, nodes=128)]
    >>> l1 = [PartialParams(over_sub=4), PartialParams(over_sub=16)]
    >>> l2 = [PartialParams(nodes=1024)]
    >>> pdt2 = PartialParams.product(l1, l2)
    >>> pdt2
    [PartialParams(over_sub=4, nodes=1024), PartialParams(over_sub=16, nodes=1024)]
    >>> PartialParams.consolidate(pdt1, pdt2)
    [PartialParams(over_sub=1, nodes=128), PartialParams(over_sub=8, nodes=128), PartialParams(over_sub=4, nodes=1024), PartialParams(over_sub=16, nodes=1024)]

    Another use case could be managing configuration when different algorithms
    uses different number of parameters and we want to explore different
    parameter combinations within an algorithm. For instance, say we want to run
    experiment comparing dctcp cc algorithm with ecmp vs. smartt cc algorithm
    with spraying and reps. Below code shows how to generate configurations for
    this set of experiments.

    >>> dctcp = [PartialParams(cc_algo='dctcp', load_balancing_algo="ecmp")]
    >>> smartt = [PartialParams(cc_algo='smartt')]
    >>> smartt_lb_choices = [PartialParams(load_balancing_algo="spraying"), PartialParams(load_balancing_algo="reps")]
    >>> smartt_exps = PartialParams.product(smartt, smartt_lb_choices)
    >>> smartt_exps
    [PartialParams(cc_algo='smartt', load_balancing_algo='spraying'), PartialParams(cc_algo='smartt', load_balancing_algo='reps')]
    >>> all_exps = PartialParams.consolidate(dctcp, smartt_exps)
    >>> all_exps
    [PartialParams(cc_algo='dctcp', load_balancing_algo='ecmp'), PartialParams(cc_algo='smartt', load_balancing_algo='spraying'), PartialParams(cc_algo='smartt', load_balancing_algo='reps')]

    For this simple example, we can obviously directly write the final list of
    configurations. But the product and union functions help as the number of
    parameter chocies/combinations we want to explore grows.
    """

    @staticmethod
    def merge(*l: Iterable):
        return PartialParams(**reduce(operator.ior, l, {}))

    @staticmethod
    def product(*l: List):
        return [PartialParams(PartialParams.merge(*x)) for x in itertools.product(*l)]

    @staticmethod
    def consolidate(*l: Iterable):
        ret = []
        for x in l:
            if isinstance(x, PartialParams):
                ret.append(x)
            else:
                assert isinstance(x, Iterable)
                ret.extend(PartialParams.consolidate(*x))
        return ret
