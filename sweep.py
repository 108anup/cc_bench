import argparse
import multiprocessing
import multiprocessing.pool
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from common import (
    MM_PKT_SIZE,
    RUN_EXPERIMENT_SH,
    TRACE_PATH,
    PartialParams,
    get_exp_string,
    try_except_wrapper,
)
from trace_generator import all_trace

N_PROCESSES = 10

ALL_CCAS = [
    PartialParams(cca="frcc"),

    PartialParams(cca="bbr"),
    # PartialParams(cca="bbr3"),

    PartialParams(cca="cubic"),
    PartialParams(cca="reno"),
    # PartialParams(cca="vegas"),

    PartialParams(cca="genericcc_markovian"),
    # PartialParams(cca="genericcc_markovian", cca_params={"COPA_CONST_VELOCITY": True}),
]

class ExperimentType(Enum):
    debug = "debug"
    debug_all = "debug_all"
    sweeps = "sweeps"
    sweep_flows = "sweep_flows"
    sweep_rtprop = "sweep_rtprop"
    sweep_bw = "sweep_bw"
    jitter = "jitter"
    jitter_sweep_bw = "jitter_sweep_bw"
    different_rtt = "different_rtt"
    different_rtt_sweep_bw = "different_rtt_sweep_bw"
    staggered = "staggered"
    sensitivity = "sensitivity"
    all = "all"


@dataclass
class Params:
    bw_ppms: int
    buf_size_bdp: int
    ow_delay_ms: int  # one way delay
    n_flows: int
    jitter_ms: int
    jitter_ppms: int
    cca: str
    cca_params: dict = field(default_factory=dict)
    jitter_shared: bool = True
    rtprop_ratio: int = 1

    experiment_type: ExperimentType = ExperimentType.debug
    jitter_type: str = "ideal"
    staggered_start: bool = False
    different_rtprop: bool = False
    seed: int = 42
    duration_s: int = 300
    overlap_duration_s: int = 180
    iperf_log_interval_s: int = 1
    log_cbr_uplink: bool = True
    log_jitter_uplink: bool = False

    def __post_init__(self):
        # Set smaller buffer for loss based CCAs.
        if self.cca in ["cubic", "reno"]:
            self.buf_size_bdp = 3

    @property
    def bdp_bytes(self):
        return MM_PKT_SIZE * self.bw_ppms * 2 * self.ow_delay_ms

    @property
    def buf_size_bytes(self):
        return self.buf_size_bdp * self.bdp_bytes

    @property
    def group_dir(self):
        """
        Used for directory names
        """
        experiment_type = self.experiment_type.name
        group_tag = get_exp_string(
            {
                "bw_ppms": self.bw_ppms,
                "ow_delay_ms": self.ow_delay_ms,
                "n_flows": self.n_flows,
            }
        )
        return f"{experiment_type}/{group_tag}"

    @property
    def exp_tag(self):
        """
        Used for file names
        """
        vars = {
            "bw_ppms": self.bw_ppms,
            "ow_delay_ms": self.ow_delay_ms,
            "n_flows": self.n_flows,
            "buf_size_bdp": self.buf_size_bdp,
            "cca": self.cca,
            "cca_param_tag": self.cca_param_tag,
        }

        if self.different_rtprop:
            vars.update({"rtprop_ratio": self.rtprop_ratio})

        if self.jitter_type != "ideal":
            vars.update(
                {
                    "jitter_ms": self.jitter_ms,
                    "jitter_ppms": self.jitter_ppms,
                    "jitter_type": self.jitter_type,
                }
            )

        return get_exp_string(vars)

    @property
    def cca_param_tag(self):
        pt = ""
        if self.cca == "frcc":
            if "TCP_FRCC_P_UB_FLOW_COUNT" in self.cca_params:
                pt += f"+flows{self.cca_params['TCP_FRCC_P_UB_FLOW_COUNT']}"
            if "TCP_FRCC_P_UB_SLOTS_PER_ROUND" in self.cca_params:
                pt += f"+slots{self.cca_params['TCP_FRCC_P_UB_SLOTS_PER_ROUND']}"
        elif self.cca == "genericcc_markovian":
            if "COPA_CONST_VELOCITY" in self.cca_params and self.cca_params["COPA_CONST_VELOCITY"]:
                pt += "+cv"
        return pt

    def generate_and_get_jitter_trace(self):
        trace_name = (
            get_exp_string(
                {
                    "seed": self.seed,
                    "bw_ppms": self.bw_ppms,
                    "ow_delay_ms": self.ow_delay_ms,
                    "jitter_ms": self.jitter_ms,
                    "jitter_ppms": self.jitter_ppms,
                    "jitter_type": self.jitter_type,
                    "duration_s": self.duration_s,
                }
            )
        )
        trace_file = f"{trace_name}.trace"
        trace_path = os.path.join(TRACE_PATH, trace_file)
        all_trace(
            self.seed,
            self.bw_ppms,
            self.ow_delay_ms,
            self.jitter_ms,
            self.jitter_ppms,
            self.duration_s,
            trace_path,
            self.jitter_type,
        )
        return trace_path

    @property
    def cbr_uplink_trace_file(self):
        return self.downlink_trace_file

    @property
    def downlink_trace_file(self):
        return os.path.join(TRACE_PATH, f"{self.bw_ppms}ppms.trace")

    @property
    def delay_uplink_trace_file(self):
        if self.jitter_type != "ideal":
            return self.generate_and_get_jitter_trace()
        return self.cbr_uplink_trace_file

    @property
    def is_genericcc(self):
        return self.cca.startswith("genericcc")

    @property
    def env(self):
        ret = {
            "bw_ppms": self.bw_ppms,
            "ow_delay_ms": self.ow_delay_ms,
            "buf_size_bdp": self.buf_size_bdp,
            "n_flows": self.n_flows,
            "cca": self.cca,
            "jitter_shared": self.jitter_shared,
            "rtprop_ratio": self.rtprop_ratio,

            "buf_size_bytes": self.buf_size_bytes,
            "exp_tag": self.exp_tag,
            "cca_param_tag": self.cca_param_tag,
            "downlink_trace_file": self.downlink_trace_file,
            "delay_uplink_trace_file": self.delay_uplink_trace_file,
            "cbr_uplink_trace_file": self.cbr_uplink_trace_file,
            "is_genericcc": self.is_genericcc,
            "group_dir": self.group_dir,

            "jitter_type": self.jitter_type,
            "staggered_start": self.staggered_start,
            "different_rtprop": self.different_rtprop,
            "duration_s": self.duration_s,
            "overlap_duration_s": self.overlap_duration_s,
            "iperf_log_interval_s": self.iperf_log_interval_s,
            "log_cbr_uplink": self.log_cbr_uplink,
            "log_jitter_uplink": self.log_jitter_uplink,
        }
        ret.update(self.cca_params)
        return ret


def convert_env_vars(env: dict):
    return {
        str(key): str(value).lower() if isinstance(value, bool) else str(value)
        for key, value in env.items()
    }


def worker(cmd: str, env: dict):
    start = time.time()
    # print(env)
    print("Starting new process with cmd: ", cmd)
    my_env = os.environ.copy()
    my_env.update(convert_env_vars(env))
    subprocess.run(cmd, env=my_env, shell=True, check=True, executable="/bin/bash")
    end = time.time()
    return cmd, end-start


def run_experiment(
    p: Params,
    port: int,
    main_output_dir: str,
    pool: Optional[multiprocessing.pool.Pool] = None,
):
    # TODO: ideally, we should log the entire set of params.
    # p_file = os.path.join(exp_dir, "cliParams.json")
    # with open(p_file, "w") as f:
    #     json.dump(p.__dict__, f)

    cmd = RUN_EXPERIMENT_SH
    env = {
        **p.env,
        "port": port,
        "outdir": os.path.abspath(main_output_dir),
        "log_dmesg": pool is None and not p.is_genericcc,
    }

    if pool:
        return pool.apply_async(worker, (cmd, env))
    else:
        return worker(cmd, env)


PORT = 7111


def run_combinations(
    pparams: List[PartialParams],
    main_output_dir: str,
    pool=None,
):
    global PORT
    ret = []
    for pp in pparams:
        p = Params(**pp)
        # ^^ This ensures all the required parameters are set (i.e.,
        # non-optional fields in Params are set), and all parameters set in
        # PartialParams are meaningful (i.e., have a field declared in Params)

        this_ret = run_experiment(p, PORT, main_output_dir, pool)
        ret.append(this_ret)

        PORT += p.n_flows  # we consume one port for each flow per experiment

        if pool is not None:
            time.sleep(2)

    return ret


def get_combinations(
    experiment_type: ExperimentType,
):
    cca_choices = ALL_CCAS
    bw_ppms_choices = [PartialParams(bw_ppms=x) for x in range(1, 9)]
    n_flows_choices = [
        PartialParams(n_flows=x) for x in range(1, 9)
        # PartialParams(n_flows=x) for x in range(1, 25)
    ]
    rtprop_choices = [
        PartialParams(ow_delay_ms=x) for x in [5, 10, 15, 25, 40, 50, 100]
    ]

    base_settings = PartialParams(
        bw_ppms=8,
        ow_delay_ms=25,
        buf_size_bdp=100,
        seed=42,
        duration_s=300,
        overlap_duration_s=90,
        experiment_type=experiment_type,
    )

    ideal_settings = PartialParams(
        bw_ppms=4,
        ow_delay_ms=25,
        n_flows=3,
        buf_size_bdp=100,
        seed=42,
        duration_s=300,
        overlap_duration_s=90,
        experiment_type=experiment_type,
        jitter_ms=0,
        jitter_ppms=0,
        jitter_type="ideal",
        staggered_start=False,
        different_rtprop=False,
    )

    if experiment_type == ExperimentType.debug:
        pparams = [
            PartialParams(
                bw_ppms=4,
                ow_delay_ms=25,
                buf_size_bdp=100,
                n_flows=2,
                jitter_ms=0,
                jitter_ppms=0,
                cca="frcc",
                # cca_params={"TCP_FRCC_P_UB_SLOTS_PER_ROUND": 40},
                jitter_type="ideal",
                staggered_start=False,
                different_rtprop=False,
                seed=42,
                duration_s=60,
                overlap_duration_s=180,
            ),
        ]

    elif experiment_type == ExperimentType.debug_all:
        debug_settings = [
            PartialParams(
                experiment_type=ExperimentType.debug_all,
                bw_ppms=4,
                ow_delay_ms=25,
                buf_size_bdp=100,
                n_flows=2,
                jitter_ms=0,
                jitter_ppms=0,
                jitter_type="ideal",
                staggered_start=False,
                different_rtprop=False,
                seed=42,
                duration_s=60,
                overlap_duration_s=180,
            ),
        ]
        pparams = PartialParams.product(
            debug_settings, cca_choices
        )

    elif experiment_type == ExperimentType.sweep_rtprop:
        this_ideal_settings = ideal_settings.copy()
        this_ideal_settings.pop("ow_delay_ms")
        pparams = PartialParams.product(
            [this_ideal_settings], cca_choices, rtprop_choices
        )

    elif experiment_type == ExperimentType.sweep_bw:
        this_ideal_settings = ideal_settings.copy()
        this_ideal_settings.pop("bw_ppms")
        pparams = PartialParams.product(
            [this_ideal_settings], cca_choices, bw_ppms_choices
        )

    elif experiment_type == ExperimentType.sweep_flows:
        this_ideal_settings = ideal_settings.copy()
        this_ideal_settings.pop("n_flows")
        pparams = PartialParams.product(
            [this_ideal_settings],
            cca_choices,
            n_flows_choices,
        )

    elif experiment_type == ExperimentType.jitter_sweep_bw:
        # for rtprop of 32, jitter of 32, burst rate of 32,
        # sweep bw
        this_updates = PartialParams(
            ow_delay_ms=16,
            jitter_ms=32,
            jitter_ppms=32,
            jitter_type="fixed_aggregation",
            jitter_shared=False,
            staggered_start=False,
            different_rtprop=False,
            n_flows=3,
        )
        this_settings = ideal_settings.copy()
        this_settings.update(this_updates)
        this_settings.pop("bw_ppms")
        pparams = PartialParams.product(
            [this_settings], cca_choices, bw_ppms_choices
        )

    elif experiment_type == ExperimentType.jitter:
        # for bw of 4, rtprop of 32, sweep jitter from 8 to 128
        this_updates = PartialParams(
            ow_delay_ms=16,
            jitter_ppms=32,
            jitter_type="fixed_aggregation",
            jitter_shared=False,
            staggered_start=False,
            different_rtprop=False,
            n_flows=3,
        )
        this_settings = ideal_settings.copy()
        this_settings.update(this_updates)
        this_settings.pop("jitter_ms")
        jitter_ms_choices = [PartialParams(jitter_ms=x) for x in [8, 16, 32, 64, 128]]
        pparams = PartialParams.product(
            [this_settings], cca_choices, jitter_ms_choices
        )

    elif experiment_type == ExperimentType.different_rtt_sweep_bw:
        # sweep bandwidth with 3 flows with rtprops 10, 20, 30
        this_updates = PartialParams(
            ow_delay_ms=5,
            rtprop_ratio=2,
            different_rtprop=True,
        )
        this_settings = ideal_settings.copy()
        this_settings.update(this_updates)
        this_settings.pop("bw_ppms")
        pparams = PartialParams.product(
            cca_choices, [this_settings], bw_ppms_choices
        )

    elif experiment_type == ExperimentType.different_rtt:
        # sweep rtt ratio from 2 to 64. lowest is 4, highest is 256
        this_updates = PartialParams(
            ow_delay_ms=2,
            different_rtprop=True,
            n_flows=2,
            bw_ppms=8,
        )
        this_settings = ideal_settings.copy()
        this_settings.update(this_updates)
        rtprop_ratio_choices = [PartialParams(rtprop_ratio=x) for x in [2, 4, 8, 16, 32, 64]]
        pparams = PartialParams.product(cca_choices, [this_settings], rtprop_ratio_choices)

    elif experiment_type == ExperimentType.staggered:
        this_settings = [
            PartialParams(
                jitter_ms=0,
                jitter_ppms=0,
                jitter_type="ideal",
                staggered_start=True,
                different_rtprop=False,
                n_flows=8,
            )
        ]
        pparams = PartialParams.product(
            [base_settings], cca_choices, this_settings
        )

    elif experiment_type == ExperimentType.sensitivity:
        pparams = []
        # raise NotImplementedError

    elif experiment_type == ExperimentType.sweeps:
        pparams = []
        pparams.extend(get_combinations(ExperimentType.sweep_flows))
        pparams.extend(get_combinations(ExperimentType.sweep_bw))
        pparams.extend(get_combinations(ExperimentType.sweep_rtprop))
        pparams.extend(get_combinations(ExperimentType.different_rtt))
        pparams.extend(get_combinations(ExperimentType.different_rtt_sweep_bw))
        pparams.extend(get_combinations(ExperimentType.jitter))
        pparams.extend(get_combinations(ExperimentType.jitter_sweep_bw))
        pparams.extend(get_combinations(ExperimentType.staggered))

    elif experiment_type == ExperimentType.all:
        pparams = []
        for et in ExperimentType:
            if et in [
                ExperimentType.all,
                ExperimentType.debug,
                ExperimentType.staggered,
            ]:
                continue
            pparams.extend(get_combinations(et))

    else:
        raise NotImplementedError

    return pparams


@try_except_wrapper
def main(args):
    pool = None
    if args.parallel:
        pool = multiprocessing.Pool(N_PROCESSES)

    pparams = get_combinations(args.experiment_type)
    run_combinations(pparams, args.output, pool)

    if pool is not None:
        pool.close()
        pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o', '--output', required=True,
        type=str, action='store',
        help='path output directory')
    parser.add_argument(
        '-t', '--experiment-type', required=True,
        type=ExperimentType, action='store',
        choices=list(ExperimentType),
    )
    parser.add_argument(
        '-p', '--parallel',
        action='store_true',
        help='run experiments in parallel'
    )
    args = parser.parse_args()

    main(args)
