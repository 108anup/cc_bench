#!/usr/bin/env bash
{
set -xeuo pipefail

LOGS=../data/logs
FIGS=../data/figs
EXP=frcc-nsdi26

# Parsing logs to produce figures
# Figure 19
python parse_pcap.py -i ${LOGS}/${EXP}/sweep_flows/ --agg n_flows
python parse_pcap.py -i ${LOGS}/${EXP}/sweep_bw/ --agg bw_mbps
python parse_pcap.py -i ${LOGS}/${EXP}/sweep_rtprop/ --agg rtprop_ms

# Figure 20
python parse_pcap.py -i ${LOGS}/${EXP}/different_rtt_sweep_bw --agg bw_mbps
python parse_pcap.py -i ${LOGS}/${EXP}/different_rtt/ --agg rtprop_ratio

# Figure 21
python parse_pcap.py -i ${LOGS}/${EXP}/jitter_sweep_bw --agg bw_mbps
python parse_pcap.py -i ${LOGS}/${EXP}/jitter --agg jitter_ms

# Figure 23
python parse_pcap.py -i ${LOGS}/${EXP}/staggered

# Copying all the relevant figures in one place
mkdir -p ${FIGS}/${EXP}/evaluation/sweeps/{bw,rtprop,flows,different_rtt,jitter}
mkdir -p ${FIGS}/${EXP}/evaluation/timeseries

# Figure 19
cp ${FIGS}/${EXP}/sweep_bw/{xput_ratio.pdf,jfi.pdf,rtt.pdf} ${FIGS}/${EXP}/evaluation/sweeps/bw
cp ${FIGS}/${EXP}/sweep_flows/{xput_ratio.pdf,jfi.pdf,rtt.pdf} ${FIGS}/${EXP}/evaluation/sweeps/flows
cp ${FIGS}/${EXP}/sweep_rtprop/{xput_ratio.pdf,jfi.pdf,rtt.pdf} ${FIGS}/${EXP}/evaluation/sweeps/rtprop

# Figure 20
cp ${FIGS}/${EXP}/different_rtt_sweep_bw/{xput_ratio.pdf,jfi.pdf,rtt.pdf} ${FIGS}/${EXP}/evaluation/sweeps/different_rtt
cp ${FIGS}/${EXP}/different_rtt/xput_ratio.pdf ${FIGS}/${EXP}/evaluation/sweeps/different_rtt/rtt_ratio_xput_ratio.pdf

# Figure 21
cp ${FIGS}/${EXP}/jitter_sweep_bw/{xput_ratio.pdf,jfi.pdf,rtt.pdf} ${FIGS}/${EXP}/evaluation/sweeps/jitter
cp ${FIGS}/${EXP}/jitter/xput_ratio.pdf ${FIGS}/${EXP}/evaluation/sweeps/jitter/jitter_xput_ratio.pdf

# Figures 1, 2, 26, 27
for cca in bbr genericcc_markovian frcc; do
  cp "${FIGS}/${EXP}/different_rtt_sweep_bw/bw_ppms[6]-ow_delay_ms[5]-n_flows[3]/bw_ppms[6]-ow_delay_ms[5]-n_flows[3]-buf_size_bdp[100]-cca[${cca}]-cca_param_tag[]-rtprop_ratio[2]/tcpdump_throughput.pdf" ${FIGS}/${EXP}/evaluation/timeseries/diff_rtt_${cca}.pdf
done
for cca in reno cubic; do
  cp "${FIGS}/${EXP}/different_rtt_sweep_bw/bw_ppms[6]-ow_delay_ms[5]-n_flows[3]/bw_ppms[6]-ow_delay_ms[5]-n_flows[3]-buf_size_bdp[3]-cca[${cca}]-cca_param_tag[]-rtprop_ratio[2]/tcpdump_throughput.pdf" ${FIGS}/${EXP}/evaluation/timeseries/diff_rtt_${cca}.pdf
done

for cca in bbr genericcc_markovian frcc; do
  cp "${FIGS}/${EXP}/jitter_sweep_bw/bw_ppms[8]-ow_delay_ms[16]-n_flows[3]/bw_ppms[8]-ow_delay_ms[16]-n_flows[3]-buf_size_bdp[100]-cca[${cca}]-cca_param_tag[]-jitter_ms[32]-jitter_ppms[32]-jitter_type[fixed_aggregation]/tcpdump_throughput.pdf" ${FIGS}/${EXP}/evaluation/timeseries/jitter_${cca}.pdf
done
for cca in reno cubic; do
  cp "${FIGS}/${EXP}/jitter_sweep_bw/bw_ppms[8]-ow_delay_ms[16]-n_flows[3]/bw_ppms[8]-ow_delay_ms[16]-n_flows[3]-buf_size_bdp[3]-cca[${cca}]-cca_param_tag[]-jitter_ms[32]-jitter_ppms[32]-jitter_type[fixed_aggregation]/tcpdump_throughput.pdf" ${FIGS}/${EXP}/evaluation/timeseries/jitter_${cca}.pdf
done

# Figure 23
cp ${FIGS}/${EXP}/staggered/bw_ppms[8]-ow_delay_ms[25]-n_flows[8]/bw_ppms[8]-ow_delay_ms[25]-n_flows[8]-buf_size_bdp[100]-cca[frcc]/tcpdump_throughput.pdf ${FIGS}/${EXP}/evaluation/convergence.pdf

exit 0
}
