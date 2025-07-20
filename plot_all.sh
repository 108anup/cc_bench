#!/usr/bin/env bash
{
set -xeuo pipefail

LOGS=../data/logs
FIGS=../data/figs
EXP=frcc-nsdi26

python parse_pcap.py -i ${LOGS}/${EXP}/sweep_bw/ --agg bw_mbps
python parse_pcap.py -i ${LOGS}/${EXP}/sweep_flows/ --agg n_flows
python parse_pcap.py -i ${LOGS}/${EXP}/sweep_rtprop/ --agg rtprop_ms

python parse_pcap.py -i ${LOGS}/${EXP}/jitter_sweep_bw --agg bw_mbps
python parse_pcap.py -i ${LOGS}/${EXP}/jitter --agg jitter_ms

python parse_pcap.py -i ${LOGS}/${EXP}/different_rtt/ --agg rtprop_ratio
python parse_pcap.py -i ${LOGS}/${EXP}/different_rtt_sweep_bw --agg bw_mbps

mkdir -p ${FIGS}/${EXP}/evaluation/sweeps/{bw,rtprop,flows,different_rtt,jitter}
mkdir -p ${FIGS}/${EXP}/evaluation/timeseries

cp ${FIGS}/${EXP}/sweep_bw/{xput_ratio.pdf,jfi.pdf,rtt.pdf} ${FIGS}/${EXP}/evaluation/sweeps/bw
cp ${FIGS}/${EXP}/sweep_flows/{xput_ratio.pdf,jfi.pdf,rtt.pdf} ${FIGS}/${EXP}/evaluation/sweeps/flows
cp ${FIGS}/${EXP}/sweep_rtprop/{xput_ratio.pdf,jfi.pdf,rtt.pdf} ${FIGS}/${EXP}/evaluation/sweeps/rtprop

cp ${FIGS}/${EXP}/jitter_sweep_bw/{xput_ratio.pdf,jfi.pdf,rtt.pdf} ${FIGS}/${EXP}/evaluation/sweeps/jitter
cp ${FIGS}/${EXP}/jitter/xput_ratio.pdf ${FIGS}/${EXP}/evaluation/sweeps/jitter/jitter_xput_ratio.pdf

cp ${FIGS}/${EXP}/different_rtt_sweep_bw/{xput_ratio.pdf,jfi.pdf,rtt.pdf} ${FIGS}/${EXP}/evaluation/sweeps/different_rtt
cp ${FIGS}/${EXP}/different_rtt/xput_ratio.pdf ${FIGS}/${EXP}/evaluation/sweeps/different_rtt/rtt_ratio_xput_ratio.pdf

for cca in bbr genericcc_markovian ndd bbr3; do
  cp "${FIGS}/${EXP}/different_rtt_sweep_bw/bw_ppms[6]-ow_delay_ms[5]-n_flows[3]/bw_ppms[6]-ow_delay_ms[5]-n_flows[3]-buf_size_bdp[100]-cca[${cca}]-cca_param_tag[]-rtprop_ratio[2]/tcpdump_throughput.pdf" ${FIGS}/${EXP}/evaluation/timeseries/diff_rtt_${cca}.pdf
done
for cca in reno cubic; do
  cp "${FIGS}/${EXP}/different_rtt_sweep_bw/bw_ppms[6]-ow_delay_ms[5]-n_flows[3]/bw_ppms[6]-ow_delay_ms[5]-n_flows[3]-buf_size_bdp[3]-cca[${cca}]-cca_param_tag[]-rtprop_ratio[2]/tcpdump_throughput.pdf" ${FIGS}/${EXP}/evaluation/timeseries/diff_rtt_${cca}.pdf
done

for cca in bbr genericcc_markovian ndd bbr3; do
  cp "${FIGS}/${EXP}/jitter_sweep_bw/bw_ppms[8]-ow_delay_ms[16]-n_flows[3]/bw_ppms[8]-ow_delay_ms[16]-n_flows[3]-buf_size_bdp[100]-cca[${cca}]-cca_param_tag[]-jitter_ms[32]-jitter_ppms[32]-jitter_type[fixed_aggregation]/tcpdump_throughput.pdf" ${FIGS}/${EXP}/evaluation/timeseries/jitter_${cca}.pdf
done
for cca in reno cubic; do
  cp "${FIGS}/${EXP}/jitter_sweep_bw/bw_ppms[8]-ow_delay_ms[16]-n_flows[3]/bw_ppms[8]-ow_delay_ms[16]-n_flows[3]-buf_size_bdp[3]-cca[${cca}]-cca_param_tag[]-jitter_ms[32]-jitter_ppms[32]-jitter_type[fixed_aggregation]/tcpdump_throughput.pdf" ${FIGS}/${EXP}/evaluation/timeseries/jitter_${cca}.pdf
done

exit 0
}
