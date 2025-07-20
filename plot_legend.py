import matplotlib
import numpy as np
import pandas as pd

from common import plot_df
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
figsize = get_fig_size()


@matplotlib.rc_context(rc=style)
def main():
    schemes = [
        "FRCC",
        "Copa",
        "BBRv1",
        "BBRv3",
        "Cubic",
        "Reno",
    ]
    records = []
    for label in schemes:
        for x in range(1, 10):
            record = {
                "label": label,
                "x": x,
                "y": x,
            }
            records.append(record)

    df = pd.DataFrame(records)
    figsize = get_fig_size(1, 0.5)
    plot_df(df, "y", "legend.pdf", "x", group="label", figsize=figsize, legend=True, legend_ncol=6, use_markers=True)


if __name__ == "__main__":
    main()
