import argparse
import logging
import sys

import pandas as pd
import seaborn as sns

SEABORN_PALETTE = "colorblind"
SEABORN_STYLE = "darkgrid"

def main(args):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger("pruning_mbr.plot_pruning_comparison")

    sns.set_style(SEABORN_STYLE)

    data_frame = pd.read_csv(args.input_path)
    # Drop rows for standard MBR
    data_frame = data_frame[data_frame["Pruning method"] != "Standard"]

    # Focus the x-axis on the range of utility calls from bootstrap pruning.
    bootstrap_num_util_calls = data_frame[
        data_frame["Pruning method"] == "Bootstrap"]["Utility calls"]

    plot = sns.relplot(
        data=data_frame, x="Utility calls", y="Value", row="Measure",
        hue="Pruning method", kind="line", palette=SEABORN_PALETTE,
        facet_kws=dict(sharey=False))
    plot.set(xlim=(min(bootstrap_num_util_calls) * 0.9,
                   max(bootstrap_num_util_calls) * 1.1))
    plot.savefig(args.output_path)
    logger.info(f"Saved figure to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_path", type=str,
        help="Path containing the output of get_decoding_stats.py.")

    parser.add_argument(
        "output_path", type=str, help="Path to write the resulting figure to."
    )

    args = parser.parse_args()
    main(args)
