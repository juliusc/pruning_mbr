"""Compute false pruning rates on the validation set.

False pruning is when the "oracle" best hypothesis (among a set of hypotheses) under MBR
is pruned by the bootstrapped pruning algorithm. The "oracle" hypothesis is not the
actual oracle best, but the best found using a larger number of pseudo-references.
"""
import argparse
from collections import defaultdict
import logging
from pathlib import Path
from tqdm import tqdm
import sys

import h5py
import numpy as np
import pandas as pd
import seaborn as sns

from pruning_mbr.lib import utils

SEABORN_PALETTE = "colorblind"
SEABORN_STYLE = "darkgrid"
DATA_SPLIT = "validation"

def get_false_pruning_stats(
        utility_matrix, all_num_pseudorefs, confidence_thresholds, num_trials,
        num_bootstraps, false_pruning_counts, rng):
    oracle_index = utility_matrix.sum(axis=1).argmax()
    for _ in range(args.num_trials):
        utility_matrix = rng.permutation(utility_matrix, axis=1)
        for num_pseudorefs in all_num_pseudorefs:
            # y_double_bar is the best hypothesis under the currently available
            # pseudo-references.
            y_double_bar_index = utility_matrix[:, :num_pseudorefs].sum(axis=1).argmax()
            if y_double_bar_index == oracle_index:
                continue
            bootstrap_indices = rng.integers(
                num_pseudorefs, size=args.num_bootstraps * num_pseudorefs)
            oracle_utils = utility_matrix[oracle_index][bootstrap_indices].reshape(
                args.num_bootstraps, num_pseudorefs).sum(axis=1)
            prediction_utils = utility_matrix[y_double_bar_index][bootstrap_indices].reshape(
                args.num_bootstraps, num_pseudorefs).sum(axis=1)
            win_rate = (oracle_utils > prediction_utils).sum() / num_bootstraps
            for confidence_threshold in confidence_thresholds:
                if win_rate < 1 - confidence_threshold:
                    false_pruning_counts[confidence_threshold][num_pseudorefs] += 1


def main(args):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger("pruning_mbr.get_false_pruning_rates")

    sns.set_style(SEABORN_STYLE)

    data_dir = Path(args.data_dir)

    all_num_pseudorefs = utils.parse_num_str(args.num_pseudorefs)
    confidence_thresholds = [float(s) for s in args.confidence_thresholds.split(",")]
    rng = np.random.default_rng(seed=args.seed)

    total_trials = 0
    # Dict of confidence_threshold to a dict of num_pseudorefs to counts.
    false_pruning_counts = defaultdict(lambda: defaultdict(int))
    logger.info("Computing false pruning rates on validation dataset...")
    for utility_matrix in utils.iter_utility_matrices(
            data_dir, DATA_SPLIT, args.metric, do_tqdm=True):
        get_false_pruning_stats(
            utility_matrix, all_num_pseudorefs, confidence_thresholds, args.num_trials,
            args.num_bootstraps, false_pruning_counts, rng)
        total_trials += args.num_trials

    # Create the figure.
    data_frame_rows = []
    for confidence_threshold in confidence_thresholds:
        for num_pseudorefs in all_num_pseudorefs:
            count = (false_pruning_counts[confidence_threshold][num_pseudorefs] /
                     total_trials)
            data_frame_rows.append([confidence_threshold, num_pseudorefs, count])
    columns = ["Confidence threshold", "Number of pseudo-references",
               "False pruning rate"]
    output_path = data_dir / f"false_pruning_rates.{args.metric}.csv"
    pd.DataFrame(data_frame_rows, columns=columns).to_csv(output_path)

    columns = ["Confidence threshold", "Number of pseudo-references",
               "False pruning rate"]
    data_frame = pd.DataFrame(data_frame_rows, columns=columns)
    data_output_path = data_dir / f"false_pruning_rates.{args.metric}.csv"
    logger.info(f"Saved data to {data_output_path}")
    data_frame.to_csv(data_output_path)

    plot = sns.lineplot(data=data_frame, x=columns[1], y=columns[2], hue=columns[0],
                        palette=SEABORN_PALETTE)
    plot_output_path = data_dir / f"false_pruning_rates.{args.metric}.png"
    plot.get_figure().savefig(plot_output_path)
    logger.info(f"Saved figure to {plot_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "data_dir", type=str,
        help=("Data directory produced by running generate.py. "
              "The figure will be written to this directory."))

    parser.add_argument(
        "metric", type=str, help="Utility metric. Either 'chrf' or 'comet'.")

    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    parser.add_argument(
        "--num_pseudorefs", type=str,
        default="4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,56,64", help=(
            "Comma-separated list of integers representing the number of "
            "pseudo-references used to measure a false pruning rates. Hyphenated "
            "inclusive ranges are supported, e.g. '4-32'."))

    parser.add_argument(
        "--confidence_thresholds", type=str, default="0.8,0.9,0.95,0.98,0.99",
        help="Comma-separated list of confidence values.")

    parser.add_argument(
        "--num_bootstraps", type=int, default=500, help="Number of bootstrap trials.")

    parser.add_argument(
        "--num_trials", type=int, default=10,
        help="Number of random trials to perform for each example.")

    args = parser.parse_args()
    main(args)
