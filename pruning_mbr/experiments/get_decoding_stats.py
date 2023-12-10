"""Compare the quality of the pruning bootstrap method versus the baseline described
in the paper, which just prunes the bottom percentage of hypotheses by utility per
step.
"""
import argparse
from collections import defaultdict
import logging
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import seaborn as sns

from pruning_mbr.lib import utils

SEABORN_PALETTE = "colorblind"
SEABORN_STYLE = "darkgrid"


def main(args):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger("pruning_mbr.get_decoding_stats")

    data_dir = Path(args.data_dir)
    if args.pseudorefs_schedule is None:
        pseudorefs_schedule = utils.get_default_pseudorefs_schedule(args.metric)
    else:
        pseudorefs_schedule = [int(s) for s in args.pseudorefs_schedule.split(",")]
    confidence_thresholds = [float(s) for s in args.confidence_thresholds.split(",")]
    bottom_pruning_pcts = [float(s) for s in args.bottom_pruning_pcts.split(",")]
    rng = np.random.default_rng(seed=args.seed)

    pruning_methods = ["Bootstrap", "Bottom-percent", "Standard"]
    stats = dict((pruning_method,
                  defaultdict(lambda: {"Score": 0.0,
                                       "Accuracy": 0,
                                       "Reciprocal rank": 0.0,
                                       "Utility calls": 0}))
                 for pruning_method in pruning_methods)
    scores_path = data_dir / f"scores.{args.data_split}.{args.metric}.hdf5"

    total_trials = 0
    logger.info(f"Computing decoding stats on {args.data_split} dataset...")
    with h5py.File(scores_path) as scores_file:
        for i, utility_matrix in enumerate(utils.iter_utility_matrices(
                data_dir, args.data_split, args.metric, do_tqdm=True)):
            oracle_sort_indices = (-utility_matrix.sum(axis=1)).argsort()
            oracle_index = oracle_sort_indices[0]
            oracle_rankings = oracle_sort_indices.argsort()
            for _ in range(args.num_trials):
                utility_matrix = rng.permutation(utility_matrix, axis=1)
                for pruning_method in pruning_methods:
                    if pruning_method == "Bootstrap":
                        pruning_hyperparams = confidence_thresholds
                    elif pruning_method == "Bottom-percent":
                        pruning_hyperparams = bottom_pruning_pcts
                    else:
                        pruning_hyperparams = [None]
                    for pruning_hyperparam in pruning_hyperparams:
                        if pruning_method == "Bootstrap":
                            prediction_index, num_util_calls = utils.decode_bootstrap(
                                utility_matrix, pruning_hyperparam, pseudorefs_schedule,
                                args.num_bootstraps, rng)
                        elif pruning_method == "Bottom-percent":
                            prediction_index, num_util_calls = utils.decode_bottom_pct(
                                utility_matrix, pruning_hyperparam, pseudorefs_schedule)
                        else:
                            prediction_index, num_util_calls = utils.decode_standard(
                                utility_matrix, pseudorefs_schedule[-1])

                        inner_stats_dict = stats[pruning_method][pruning_hyperparam]
                        score = scores_file["eval_scores"][i][prediction_index]
                        inner_stats_dict["Score"] += score
                        inner_stats_dict["Utility calls"] += num_util_calls
                        if prediction_index == oracle_index:
                            inner_stats_dict["Accuracy"] += 1
                        reciprocal_rank = 1 / (oracle_rankings[prediction_index] + 1)
                        inner_stats_dict["Reciprocal rank"] += reciprocal_rank
                total_trials += 1

    columns = ["Pruning method", "Measure", "Value", "Utility calls"]
    data_frame_rows = []
    for pruning_method in pruning_methods:
        for pruning_hyperparam in stats[pruning_method]:
            for measure in ["Score", "Accuracy", "Reciprocal rank"]:
                inner_stats_dict = stats[pruning_method][pruning_hyperparam]
                num_util_calls = inner_stats_dict["Utility calls"] / total_trials
                value = inner_stats_dict[measure] / total_trials
                data_frame_rows.append([pruning_method, measure, value, num_util_calls])
    output_path = data_dir / f"decoding_stats.{args.data_split}.{args.metric}.csv"
    pd.DataFrame(data_frame_rows, columns=columns).to_csv(output_path)
    logger.info(f"Saved data to {output_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "data_dir", type=str,
        help=("Data directory produced by running generate.py. "
              "The figure will be written to this directory."))

    parser.add_argument(
        "data_split", type=str, help="Data split. Either 'validation' or 'test'.")

    parser.add_argument(
        "metric", type=str, help="Utility metric. Either 'chrf' or 'comet'.")

    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    parser.add_argument(
        "--pseudorefs_schedule", type=str, help=(
            "Comma-separated list of integers representing the pseudo-reference size "
            "schedule. If not provided, set to 16,32,64,128,256 for chrF++ and "
            "8,16,32,64,128,256 for COMET as per the paper."))

    parser.add_argument(
        "--confidence_thresholds", type=str, default="0.8,0.9,0.95,0.98,0.99",
        help="Comma-separated list of confidence values for bootstrap pruning.")

    parser.add_argument(
        "--bottom_pruning_pcts", type=str,
        default=("0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,"
                 "0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95"),
        help=("Comma-separated list of percentage values for bottom-percentage "
              "baseline pruning method."))

    parser.add_argument(
        "--num_bootstraps", type=int, default=500, help="Number of bootstrap trials.")

    parser.add_argument(
        "--num_trials", type=int, default=10,
        help="Number of random trials to perform for each example.")

    args = parser.parse_args()
    main(args)
