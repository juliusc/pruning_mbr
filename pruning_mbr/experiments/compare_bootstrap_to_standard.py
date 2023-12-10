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
DATA_SPLIT = "test"


def main(args):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger("pruning_mbr.generate")

    sns.set_style(SEABORN_STYLE)

    data_dir = Path(args.data_dir)
    if args.pseudorefs_schedule is None:
        utils.get_default_pseudorefs_schedule(args.metric)
    else:
        pseudorefs_schedule = [int(s) for s in args.pseudorefs_schedule.split(",")]
    confidence_thresholds = [float(s) for s in args.confidence_thresholds.split(",")]
    rng = np.random.default_rng(seed=args.seed)

    pruning_methods = ["Bootstrap", "Standard"]
    stats = dict((pruning_method,
                  defaultdict(lambda: {"Score": 0.0,
                                       "Accuracy": 0,
                                       "Reciprocal rank": 0.0,
                                       "Utility calls": 0}))
                 for pruning_method in pruning_methods)
    scores_path = data_dir / f"scores.{DATA_SPLIT}.{args.metric}.hdf5"

    total_trials = 0
    logger.info("Computing stats on test dataset...")
    with h5py.File(scores_path) as scores_file:
        for i, utility_matrix in enumerate(utils.iter_utility_matrices(
                data_dir, DATA_SPLIT, args.metric, do_tqdm=True)):
            oracle_sort_indices = (-utility_matrix.sum(axis=1)).argsort()
            oracle_index = oracle_sort_indices[0]
            oracle_rankings = oracle_sort_indices.argsort()
            for _ in range(args.num_trials):
                utility_matrix = rng.permutation(utility_matrix, axis=1)
                for pruning_method in pruning_methods:
                    if pruning_method == "Bootstrap":
                        pruning_hyperparams = confidence_thresholds
                    else:
                        pruning_hyperparams = bottom_pruning_pcts
                    for pruning_hyperparam in pruning_hyperparams:
                        if pruning_method == "Bootstrap":
                            prediction_index, num_util_calls = prune_bootstrap(
                                utility_matrix, pruning_hyperparam, pseudorefs_schedule,
                                args.num_bootstraps, rng)
                        else:
                            prediction_index, num_util_calls = prune_bottom_pct(
                                utility_matrix, pruning_hyperparam, pseudorefs_schedule)

                        inner_stats_dict = stats[pruning_method][pruning_hyperparam]
                        score = scores_file["eval_scores"][i][prediction_index]
                        inner_stats_dict["Score"] += score
                        inner_stats_dict["Utility calls"] += num_util_calls
                        if prediction_index == oracle_index:
                            inner_stats_dict["Accuracy"] += 1
                        reciprocal_rank = 1 / (oracle_rankings[prediction_index] + 1)
                        inner_stats_dict["Reciprocal rank"] += reciprocal_rank
                        total_trials += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "data_dir", type=str, help="Data directory produced by running generate.py.")

    parser.add_argument(
        "metric", type=str, help="Utility metric. Either 'chrf' or 'comet'.")

    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    parser.add_argument(
        "--pseudorefs_schedule", type=str, help=(
            "Comma-separated list of integers representing the pseudo-reference size "
            "schedule. If not provided, set to 16,32,64,128,256 for chrF++ and "
            "8,16,32,64,128,256 for COMET as per the paper."))

    parser.add_argument(
        "--confidence_thresholds", type=str, default="0.9,0.99",
        help="Comma-separated list of confidence values for bootstrap pruning.")

    parser.add_argument(
        "--num_bootstraps", type=int, default=500, help="Number of bootstrap trials.")

    parser.add_argument(
        "--num_trials", type=int, default=10,
        help="Number of random trials to perform for each example.")

    args = parser.parse_args()
    main(args)
