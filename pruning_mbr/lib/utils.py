import h5py
import json
import numpy as np
from tqdm import tqdm

CHRF_PSEUDOREFS_SCHEDULE = [16, 32, 64, 128, 256]
COMET_PSEUDOREFS_SCHEDULE = [8, 16, 32, 64, 128, 256]


def parse_num_str(num_str):
    nums = []
    for substr in num_str.split(","):
        if "-" in substr:
            start, end = (int(n) for n in substr.split('-'))
            nums.extend(range(start, end + 1))
        else:
            nums.append(int(substr))
    return nums


def get_default_pseudorefs_schedule(metric):
    if metric == "chrf":
        return CHRF_PSEUDOREFS_SCHEDULE
    elif metric == "comet":
        return COMET_PSEUDOREFS_SCHEDULE
    else:
        raise ValueError(f"Unknown metric '{metric}'")


def iterate_utility_matrices(hypotheses_path, pseudorefs_path, scores_path):
    with h5py.File(scores_path) as scores_file:
        for example_index, (hyps_line, pseudorefs_line) in enumerate(
                zip(open(hypotheses_path), open(pseudorefs_path))):
            hyps = json.loads(hyps_line)
            pseudorefs, pseudoref_counts = zip(*json.loads(pseudorefs_line))
            utils = np.frombuffer(
                scores_file["utilities"][example_index], dtype=np.float32
            ).reshape((len(hyps), -1))
            columns = []
            for index, count in enumerate(pseudoref_counts):
                columns.append(utils[:, index:index+1].repeat(count, axis=1))
            yield np.concatenate(columns, axis=1)


def iter_utility_matrices(data_dir, data_split, metric, do_tqdm=False):
    hypotheses_path = data_dir / f"hypotheses.{data_split}.jsonl"
    pseudorefs_path = data_dir / f"pseudorefs.{data_split}.jsonl"
    scores_path = data_dir / f"scores.{data_split}.{metric}.hdf5"

    def iter():
        with h5py.File(scores_path) as scores_file:
            for example_index, (hyps_line, pseudorefs_line) in enumerate(
                    zip(open(hypotheses_path), open(pseudorefs_path))):
                hyps = json.loads(hyps_line)
                _, pseudoref_counts = zip(*json.loads(pseudorefs_line))
                utils = np.frombuffer(
                    scores_file["utilities"][example_index], dtype=np.float32
                ).reshape((len(hyps), -1))
                columns = []
                for index, count in enumerate(pseudoref_counts):
                    columns.append(utils[:, index:index+1].repeat(count, axis=1))
                yield np.concatenate(columns, axis=1)

    if do_tqdm:
        return tqdm(iter(), total=h5py.File(scores_path)["utilities"].shape[0])
    else:
        return iter()


def decode_bottom_pct(utility_matrix, prune_pct, pseudorefs_schedule):
    """Do MBR decoding with naive bottom-percent pruning."""
    orig_indices = np.arange(utility_matrix.shape[0])
    num_util_calls = 0
    last_num_pseudorefs = 0
    for num_pseudorefs in pseudorefs_schedule:
        if utility_matrix.shape[0] == 1:
            break
        num_hyps_to_keep = max(
            1, np.floor(utility_matrix.shape[0] * (1 - prune_pct)).astype(int))
        kept_indices = (-utility_matrix[:, :num_pseudorefs].sum(axis=1)).argsort()[
            :num_hyps_to_keep]

        num_util_calls += (num_pseudorefs - last_num_pseudorefs) * \
            utility_matrix.shape[0]
        last_num_pseudorefs = num_pseudorefs

        orig_indices = orig_indices[kept_indices]
        utility_matrix = utility_matrix[kept_indices]

    if utility_matrix.shape[0] == 1:
        prediction_index = orig_indices[0]
    else:
        prediction_index = orig_indices[
            utility_matrix[:, :pseudorefs_schedule[-1]].mean(axis=1).argmax()]
    return prediction_index, num_util_calls


def decode_bootstrap(utility_matrix, confidence_threshold, pseudorefs_schedule,
                     num_bootstraps, rng):
    """Do MBR decoding with bootstrap pruning."""
    orig_indices = np.arange(utility_matrix.shape[0])
    num_util_calls = 0
    last_num_pseudorefs = 0
    for num_pseudorefs in pseudorefs_schedule:
        if utility_matrix.shape[0] == 1:
            break
        # y_double_bar is the best hypothesis under the currently available
        # pseudo-references.
        y_double_bar_index = utility_matrix[:, :num_pseudorefs].sum(axis=1).argmax()
        bootstrap_indices = rng.integers(
            num_pseudorefs, size=num_bootstraps * num_pseudorefs)
        bootstrapped_utilities = utility_matrix[:, bootstrap_indices].reshape(
            utility_matrix.shape[0], num_bootstraps, num_pseudorefs).sum(axis=2)
        win_rates = (bootstrapped_utilities >=
                     bootstrapped_utilities[y_double_bar_index]).mean(axis=1)

        num_util_calls += ((num_pseudorefs - last_num_pseudorefs) *
                           utility_matrix.shape[0])
        last_num_pseudorefs = num_pseudorefs

        kept_indices = np.asarray(win_rates > 1 - confidence_threshold).nonzero()[0]
        orig_indices = orig_indices[kept_indices]
        utility_matrix = utility_matrix[kept_indices]

    if utility_matrix.shape[0] == 1:
        prediction_index = orig_indices[0]
    else:
        prediction_index = orig_indices[
            utility_matrix[:, :pseudorefs_schedule[-1]].mean(axis=1).argmax()]
    return prediction_index, num_util_calls


def decode_standard(utility_matrix, num_pseudorefs):
    prediction_index = utility_matrix[:, :num_pseudorefs].sum(axis=1).argmax()
    num_util_calls = utility_matrix.shape[0] * num_pseudorefs
    return prediction_index, num_util_calls
