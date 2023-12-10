"""
Generate all prerequisites for experiments: hypotheses, pseudo-references, utility
matrices, and sentence-level evaluation scores for each hypothesis.
"""
import argparse
import logging
import json
import sys

from collections import Counter
from pathlib import Path
from tqdm import tqdm

import h5py
import numpy as np
import torch

from transformers import GenerationConfig
from pruning_mbr.lib import datasets, models, metrics

MAX_GENERATION_LENGTH = 1024
EPSILON_CUTOFF = 0.02


def get_donefile_path(path):
    return path.with_suffix('.DONE')


def generate_hypotheses(model, tokenizer, dataset, num_hypotheses, output_path):
    """Generate hypotheses with beam top-k."""
    hypothesis_generation_config = GenerationConfig(
        max_length=MAX_GENERATION_LENGTH,
        num_beams=num_hypotheses,
        num_return_sequences=num_hypotheses
    )
    with open(output_path, 'w') as output_file:
        for row in tqdm(dataset):
            result = model.generate(
                tokenizer.encode(
                    row["src"], return_tensors="pt"
                ).to(model.device),
                generation_config=hypothesis_generation_config)
            # De-duplicated list of hypotheses
            hypotheses = list(set(
                tokenizer.batch_decode(result, skip_special_tokens=True)))
            output_file.write(f"{json.dumps(hypotheses)}\n")


def generate_pseudorefs(model, tokenizer, dataset, num_pseudorefs, output_path):
    """Generate pseudo-references with epsilon sampling."""
    pseudoref_generation_config = GenerationConfig(
        max_length=MAX_GENERATION_LENGTH,
        num_return_sequences=num_pseudorefs,
        do_sample=True,
        epsilon_cutoff=EPSILON_CUTOFF
    )
    with open(output_path, 'w') as output_file:
        for row in tqdm(dataset):
            result = model.generate(
                tokenizer.encode(
                    row["src"], return_tensors="pt"
                ).to(model.device),
                generation_config=pseudoref_generation_config)
            # List of tuples of (pseudorefs, count).
            pseudoref_data = list(Counter(
                tokenizer.batch_decode(result, skip_special_tokens=True)).items())
            output_file.write(f"{json.dumps(pseudoref_data)}\n")


def compute_scores(metric,
                   hyps_path,
                   pseudoref_path,
                   dataset,
                   scores_output_path):
    """Compute MBR utilities and evaluation scores for all hypotheses."""
    def get_hyp_pseudoref_iterator():
        for hyps_line, pseudorefs_line, example in tqdm(
                zip(open(hyps_path), open(pseudoref_path), dataset),
                total=len(dataset)):
            hyps = json.loads(hyps_line)
            pseudorefs = [item[0] for item in json.loads(pseudorefs_line)]
            yield hyps, pseudorefs, example

    with h5py.File(scores_output_path, "w") as output_file:
        utils_dataset = output_file.create_dataset(
            "utilities", (len(dataset),), dtype=h5py.vlen_dtype(np.dtype('float32')))
        scores_dataset = output_file.create_dataset(
            "eval_scores", (len(dataset),), dtype=h5py.vlen_dtype(np.dtype('float32')))
        for i, (utils, scores) in enumerate(
                metrics.get_utils_and_scores(get_hyp_pseudoref_iterator(), metric)):
            utils_dataset[i] = utils.reshape(-1)
            scores_dataset[i] = scores


def main(args):
    # Logging format borrowed from Fairseq.
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger("pruning_mbr.generate")

    src_lang = args.language_pair[0:2]
    tgt_lang = args.language_pair[2:4]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Potential TODO: lazy load these because they may not be needed.
    model, tokenizer = models.load_model_and_tokenizer(src_lang, tgt_lang)
    model.to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    for split in ["validation", "test"]:
        def dataset_map_fn(example):
            return {"src": example["translation"][src_lang],
                    "tgt": example["translation"][tgt_lang]}

        dataset = datasets.load_dataset(
            args.language_pair, split).map(dataset_map_fn)
        if args.subset:
            dataset = dataset.select(range(args.subset))

        # For each step of generation, check for the existence of a done file.
        # To redo a step, delete the associated done file.

        # Generate hypotheses
        hyp_output_path = output_dir / f"hypotheses.{split}.jsonl"
        hyp_donefile_path = get_donefile_path(hyp_output_path)
        if not hyp_donefile_path.exists():
            logger.info(f"Generating hypotheses for {split} split...")
            generate_hypotheses(model, tokenizer, dataset,
                                args.num_hypotheses, hyp_output_path)
            hyp_donefile_path.touch()
        else:
            logger.info(
                f"Skipping hypothesis generation for {split} split: file exists.")

        # Generate pseudo-references
        pseudoref_output_path = output_dir / f"pseudorefs.{split}.jsonl"
        pseudoref_donefile_path = get_donefile_path(pseudoref_output_path)
        if not pseudoref_donefile_path.exists():
            logger.info(f"Generating pseudo-references for {split} split...")
            generate_pseudorefs(model, tokenizer, dataset,
                                args.num_pseudorefs, pseudoref_output_path)
            pseudoref_donefile_path.touch()
        else:
            logger.info(
                f"Skipping pseudo-reference generation for {split} split: file exists.")

        # Generate hypothesis utilities and evaluation scores
        for metric in args.metrics.split(','):
            if metric not in metrics.METRIC_NAMES:
                raise ValueError(f"Unknown metric '{metric}'")
            metric_name = metrics.METRIC_NAMES[metric]

            scores_output_path = output_dir / f"scores.{split}.{metric}.hdf5"
            scores_donefile_path = get_donefile_path(scores_output_path)

            if not scores_donefile_path.exists():
                logger.info(
                    f"Generating utilities and evaluation scores for {split} "
                    f"split and metric {metric_name}...")
                compute_scores(
                    metric, hyp_output_path, pseudoref_output_path, dataset,
                    scores_output_path)
                scores_donefile_path.touch()
            else:
                logger.info(
                    f"Skipping utilities and evaluation scores for {split} split and "
                    f"metric {metric_name}: file exists.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "output_dir", help="Output directory. Will be created if doesn't exist.")

    parser.add_argument(
        "language_pair",
        help=("One of the available language pairs in WMT18. "
              "Format like 'deen', 'enet'."))

    parser.add_argument(
        "--num_hypotheses", type=int, default=256,
        help="Number of hypotheses to generate and beam size for top-k.")

    parser.add_argument(
        "--num_pseudorefs", type=int, default=1024,
        help="Number of pseudo-references.")

    parser.add_argument(
        "--metrics", type=str, default='chrf,comet',
        help=("Comma-separated list of utility/evaluation metrics. Only 'chrf' for "
              "chrF++ and 'comet' for COMET."))

    parser.add_argument(
        "--subset", type=int, help=(
            "Only process the first n examples from the dataset. Used for testing."))

    args = parser.parse_args()
    main(args)
