import itertools
import multiprocessing

import comet
import numpy as np
import torch
import sacrebleu

METRIC_NAMES = {
    "chrf": "chrF++",
    "comet": "COMET"
}

def sentence_chrf(hyp, ref):
    if hyp == '' and ref == '':
        return 1.0
    else:
        return sacrebleu.sentence_chrf(hypothesis=hyp,
                                       references=[ref]).score


def get_utils_and_scores_chrf_one(params):
    hypotheses, pseudorefs, example = params
    ref = example["tgt"]
    matrix = np.zeros((len(hypotheses), len(pseudorefs)), dtype=np.float32)
    for i, hyp in enumerate(hypotheses):
        for j, pseudoref in enumerate(pseudorefs):
            matrix[i, j] = sentence_chrf(hyp, pseudoref)
    scores = [sentence_chrf(hyp, ref) for hyp in hypotheses]
    return matrix, scores


def get_utils_and_scores_chrf(data_iterator, num_cpus=None):
    if num_cpus is None:
        num_cpus = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_cpus) as pool:
        for matrix, scores in pool.imap(get_utils_and_scores_chrf_one, data_iterator):
            yield matrix, scores


def get_utils_and_scores_comet(data_iterator,
                               model_name="Unbabel/wmt22-comet-da",
                               device=None,
                               embed_batch_size=100,
                               estimate_batch_size=1000):
    def batched(iterator, batch_size):
        batch = []
        for x in iterator:
            batch.append(x)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = comet.download_model(model_name)
    model = comet.load_from_checkpoint(model_path).eval().to(device)

    for hypotheses, pseudorefs, example in data_iterator:
        all_sequences = list(set(
            hypotheses + pseudorefs + [example["src"], example["tgt"]]))
        # Get embeddings for all sequences
        batch_embeddings = []
        for seq_batch in batched(all_sequences, embed_batch_size):
            inputs = model.encoder.prepare_sample(seq_batch).to(device)
            with torch.no_grad():
                embeddings = model.get_sentence_embedding(
                    inputs["input_ids"], inputs["attention_mask"])
                batch_embeddings.append(embeddings)
        sequence_embeddings = torch.vstack(batch_embeddings)
        sequence_map = dict((seq, i) for i, seq in enumerate(all_sequences))

        # Evaluate each hypotheses with respect to the reference in addition to
        # pseudo-references.
        try:
            ref_index = pseudorefs.index(example["tgt"])
            ref_in_pseudorefs = True
            ref_and_pseudorefs = pseudorefs
        except ValueError:
            ref_index = -1
            ref_in_pseudorefs = False
            ref_and_pseudorefs = pseudorefs + [example["tgt"]]

        utils = np.zeros(len(hypotheses) * len(ref_and_pseudorefs))
        for i, matrix_indices_batch in enumerate(batched(
                itertools.product(range(len(hypotheses)),
                                  range(len(ref_and_pseudorefs))),
                estimate_batch_size)):
            hyp_indices, ref_indices = zip(*matrix_indices_batch)
            src_embeddings = sequence_embeddings[sequence_map[example["src"]]].unsqueeze(
                0).repeat_interleave(len(matrix_indices_batch), 0)
            hyp_embeddings = torch.vstack([
                sequence_embeddings[sequence_map[hypotheses[hyp_index]]]
                for hyp_index in hyp_indices])
            ref_embeddings = torch.vstack([
                sequence_embeddings[sequence_map[ref_and_pseudorefs[ref_index]]]
                for ref_index in ref_indices])

            with torch.no_grad():
                batch_utils = model.estimate(
                    src_embeddings, hyp_embeddings, ref_embeddings)["score"]

            utils[i*estimate_batch_size:i*estimate_batch_size +
                  len(matrix_indices_batch)] = batch_utils.cpu().numpy()
        utils = utils.reshape(len(hypotheses), len(ref_and_pseudorefs))

        if ref_in_pseudorefs:
            yield utils, utils[:, ref_index]
        else:
            yield utils[:, :-1], utils[:, -1]


def get_utils_and_scores(data_iterator, metric):
    """Compute the MBR utility matrix as well as hypotheses scores.

    Assumes hypotheses and pseudorefs are already deduplicated.

    The utility matrices and hypothesis scores are done together because metrics like
    COMET have a slow embedding step and a fast comparison step, and we only want
    to compute the embeddings once.

    Args:
        data_iterator (generator): Yields tuples of (list of str, list of str, str),
            representing the hypotheses, pseudo-references, and reference for one
            data instance.
        metric (str): The utility/evaluation metric. Either "chrf" or "comet".

    Returns:
        (generator): An iterator over (np.array, np.array)
    """
    if metric == "chrf":
        return get_utils_and_scores_chrf(data_iterator)
    elif metric == "comet":
        return get_utils_and_scores_comet(data_iterator)
