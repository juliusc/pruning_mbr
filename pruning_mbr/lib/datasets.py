import datasets

# kken is in the WMT18 dataset, but the Hugging Face repo is missing the data, so
# we omit it.
WMT18_LANGUAGE_PAIRS = ["csen", "deen", "eten", "fien", "ruen", "tren", "zhen"]
WMT18_LANGUAGE_PAIRS += [lp[2:4] + lp[0:2] for lp in WMT18_LANGUAGE_PAIRS]

def load_dataset(language_pair, split):
    if language_pair not in WMT18_LANGUAGE_PAIRS:
        raise ValueError(
            f"Language pair '{language_pair}' not supported by WMT18. "
            f"Supported values are {WMT18_LANGUAGE_PAIRS}")
    if language_pair[0:2] == "en":
        language_pair = language_pair[2:4] + language_pair[0:2]
    validation_subsets = []
    test_subsets = []
    if split == 'validation':
        validation_subsets.append('newstest2017')
    elif split == 'test':
        test_subsets.append('newstest2018')
    else:
        raise ValueError(f"Unknown split '{split}'")

    builder = datasets.load_dataset_builder(
        "wmt18",
        language_pair=(language_pair[0:2], language_pair[2:4]),
        subsets={
            datasets.Split.TRAIN: [],
            datasets.Split.VALIDATION: validation_subsets,
            datasets.Split.TEST: test_subsets,
        }
    )
    builder.download_and_prepare(verification_mode=datasets.VerificationMode.NO_CHECKS)
    return builder.as_dataset(split=split)
