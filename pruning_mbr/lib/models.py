from transformers import (FSMTForConditionalGeneration, FSMTTokenizer,
                          M2M100ForConditionalGeneration, M2M100Tokenizer)

from pruning_mbr.lib import datasets

DEEN_MODEL_NAME = "facebook/wmt19-de-en"
MULTILINGUAL_MODEL_NAME = "facebook/m2m100_418M"


class UninitializedTargetLanguageError(Exception):
    pass

class M2MModelWrapper(M2M100ForConditionalGeneration):
    """Wrapper for M2M100ForConditionalGeneration.

    This sets the target language automatically so the multilingual model has the
    same interface for generate() as other models."""
    def set_tgt_lang(self, tgt_lang, tokenizer):
        self.forced_bos_token_id = tokenizer.get_lang_id(tgt_lang)

    def generate(self, *args, **kwargs):
        if getattr(self, "forced_bos_token_id") is None:
            raise UninitializedTargetLanguageError(
                "Need to call set_tgt_lang() before using the model.")
        kwargs["forced_bos_token_id"] = self.forced_bos_token_id
        return super().generate(*args, **kwargs)


def load_model_and_tokenizer(src_lang, tgt_lang):
    if src_lang == "de" and tgt_lang == "en":
        model = FSMTForConditionalGeneration.from_pretrained(DEEN_MODEL_NAME)
        tokenizer = FSMTTokenizer.from_pretrained(DEEN_MODEL_NAME)
    else:
        model = M2MModelWrapper.from_pretrained(MULTILINGUAL_MODEL_NAME)
        tokenizer = M2M100Tokenizer.from_pretrained(MULTILINGUAL_MODEL_NAME)
        model.set_tgt_lang(tgt_lang, tokenizer)
        tokenizer.src_lang = src_lang
        tokenizer.tgt_lang = tgt_lang

    return model, tokenizer
