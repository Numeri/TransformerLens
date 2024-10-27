import pytest

from torch.testing import assert_close
import comet
import torch

from transformer_lens import HookedEncoder

MODEL_NAME = "Unbabel/wmt22-cometkiwi-da"


@pytest.fixture(scope="module")
def our_cometkiwi():
    return HookedEncoder.from_pretrained(MODEL_NAME, device="cpu")


@pytest.fixture(scope="module")
def huggingface_cometkiwi():
    return comet.load_from_checkpoint(
        comet.download_model(MODEL_NAME)
    )


@pytest.fixture(scope="module")
def tokenizer(huggingface_cometkiwi):
    return huggingface_cometkiwi.encoder.tokenizer


@pytest.fixture
def hello_world_tokens(tokenizer):
    return tokenizer("Hello, world!", return_tensors="pt")["input_ids"]


def test_full_model(our_cometkiwi, huggingface_cometkiwi, tokenizer):
    sequences = [
        "Es sind viele Ereignisse geschehen, die ihr lieber vergesst, als euch mit ihnen auseinanderzusetzen.",
        "There have been many happenings that you choose to forget, rather than confront.",
    ]
    tokenized = tokenizer(*sequences, return_tensors="pt", padding=True)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    comet_scores = huggingface_cometkiwi.predict([{'src': sequences[1], 'mt': sequences[0]}])
    comet_scores = torch.tensor(comet_scores.scores)
    hf_score = huggingface_cometkiwi.forward(input_ids, attention_mask=attention_mask, output_hidden_states=True)['score']
    our_score = our_cometkiwi.forward(input_ids, one_zero_attention_mask=attention_mask)

    assert_close(comet_scores, our_score, rtol=1.3e-6, atol=4e-5)
    assert_close(comet_scores, hf_score, rtol=1.3e-6, atol=4e-5)
